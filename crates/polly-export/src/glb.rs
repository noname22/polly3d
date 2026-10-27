//! Binary glTF 2.0 export. Textures are embedded as PNG with nearest/linear
//! samplers matching each texture's filter setting. The layout for characters
//! with swappable gear is described in `docs/gltf-extras.md`:
//!
//! - Every object with bones gets joint nodes under its node (one per bone, in
//!   rest pose). Each armature has one skin, shared by every mesh skinned to it;
//!   those meshes are written in the armature's space and placed under the
//!   armature node. Vertices without usable weights use an `<armature>_root` joint.
//! - Bone-attached objects are children of their joint node.
//! - Parts hidden by some variant are split into their own child nodes
//!   (`<object>.<part>`), so they can be hidden independently.
//! - Non-default variants and parts hidden by default variants carry
//!   `KHR_node_visibility` (used, not required); variants, parts and the slot
//!   table are described in `extras.polly`.
//! - The static pose is not baked: joints are in rest pose, clips become
//!   animations (bone channels as full joint-node TRS, morph weights merged per
//!   node, loop settings in `extras.polly`).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use polly_core::anim::{BoneId, Channel, Interpolation, Target};
use polly_core::glam::{Mat4, Quat, Vec3, Vec4};
use polly_core::mesh::VertId;
use polly_core::scene::{Filter, ObjectId, Scene};
use serde_json::{Value, json};

const NEAREST: u32 = 9728;
const LINEAR: u32 = 9729;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
const UNSIGNED_SHORT: u32 = 5123;
const UNSIGNED_INT: u32 = 5125;

#[derive(Default)]
struct Builder {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Builder {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let mut v = json!({"buffer": 0, "byteOffset": self.bin.len(), "byteLength": bytes.len()});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.bin.extend_from_slice(bytes);
        self.views.push(v);
        self.views.len() - 1
    }

    /// Float vertex attribute accessor.
    fn floats(&mut self, data: &[f32], comps: usize, bounds: bool) -> usize {
        self.float_accessor(data, comps, bounds, Some(ARRAY_BUFFER))
    }

    /// Float accessor (`comps` = 1, 2, 3, 4 or 16) with an optional buffer view target.
    fn float_accessor(&mut self, data: &[f32], comps: usize, bounds: bool, target: Option<u32>) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = self.view(&bytes, target);
        let ty = match comps {
            1 => "SCALAR",
            2 => "VEC2",
            3 => "VEC3",
            16 => "MAT4",
            _ => "VEC4",
        };
        let mut a = json!({"bufferView": view, "componentType": FLOAT, "count": data.len() / comps, "type": ty});
        if bounds {
            let mut lo = vec![f32::MAX; comps];
            let mut hi = vec![f32::MIN; comps];
            for c in data.chunks(comps) {
                for i in 0..comps {
                    lo[i] = lo[i].min(c[i]);
                    hi[i] = hi[i].max(c[i]);
                }
            }
            a["min"] = json!(lo);
            a["max"] = json!(hi);
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }

    fn joints(&mut self, data: &[u16]) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(ARRAY_BUFFER));
        self.accessors.push(json!({"bufferView": view, "componentType": UNSIGNED_SHORT, "count": data.len() / 4, "type": "VEC4"}));
        self.accessors.len() - 1
    }

    fn indices(&mut self, idx: &[u32]) -> usize {
        let bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(ELEMENT_ARRAY_BUFFER));
        self.accessors.push(json!({"bufferView": view, "componentType": UNSIGNED_INT, "count": idx.len(), "type": "SCALAR"}));
        self.accessors.len() - 1
    }
}

/// Vertex buffers for one material of one mesh.
#[derive(Default)]
struct Prim {
    pos: Vec<f32>,
    nrm: Vec<f32>,
    uv: Vec<f32>,
    col: Vec<f32>,
    joints: Vec<u16>,
    weights: Vec<f32>,
    /// MikkTSpace tangents (xyz + handedness), only for normal-mapped materials.
    tangent: Vec<f32>,
    /// Source vertex of each output vertex (for morph deltas).
    vids: Vec<VertId>,
    idx: Vec<u32>,
    /// Corner attributes plus, for skinned / morphed meshes, the vertex id
    /// (which determines joints, weights and morph deltas).
    dedup: HashMap<([u32; 12], Option<VertId>), u32>,
}

impl Prim {
    #[allow(clippy::too_many_arguments)]
    fn vertex(&mut self, vid: VertId, per_vertex: bool, p: [f32; 3], n: [f32; 3], uv: [f32; 2], c: [f32; 4], skin: Option<&([u16; 4], [f32; 4])>) -> u32 {
        let mut key = [0u32; 12];
        for (k, v) in p.iter().chain(&n).chain(&uv).chain(&c).enumerate() {
            key[k] = v.to_bits();
        }
        let key = (key, per_vertex.then_some(vid));
        if let Some(i) = self.dedup.get(&key) {
            return *i;
        }
        let i = (self.pos.len() / 3) as u32;
        self.pos.extend(p);
        self.nrm.extend(n);
        self.uv.extend(uv);
        self.col.extend(c);
        self.vids.push(vid);
        if let Some((j, w)) = skin {
            self.joints.extend(j);
            self.weights.extend(w);
        }
        self.dedup.insert(key, i);
        i
    }
}

/// MikkTSpace view of a primitive's triangle list (one tangent per corner).
struct MikkPrim<'a> {
    p: &'a Prim,
    out: Vec<[f32; 4]>,
}

impl bevy_mikktspace::Geometry for MikkPrim<'_> {
    fn num_faces(&self) -> usize {
        self.p.idx.len() / 3
    }
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let i = self.p.idx[face * 3 + vert] as usize * 3;
        self.p.pos[i..i + 3].try_into().unwrap()
    }
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let i = self.p.idx[face * 3 + vert] as usize * 3;
        self.p.nrm[i..i + 3].try_into().unwrap()
    }
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let i = self.p.idx[face * 3 + vert] as usize * 2;
        self.p.uv[i..i + 2].try_into().unwrap()
    }
    fn set_tangent(&mut self, ts: Option<bevy_mikktspace::TangentSpace>, face: usize, vert: usize) {
        self.out[face * 3 + vert] = ts.map_or([1.0, 0.0, 0.0, 1.0], |t| t.tangent_encoded());
    }
}

impl Prim {
    /// Generate MikkTSpace tangents per triangle corner, then split vertices
    /// whose corners disagree on the tangent (mirrored UV islands).
    fn add_tangents(&mut self) {
        let mut m = MikkPrim { p: self, out: vec![[1.0, 0.0, 0.0, 1.0]; self.idx.len()] };
        if bevy_mikktspace::generate_tangents(&mut m).is_err() {
            return;
        }
        let corner_t = m.out;
        let old = std::mem::take(self);
        let mut remap: HashMap<(u32, [u32; 4]), u32> = HashMap::new();
        for (c, &vi) in old.idx.iter().enumerate() {
            let mut t = corner_t[c];
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            if len > 1e-8 && len.is_finite() {
                t = [t[0] / len, t[1] / len, t[2] / len, if t[3] < 0.0 { -1.0 } else { 1.0 }];
            } else {
                t = [1.0, 0.0, 0.0, 1.0];
            }
            let key = (vi, t.map(f32::to_bits));
            let ni = *remap.entry(key).or_insert_with(|| {
                let v = vi as usize;
                self.pos.extend_from_slice(&old.pos[v * 3..v * 3 + 3]);
                self.nrm.extend_from_slice(&old.nrm[v * 3..v * 3 + 3]);
                self.uv.extend_from_slice(&old.uv[v * 2..v * 2 + 2]);
                self.col.extend_from_slice(&old.col[v * 4..v * 4 + 4]);
                if !old.joints.is_empty() {
                    self.joints.extend_from_slice(&old.joints[v * 4..v * 4 + 4]);
                    self.weights.extend_from_slice(&old.weights[v * 4..v * 4 + 4]);
                }
                self.vids.push(old.vids[v]);
                self.tangent.extend(t);
                (self.vids.len() - 1) as u32
            });
            self.idx.push(ni);
        }
    }
}

/// glTF material JSON and the extensions it uses.
fn material_json(m: &polly_core::scene::Material, tex_index: &HashMap<u32, usize>) -> (Value, Vec<&'static str>) {
    let tex = |t: Option<u32>| t.and_then(|t| tex_index.get(&t)).copied();
    let mut exts: Vec<&'static str> = Vec::new();
    let mut pbr = json!({"baseColorFactor": m.base_color, "metallicFactor": m.metallic, "roughnessFactor": m.roughness});
    if let Some(ti) = tex(m.texture) {
        pbr["baseColorTexture"] = json!({"index": ti});
    }
    if let Some(ti) = tex(m.metallic_roughness_texture) {
        pbr["metallicRoughnessTexture"] = json!({"index": ti});
    }
    let mut mj = json!({"name": m.name, "pbrMetallicRoughness": pbr, "doubleSided": m.double_sided});
    if let Some(ti) = tex(m.normal_texture) {
        mj["normalTexture"] = json!({"index": ti, "scale": m.normal_scale});
    }
    if let Some(ti) = tex(m.occlusion_texture) {
        mj["occlusionTexture"] = json!({"index": ti, "strength": m.occlusion_strength});
    }
    if m.emissive.iter().any(|c| *c != 0.0) || m.emissive_texture.is_some() {
        // An emissive texture alone needs a non-zero factor to show.
        let f = if m.emissive.iter().all(|c| *c == 0.0) { [1.0; 3] } else { m.emissive };
        mj["emissiveFactor"] = json!(f);
        if let Some(ti) = tex(m.emissive_texture) {
            mj["emissiveTexture"] = json!({"index": ti});
        }
    }
    if let Some(c) = m.alpha_cutoff {
        mj["alphaMode"] = json!("MASK");
        mj["alphaCutoff"] = json!(c);
    } else if m.blend {
        mj["alphaMode"] = json!("BLEND");
    }
    let mut e: Vec<(&'static str, Value)> = Vec::new();
    if m.unlit {
        e.push(("KHR_materials_unlit", json!({})));
    }
    if m.emissive_strength != 1.0 && mj.get("emissiveFactor").is_some() {
        e.push(("KHR_materials_emissive_strength", json!({"emissiveStrength": m.emissive_strength})));
    }
    if let Some(c) = m.clearcoat {
        e.push(("KHR_materials_clearcoat", json!({"clearcoatFactor": c.factor, "clearcoatRoughnessFactor": c.roughness})));
    }
    if let Some(a) = m.anisotropy {
        e.push(("KHR_materials_anisotropy", json!({"anisotropyStrength": a.strength, "anisotropyRotation": a.rotation})));
    }
    if let Some(t) = m.transmission {
        e.push(("KHR_materials_transmission", json!({"transmissionFactor": t.factor})));
        e.push(("KHR_materials_ior", json!({"ior": t.ior})));
        if t.thickness > 0.0 {
            e.push(("KHR_materials_volume", json!({"thicknessFactor": t.thickness})));
        }
    }
    if let Some(sp) = m.specular {
        e.push(("KHR_materials_specular", json!({"specularFactor": sp.factor, "specularColorFactor": sp.color})));
    }
    if let Some(sh) = &m.shader {
        mj["extras"] = json!({"polly": {"shader": {"name": sh.name, "params": sh.params}}});
    }
    if !e.is_empty() {
        exts = e.iter().map(|(k, _)| *k).collect();
        mj["extensions"] = Value::Object(e.into_iter().map(|(k, v)| (k.to_string(), v)).collect());
    }
    (mj, exts)
}

/// Joint nodes of one exported skeleton (an armature).
struct SkinInfo {
    /// Joint node of each bone.
    joint_node: HashMap<BoneId, usize>,
    /// Rest local transform (relative to the parent joint / armature node) of each bone.
    local: HashMap<BoneId, (Vec3, Quat)>,
    /// Bones in joint order (parent first).
    bones: Vec<BoneId>,
    /// Extra identity joint for vertices without usable weights.
    root_joint: Option<usize>,
    /// The glTF skin, once a skinned mesh needs it.
    skin: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Objects to export (with their descendants); `None` = all visible.
    /// Armatures and bone parents of exported objects are always included.
    pub objects: Option<Vec<ObjectId>>,
    /// Subdivision level for every object; `None` = each object's export level.
    pub subdivision: Option<u8>,
}

/// Top-4, normalized joint indices and weights of a vertex; `None` when it has no usable weight.
fn vertex_skin(list: Option<&Vec<(BoneId, f32)>>, joint_idx: &HashMap<BoneId, u16>) -> Option<([u16; 4], [f32; 4])> {
    let mut l: Vec<(u16, f32)> =
        list?.iter().filter(|(_, w)| w.is_finite() && *w > 1e-6).filter_map(|(b, w)| Some((*joint_idx.get(b)?, *w))).collect();
    l.sort_by(|a, b| b.1.total_cmp(&a.1));
    l.truncate(4);
    let sum: f32 = l.iter().map(|(_, w)| w).sum();
    if l.is_empty() || sum <= 1e-6 {
        return None;
    }
    let mut j = [0u16; 4];
    let mut w = [0f32; 4];
    for (i, (ji, wi)) in l.iter().enumerate() {
        j[i] = *ji;
        w[i] = wi / sum;
    }
    Some((j, w))
}

fn trs(m: Mat4) -> Value {
    let (s, r, t) = m.to_scale_rotation_translation();
    json!({"translation": t.to_array(), "rotation": r.normalize().to_array(), "scale": s.to_array()})
}

fn add_child(nodes: &mut [Value], parent: usize, child: usize) {
    let children = nodes[parent].as_object_mut().unwrap().entry("children").or_insert(json!([]));
    children.as_array_mut().unwrap().push(json!(child));
}

type HideGroups = BTreeMap<ObjectId, BTreeMap<BTreeSet<String>, BTreeSet<polly_core::mesh::FaceId>>>;

/// Node name of an object's faces hidden by exactly these hide rules.
fn group_node_name(object: &str, parts: &BTreeSet<String>) -> String {
    format!("{object}.{}", parts.iter().cloned().collect::<Vec<_>>().join("+"))
}

pub fn export_glb(scene: &Scene, opts: &ExportOptions) -> Result<Vec<u8>> {
    // Subdivision modifiers: export the subdivided surfaces (skin weights,
    // morphs and hide groups follow the cage).
    let subdivided;
    let scene = if opts.subdivision.unwrap_or(0) > 0 || scene.objects.values().any(|o| o.subdivision.is_some_and(|s| s.export() > 0)) {
        subdivided = polly_core::subdiv::apply_for_export(scene, opts.subdivision);
        &subdivided
    } else {
        scene
    };
    let mut b = Builder::default();
    let mut selected: BTreeSet<ObjectId> = match &opts.objects {
        Some(list) => list.iter().copied().collect(),
        None => scene.objects.iter().filter(|(_, o)| o.visible).map(|(id, _)| *id).collect(),
    };
    // Descendants, plus what exported objects need: their armature and bone parents.
    loop {
        let before = selected.len();
        let need: Vec<ObjectId> = selected
            .iter()
            .flat_map(|id| {
                let o = &scene.objects[id];
                let arm = o.skin.map(|s| s.armature);
                let bone_parent = o.parent_bone.and(o.parent);
                arm.into_iter().chain(bone_parent)
            })
            .chain(scene.objects.iter().filter(|(_, o)| o.parent.is_some_and(|p| selected.contains(&p))).map(|(id, _)| *id))
            .filter(|id| scene.objects.contains_key(id))
            .collect();
        selected.extend(need);
        if selected.len() == before {
            break;
        }
    }
    // Textures / images / samplers: one per texture used in any slot.
    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut samplers = Vec::new();
    let mut tex_index: HashMap<u32, usize> = HashMap::new();
    let used_materials: BTreeSet<u32> = selected
        .iter()
        .flat_map(|id| scene.objects[id].mesh.faces.values().map(|f| f.material).collect::<Vec<_>>())
        .collect();
    let material = |mid: &u32| scene.materials.get(mid).cloned().unwrap_or_else(|| scene.materials[&0].clone());
    for mid in &used_materials {
        for (_, tid) in material(mid).textures() {
            if tex_index.contains_key(&tid) {
                continue;
            }
            let Some(t) = scene.textures.get(&tid) else { continue };
            let png = crate::png(t)?;
            let view = b.view(&png, None);
            images.push(json!({"bufferView": view, "mimeType": "image/png", "name": t.name}));
            let f = if t.filter == Filter::Nearest { NEAREST } else { LINEAR };
            let sampler = json!({"magFilter": f, "minFilter": f, "wrapS": 10497, "wrapT": 10497});
            let si = samplers.iter().position(|s| *s == sampler).unwrap_or_else(|| {
                samplers.push(sampler);
                samplers.len() - 1
            });
            textures.push(json!({"source": images.len() - 1, "sampler": si}));
            tex_index.insert(tid, textures.len() - 1);
        }
    }

    let mut materials = Vec::new();
    let mut mat_index: BTreeMap<u32, usize> = BTreeMap::new();
    // Materials whose primitives need tangents (normal-mapped).
    let mut needs_tangents: BTreeSet<u32> = BTreeSet::new();
    let mut extensions_used: BTreeSet<&str> = BTreeSet::new();
    for mid in &used_materials {
        let m = material(mid);
        let (mj, exts) = material_json(&m, &tex_index);
        extensions_used.extend(exts);
        if m.normal_texture.is_some_and(|t| tex_index.contains_key(&t)) {
            needs_tangents.insert(*mid);
        }
        materials.push(mj);
        mat_index.insert(*mid, materials.len() - 1);
    }

    let mut meshes = Vec::new();
    let mut nodes: Vec<Value> = Vec::new();
    let mut skins = Vec::new();
    let mut node_of: BTreeMap<ObjectId, usize> = BTreeMap::new();
    let mut skin_info: HashMap<ObjectId, SkinInfo> = HashMap::new();
    // Nodes animated by an object's morph weights (its own node and its part nodes).
    let mut morph_nodes: BTreeMap<ObjectId, Vec<usize>> = BTreeMap::new();
    // Faces hidden by variants, grouped by which hide rules cover them: each
    // group becomes its own node so it can be hidden independently.
    let hide_groups: HideGroups = polly_core::variants::hide_groups(scene);

    // 1. A node per object (filled in below) and joint nodes per skeleton.
    for id in &selected {
        node_of.insert(*id, nodes.len());
        nodes.push(json!({"name": scene.objects[id].name}));
    }
    for id in &selected {
        let Some(sk) = scene.objects[id].skeleton.as_ref().filter(|sk| !sk.bones.is_empty()) else { continue };
        let bones = sk.ordered();
        let rest: HashMap<BoneId, Mat4> = bones.iter().map(|b| (*b, sk.bones[b].rest_matrix())).collect();
        let mut info = SkinInfo { joint_node: HashMap::new(), local: HashMap::new(), bones: bones.clone(), root_joint: None, skin: None };
        for bid in &bones {
            let bone = &sk.bones[bid];
            let parent = bone.parent.filter(|p| rest.contains_key(p));
            let parent_rest = parent.map_or(Mat4::IDENTITY, |p| rest[&p]);
            let (_, r, tr) = (parent_rest.inverse() * rest[bid]).to_scale_rotation_translation();
            let r = r.normalize();
            let n = nodes.len();
            nodes.push(json!({"name": bone.name, "translation": tr.to_array(), "rotation": r.to_array()}));
            let pn = parent.map_or(node_of[id], |p| info.joint_node[&p]);
            add_child(&mut nodes, pn, n);
            info.joint_node.insert(*bid, n);
            info.local.insert(*bid, (tr, r));
        }
        skin_info.insert(*id, info);
    }

    // 2. Place object nodes. Skinned meshes live in their armature's space under
    // the armature node; bone-attached objects under their joint; the rest keep
    // Polly's world placement relative to their parent's node.
    // Which exported skeleton skins each object (it must have weights).
    let skinned: BTreeMap<ObjectId, ObjectId> = selected
        .iter()
        .filter_map(|id| {
            let src = scene.skeleton_source(*id)?;
            (skin_info.contains_key(&src) && !scene.objects[id].mesh.weights.is_empty()).then_some((*id, src))
        })
        .collect();
    let skinned_to = |id: &ObjectId| skinned.get(id).copied();
    let export_world = |id: &ObjectId| -> Mat4 {
        match skinned_to(id) {
            Some(arm) if arm != *id => scene.world_matrix(arm),
            _ => scene.world_matrix(*id),
        }
    };
    let mut roots = Vec::new();
    for id in &selected {
        let o = &scene.objects[id];
        let n = node_of[id];
        let joint = o.parent_bone.zip(o.parent).and_then(|(b, p)| skin_info.get(&p)?.joint_node.get(&b).copied());
        let (parent_node, local) = match (skinned_to(id), joint) {
            (Some(arm), _) if arm != *id => (Some(node_of[&arm]), Mat4::IDENTITY),
            (_, Some(j)) => (Some(j), o.transform.matrix()),
            _ => match o.parent.filter(|p| selected.contains(p)) {
                Some(p) => (Some(node_of[&p]), export_world(&p).inverse() * scene.world_matrix(*id)),
                None => (None, scene.world_matrix(*id)),
            },
        };
        for (k, v) in trs(local).as_object().unwrap() {
            nodes[n][k] = v.clone();
        }
        match parent_node {
            Some(p) => add_child(&mut nodes, p, n),
            None => roots.push(n),
        }
    }

    // Joints for vertices without usable weights, per skeleton.
    for id in &selected {
        let Some(arm) = skinned_to(id) else { continue };
        let info = &skin_info[&arm];
        let joint_idx: HashMap<BoneId, u16> = info.bones.iter().enumerate().map(|(i, b)| (*b, i as u16)).collect();
        let mesh = &scene.objects[id].mesh;
        let unweighted = mesh.faces.values().flat_map(|f| f.verts.iter()).any(|v| vertex_skin(mesh.weights.get(v), &joint_idx).is_none());
        if unweighted && info.root_joint.is_none() {
            let n = nodes.len();
            nodes.push(json!({"name": format!("{}_root", scene.objects[&arm].name)}));
            add_child(&mut nodes, node_of[&arm], n);
            skin_info.get_mut(&arm).unwrap().root_joint = Some(n);
        }
    }

    // 3. Meshes: one per object, plus one per hideable part on its own child node.
    for id in &selected {
        let o = &scene.objects[id];
        let mesh = &o.mesh;
        let arm = skinned_to(id);
        let to_skel = if arm.is_some() { scene.to_skeleton_space(*id) } else { Mat4::IDENTITY };
        let normal_m = to_skel.inverse().transpose();
        let normals = mesh.corner_normals();
        let mut vert_skin: HashMap<VertId, ([u16; 4], [f32; 4])> = HashMap::new();
        if let Some(arm) = arm {
            let info = &skin_info[&arm];
            let joint_idx: HashMap<BoneId, u16> = info.bones.iter().enumerate().map(|(i, b)| (*b, i as u16)).collect();
            let root = info.bones.len() as u16;
            for v in mesh.faces.values().flat_map(|f| f.verts.iter()) {
                let s = vertex_skin(mesh.weights.get(v), &joint_idx).unwrap_or(([root, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]));
                vert_skin.insert(*v, s);
            }
            if info.skin.is_none() {
                let mut joints: Vec<usize> = info.bones.iter().map(|bn| info.joint_node[bn]).collect();
                let sk = scene.objects[&arm].skeleton.as_ref().unwrap();
                let mut ibm: Vec<f32> = info.bones.iter().flat_map(|bn| sk.bones[bn].rest_matrix().inverse().to_cols_array()).collect();
                if let Some(r) = info.root_joint {
                    joints.push(r);
                    ibm.extend(Mat4::IDENTITY.to_cols_array());
                }
                let acc = b.float_accessor(&ibm, 16, false, None);
                skins.push(json!({"name": scene.objects[&arm].name, "joints": joints, "inverseBindMatrices": acc, "skeleton": node_of[&arm]}));
                skin_info.get_mut(&arm).unwrap().skin = Some(skins.len() - 1);
            }
        }
        let skin_index = arm.and_then(|a| skin_info[&a].skin);
        let skinned = arm.is_some();
        let morphed = !mesh.morphs.is_empty();
        let group_of: HashMap<polly_core::mesh::FaceId, &BTreeSet<String>> =
            hide_groups.get(id).into_iter().flatten().flat_map(|(k, fs)| fs.iter().map(move |f| (*f, k))).collect();
        // Faces grouped by hide group (None = never hidden), then by material.
        let mut groups: BTreeMap<Option<BTreeSet<String>>, BTreeMap<u32, Prim>> = BTreeMap::new();
        for (&fid, f) in &mesh.faces {
            let pts: Vec<Vec3> = mesh.face_positions(f).iter().map(|p| to_skel.transform_point3(*p)).collect();
            let group = group_of.get(&fid).map(|k| (*k).clone());
            let prim = groups.entry(group).or_default().entry(f.material).or_default();
            let vidx: Vec<u32> = (0..f.len())
                .map(|i| {
                    let c = &f.corners[i];
                    let v = f.verts[i];
                    let n = normal_m.transform_vector3(normals[&(fid, i)]).normalize_or_zero();
                    prim.vertex(v, skinned || morphed, pts[i].to_array(), n.to_array(), c.uv.to_array(), c.color.to_array(), vert_skin.get(&v))
                })
                .collect();
            for [a, bb, c] in polly_core::geom::triangulate_polygon(&pts) {
                prim.idx.extend([vidx[a], vidx[bb], vidx[c]]);
            }
        }
        for (group, prims) in groups {
            let mut pj = Vec::new();
            for (mid, mut p) in prims {
                if p.idx.is_empty() {
                    continue;
                }
                if needs_tangents.contains(&mid) {
                    p.add_tangents();
                }
                let mut attrs = json!({
                    "POSITION": b.floats(&p.pos, 3, true),
                    "NORMAL": b.floats(&p.nrm, 3, false),
                    "TEXCOORD_0": b.floats(&p.uv, 2, false),
                });
                if !p.tangent.is_empty() {
                    attrs["TANGENT"] = json!(b.floats(&p.tangent, 4, false));
                }
                if p.col.iter().any(|c| (*c - 1.0).abs() > 1e-6) {
                    attrs["COLOR_0"] = json!(b.floats(&p.col, 4, false));
                }
                if skinned {
                    attrs["JOINTS_0"] = json!(b.joints(&p.joints));
                    attrs["WEIGHTS_0"] = json!(b.floats(&p.weights, 4, false));
                }
                let mut prim = json!({"attributes": attrs, "indices": b.indices(&p.idx), "material": mat_index[&mid], "mode": 4});
                if morphed {
                    let targets: Vec<Value> = mesh
                        .morphs
                        .iter()
                        .map(|m| {
                            let d: Vec<f32> = p
                                .vids
                                .iter()
                                .flat_map(|v| to_skel.transform_vector3(m.offsets.get(v).copied().unwrap_or(Vec3::ZERO)).to_array())
                                .collect();
                            json!({"POSITION": b.floats(&d, 3, true)})
                        })
                        .collect();
                    prim["targets"] = json!(targets);
                }
                pj.push(prim);
            }
            if pj.is_empty() {
                continue;
            }
            let mut mj = json!({"name": o.name, "primitives": pj});
            if morphed {
                let w: Vec<f32> = mesh.morphs.iter().map(|m| o.morph_weights.get(&m.name).copied().unwrap_or(0.0)).collect();
                let names: Vec<&str> = mesh.morphs.iter().map(|m| m.name.as_str()).collect();
                mj["weights"] = json!(w);
                mj["extras"] = json!({"targetNames": names});
            }
            meshes.push(mj);
            let node = match &group {
                None => node_of[id],
                Some(parts) => {
                    let n = nodes.len();
                    nodes.push(json!({
                        "name": group_node_name(&o.name, parts),
                        "extras": {"polly": {"object": o.name, "parts": parts}},
                    }));
                    add_child(&mut nodes, node_of[id], n);
                    n
                }
            };
            nodes[node]["mesh"] = json!(meshes.len() - 1);
            if let Some(si) = skin_index {
                nodes[node]["skin"] = json!(si);
            }
            if morphed {
                morph_nodes.entry(*id).or_default().push(node);
            }
            if let Some(parts) = &group {
                // Hidden by default when a default variant hides any of it.
                let hidden = scene.slots.values().any(|s| {
                    s.variants.iter().any(|v| v.name == s.default && v.hides.iter().any(|h| h.object == *id && parts.contains(&h.part)))
                });
                if hidden {
                    nodes[node]["extensions"] = json!({"KHR_node_visibility": {"visible": false}});
                    extensions_used.insert("KHR_node_visibility");
                }
            }
        }
    }

    // 4. Variants: tag variant nodes; hide the non-default ones.
    for (slot, s) in &scene.slots {
        let members: BTreeSet<ObjectId> = s.variants.iter().flat_map(|v| v.objects.iter().copied()).collect();
        for oid in members {
            let Some(&n) = node_of.get(&oid) else { continue };
            let variants: Vec<&str> = s.variants.iter().filter(|v| v.objects.contains(&oid)).map(|v| v.name.as_str()).collect();
            let default = variants.contains(&s.default.as_str());
            nodes[n]["extras"] = json!({"polly": {"slot": slot, "variants": variants, "default": default}});
            if !default {
                nodes[n]["extensions"] = json!({"KHR_node_visibility": {"visible": false}});
                extensions_used.insert("KHR_node_visibility");
            }
        }
    }

    // Animations.
    let mut animations = Vec::new();
    for clip in scene.clips.values() {
        let mut samplers_j = Vec::new();
        let mut channels_j = Vec::new();
        let mut add = |b: &mut Builder, times: &[f32], values: &[f32], comps: usize, interp: Interpolation, node: usize, path: &str| {
            let input = b.float_accessor(times, 1, true, None);
            let output = b.float_accessor(values, comps, false, None);
            let interp = match interp {
                Interpolation::Step => "STEP",
                Interpolation::Linear => "LINEAR",
                Interpolation::Smooth | Interpolation::Ease => "CUBICSPLINE",
            };
            samplers_j.push(json!({"input": input, "output": output, "interpolation": interp}));
            channels_j.push(json!({"sampler": samplers_j.len() - 1, "target": {"node": node, "path": path}}));
        };
        let mut morph_channels: BTreeMap<ObjectId, Vec<&Channel>> = BTreeMap::new();
        for ch in &clip.channels {
            if ch.keys.is_empty() {
                continue;
            }
            let object = ch.target.object();
            let Some(&obj_node) = node_of.get(&object) else { continue };
            let times: Vec<f32> = ch.keys.iter().map(|k| k.time).collect();
            let bone_target = |bone: &BoneId| skin_info.get(&object).and_then(|s| Some((s.joint_node.get(bone)?, s.local[bone])));
            // Output values: per key, or [in-tangent, value, out-tangent] for cubic
            // curves. `val` maps a Polly value to the glTF one, `tan` maps a tangent
            // (the linear part of the same map).
            let cyclic = clip.is_cyclic(ch);
            let curve = |val: &dyn Fn(Vec4) -> Vec<f32>, tan: &dyn Fn(Vec4) -> Vec<f32>| -> Vec<f32> {
                if ch.interpolation.is_cubic() {
                    let v = ch.aligned_values();
                    let m = ch.tangents(cyclic);
                    v.iter().zip(&m).flat_map(|(v, m)| [tan(*m), val(*v), tan(*m)].concat()).collect()
                } else {
                    ch.keys.iter().flat_map(|k| val(k.value)).collect()
                }
            };
            let xyz = |v: Vec4| v.truncate().to_array().to_vec();
            let quat = |v: Vec4| Quat::from_vec4(v).to_array().to_vec();
            match &ch.target {
                Target::BoneTranslation { bone, .. } => {
                    let Some((&n, (lt, lr))) = bone_target(bone) else { continue };
                    let v = curve(&|v| (lt + lr * v.truncate()).to_array().to_vec(), &|m| (lr * m.truncate()).to_array().to_vec());
                    add(&mut b, &times, &v, 3, ch.interpolation, n, "translation");
                }
                Target::BoneRotation { bone, .. } => {
                    let Some((&n, (_, lr))) = bone_target(bone) else { continue };
                    let v = curve(&|v| (lr * Quat::from_vec4(v)).normalize().to_array().to_vec(), &|m| (lr * Quat::from_vec4(m)).to_array().to_vec());
                    add(&mut b, &times, &v, 4, ch.interpolation, n, "rotation");
                }
                Target::BoneScale { bone, .. } => {
                    let Some((&n, _)) = bone_target(bone) else { continue };
                    add(&mut b, &times, &curve(&xyz, &xyz), 3, ch.interpolation, n, "scale");
                }
                Target::ObjectTranslation { .. } => add(&mut b, &times, &curve(&xyz, &xyz), 3, ch.interpolation, obj_node, "translation"),
                Target::ObjectRotation { .. } => add(&mut b, &times, &curve(&|v| Quat::from_vec4(v).normalize().to_array().to_vec(), &quat), 4, ch.interpolation, obj_node, "rotation"),
                Target::ObjectScale { .. } => add(&mut b, &times, &curve(&xyz, &xyz), 3, ch.interpolation, obj_node, "scale"),
                Target::MorphWeight { morph, .. } => {
                    if morph_nodes.contains_key(&object) && scene.objects[&object].mesh.morphs.iter().any(|m| &m.name == morph) {
                        morph_channels.entry(object).or_default().push(ch);
                    }
                }
            }
        }
        // glTF animates all morph weights of a node at once: merge per object.
        for (object, chans) in morph_channels {
            let o = &scene.objects[&object];
            // Cubic channels can't share one sampler with others: bake them at the clip's frame rate.
            let cubic = chans.iter().any(|c| c.interpolation.is_cubic());
            let mut times: Vec<f32> = chans.iter().flat_map(|c| c.keys.iter().map(|k| k.time)).collect();
            if cubic {
                let frames = (clip.length * clip.fps).ceil() as usize;
                times.extend((0..=frames).map(|f| (f as f32 / clip.fps).min(clip.length)));
            }
            times.sort_by(f32::total_cmp);
            times.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
            let mut values = Vec::with_capacity(times.len() * o.mesh.morphs.len());
            for &t in &times {
                for m in &o.mesh.morphs {
                    let ch = chans.iter().find(|c| matches!(&c.target, Target::MorphWeight { morph, .. } if *morph == m.name));
                    let w = match ch.and_then(|c| clip.sample(c, t)) {
                        Some(v) => v.x,
                        None => o.morph_weights.get(&m.name).copied().unwrap_or(0.0),
                    };
                    values.push(w);
                }
            }
            let interp =
                if chans.iter().all(|c| c.interpolation == Interpolation::Step) { Interpolation::Step } else { Interpolation::Linear };
            for &n in &morph_nodes[&object] {
                add(&mut b, &times, &values, 1, interp, n, "weights");
            }
        }
        if channels_j.is_empty() {
            continue; // glTF requires at least one channel per animation
        }
        animations.push(json!({
            "name": clip.name,
            "samplers": samplers_j,
            "channels": channels_j,
            "extras": {"polly": {
                "length": clip.length,
                "fps": clip.fps,
                "looping": clip.looping,
                "loop_start": clip.loop_start,
                "loop_end": clip.loop_end,
            }},
        }));
    }

    while b.bin.len() % 4 != 0 {
        b.bin.push(0);
    }
    let mut scene_j = json!({"nodes": roots});
    if !scene.slots.is_empty() {
        scene_j["extras"] = json!({"polly": slots_extras(scene, &selected, &hide_groups)});
    }
    let mut doc = json!({
        "asset": {"version": "2.0", "generator": "polly3d"},
        "scene": 0,
        "scenes": [scene_j],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": b.bin.len()}],
    });
    if !textures.is_empty() {
        doc["textures"] = json!(textures);
        doc["images"] = json!(images);
        doc["samplers"] = json!(samplers);
    }
    if !skins.is_empty() {
        doc["skins"] = json!(skins);
    }
    if !animations.is_empty() {
        doc["animations"] = json!(animations);
    }
    if !extensions_used.is_empty() {
        // Used, never required: importers that don't know them still load the file.
        doc["extensionsUsed"] = json!(extensions_used);
    }
    let mut js = serde_json::to_vec(&doc)?;
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + b.bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(b"glTF");
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((js.len() as u32).to_le_bytes());
    out.extend(b"JSON");
    out.extend(js);
    out.extend((b.bin.len() as u32).to_le_bytes());
    out.extend(b"BIN\0");
    out.extend(b.bin);
    Ok(out)
}

/// Slot table for the scene's extras (schema version 1, see docs/gltf-extras.md).
fn slots_extras(scene: &Scene, selected: &BTreeSet<ObjectId>, groups: &HideGroups) -> Value {
    let name = |id: &ObjectId| scene.objects[id].name.clone();
    let slots: serde_json::Map<String, Value> = scene
        .slots
        .iter()
        .map(|(slot, s)| {
            let variants: serde_json::Map<String, Value> = s
                .variants
                .iter()
                .map(|v| (v.name.clone(), json!(v.objects.iter().filter(|o| selected.contains(o)).map(name).collect::<Vec<_>>())))
                .collect();
            let hides: serde_json::Map<String, Value> = s
                .variants
                .iter()
                .filter(|v| !v.hides.is_empty())
                .map(|v| {
                    // One entry per node the rule hides (its faces may be split by overlapping rules).
                    let parts: Vec<Value> = v
                        .hides
                        .iter()
                        .filter(|h| selected.contains(&h.object))
                        .flat_map(|h| {
                            groups.get(&h.object).into_iter().flatten().filter(|(k, _)| k.contains(&h.part)).map(|(k, _)| {
                                json!([name(&h.object), h.part, group_node_name(&name(&h.object), k)])
                            })
                        })
                        .collect();
                    (v.name.clone(), json!(parts))
                })
                .collect();
            let mut j = json!({"default": s.default, "variants": variants});
            if !hides.is_empty() {
                j["hides"] = json!(hides);
            }
            (slot.clone(), j)
        })
        .collect();
    json!({"version": 1, "slots": slots})
}
