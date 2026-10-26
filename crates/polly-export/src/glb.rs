//! Binary glTF 2.0 export. One node per object, one primitive per material,
//! textures embedded as PNG with nearest/linear samplers matching the
//! texture's filter setting.
//!
//! Rigging and animation:
//! - An object with a skeleton and skin weights gets one joint node per bone,
//!   parented under the object's node (root bones directly, other bones under
//!   their parent bone), and a skin whose inverse bind matrices are the
//!   inverted bone rest frames in object space. Vertices without weights are
//!   bound to an extra, never-animated `<object>_root` joint.
//! - The skeleton is exported in its rest pose: the object's static `pose` is
//!   NOT baked into the joints (clips are exported as animations instead).
//! - Morph targets become per-primitive POSITION targets; the mesh's default
//!   `weights` are the object's static morph weights and
//!   `extras.targetNames` lists the morph names.
//! - Every clip becomes a glTF animation. Bone channels are converted from
//!   pose-relative values to full joint-node local TRS; morph weight channels
//!   of one object are merged into a single `weights` sampler. Loop settings
//!   go in `extras.polly`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use polly_core::anim::{BoneId, Channel, Interpolation, Target};
use polly_core::glam::{Mat4, Quat, Vec3};
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

/// Per-object skin data gathered while building nodes.
struct SkinInfo {
    /// Joint node of each bone.
    joint_node: HashMap<BoneId, usize>,
    /// Rest local transform (relative to the parent joint / object node) of each bone.
    local: HashMap<BoneId, (Vec3, Quat)>,
}

pub struct ExportOptions {
    /// Objects to export (with their descendants); `None` = all visible.
    pub objects: Option<Vec<ObjectId>>,
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

pub fn export_glb(scene: &Scene, opts: &ExportOptions) -> Result<Vec<u8>> {
    let mut b = Builder::default();
    let selected: BTreeSet<ObjectId> = match &opts.objects {
        Some(list) => {
            let mut s: BTreeSet<ObjectId> = list.iter().copied().collect();
            // include descendants
            loop {
                let before = s.len();
                for (id, o) in &scene.objects {
                    if o.parent.is_some_and(|p| s.contains(&p)) {
                        s.insert(*id);
                    }
                }
                if s.len() == before {
                    break;
                }
            }
            s
        }
        None => scene.objects.iter().filter(|(_, o)| o.visible).map(|(id, _)| *id).collect(),
    };

    // Textures / images / samplers.
    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut samplers = Vec::new();
    let mut tex_index: HashMap<u32, usize> = HashMap::new();
    let used_materials: BTreeSet<u32> = selected
        .iter()
        .flat_map(|id| scene.objects[id].mesh.faces.values().map(|f| f.material).collect::<Vec<_>>())
        .collect();
    for mid in &used_materials {
        let Some(tid) = scene.materials.get(mid).and_then(|m| m.texture) else { continue };
        if tex_index.contains_key(&tid) {
            continue;
        }
        let Some(t) = scene.textures.get(&tid) else { continue };
        let png = crate::png(t)?;
        let view = b.view(&png, None);
        images.push(json!({"bufferView": view, "mimeType": "image/png", "name": t.name}));
        let f = if t.filter == Filter::Nearest { NEAREST } else { LINEAR };
        samplers.push(json!({"magFilter": f, "minFilter": f, "wrapS": 10497, "wrapT": 10497}));
        textures.push(json!({"source": images.len() - 1, "sampler": samplers.len() - 1}));
        tex_index.insert(tid, textures.len() - 1);
    }

    let mut materials = Vec::new();
    let mut mat_index: BTreeMap<u32, usize> = BTreeMap::new();
    let mut uses_unlit = false;
    for mid in &used_materials {
        let m = scene.materials.get(mid).cloned().unwrap_or_else(|| scene.materials[&0].clone());
        let mut pbr = json!({"baseColorFactor": m.base_color, "metallicFactor": 0.0, "roughnessFactor": 1.0});
        if let Some(ti) = m.texture.and_then(|t| tex_index.get(&t)) {
            pbr["baseColorTexture"] = json!({"index": ti});
        }
        let mut mj = json!({"name": m.name, "pbrMetallicRoughness": pbr, "doubleSided": m.double_sided});
        if let Some(c) = m.alpha_cutoff {
            mj["alphaMode"] = json!("MASK");
            mj["alphaCutoff"] = json!(c);
        }
        if m.unlit {
            mj["extensions"] = json!({"KHR_materials_unlit": {}});
            uses_unlit = true;
        }
        materials.push(mj);
        mat_index.insert(*mid, materials.len() - 1);
    }

    let mut meshes = Vec::new();
    let mut nodes: Vec<Value> = Vec::new();
    let mut skins = Vec::new();
    let mut node_of: BTreeMap<ObjectId, usize> = BTreeMap::new();
    let mut skin_info: HashMap<ObjectId, SkinInfo> = HashMap::new();
    // Objects whose exported mesh has morph targets.
    let mut has_morphs: BTreeSet<ObjectId> = BTreeSet::new();
    for id in &selected {
        let o = &scene.objects[id];
        let mesh = &o.mesh;
        let normals = mesh.corner_normals();
        let used_verts: BTreeSet<VertId> = mesh.faces.values().flat_map(|f| f.verts.iter().copied()).collect();

        // Skin: joints in parent-first order, plus a root joint for unweighted vertices.
        let skeleton = o.skeleton.as_ref().filter(|sk| !sk.bones.is_empty() && !mesh.weights.is_empty());
        let mut bones: Vec<BoneId> = Vec::new();
        let mut vert_skin: HashMap<VertId, ([u16; 4], [f32; 4])> = HashMap::new();
        let mut needs_root = false;
        if let Some(sk) = skeleton {
            bones = sk.ordered();
            let joint_idx: HashMap<BoneId, u16> = bones.iter().enumerate().map(|(i, b)| (*b, i as u16)).collect();
            let root = bones.len() as u16;
            for v in &used_verts {
                let s = vertex_skin(mesh.weights.get(v), &joint_idx).unwrap_or_else(|| {
                    needs_root = true;
                    ([root, 0, 0, 0], [1.0, 0.0, 0.0, 0.0])
                });
                vert_skin.insert(*v, s);
            }
        }
        let skinned = skeleton.is_some();
        let morphed = !mesh.morphs.is_empty();

        let mut prims: BTreeMap<u32, Prim> = BTreeMap::new();
        for (&fid, f) in &mesh.faces {
            let pts = mesh.face_positions(f);
            let prim = prims.entry(f.material).or_default();
            let vidx: Vec<u32> = (0..f.len())
                .map(|i| {
                    let c = &f.corners[i];
                    let v = f.verts[i];
                    prim.vertex(
                        v,
                        skinned || morphed,
                        pts[i].to_array(),
                        normals[&(fid, i)].to_array(),
                        c.uv.to_array(),
                        c.color.to_array(),
                        vert_skin.get(&v),
                    )
                })
                .collect();
            for [a, bb, c] in polly_core::geom::triangulate_polygon(&pts) {
                prim.idx.extend([vidx[a], vidx[bb], vidx[c]]);
            }
        }
        let mut pj = Vec::new();
        for (mid, p) in prims {
            if p.idx.is_empty() {
                continue;
            }
            let mut attrs = json!({
                "POSITION": b.floats(&p.pos, 3, true),
                "NORMAL": b.floats(&p.nrm, 3, false),
                "TEXCOORD_0": b.floats(&p.uv, 2, false),
            });
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
                        let d: Vec<f32> =
                            p.vids.iter().flat_map(|v| m.offsets.get(v).copied().unwrap_or(Vec3::ZERO).to_array()).collect();
                        json!({"POSITION": b.floats(&d, 3, true)})
                    })
                    .collect();
                prim["targets"] = json!(targets);
            }
            pj.push(prim);
        }
        let t = &o.transform;
        let mut node = json!({
            "name": o.name,
            "translation": t.translation.to_array(),
            "rotation": t.rotation.to_array(),
            "scale": t.scale.to_array(),
        });
        if !pj.is_empty() {
            let mut mj = json!({"name": o.name, "primitives": pj});
            if morphed {
                let w: Vec<f32> = mesh.morphs.iter().map(|m| o.morph_weights.get(&m.name).copied().unwrap_or(0.0)).collect();
                let names: Vec<&str> = mesh.morphs.iter().map(|m| m.name.as_str()).collect();
                mj["weights"] = json!(w);
                mj["extras"] = json!({"targetNames": names});
                has_morphs.insert(*id);
            }
            meshes.push(mj);
            node["mesh"] = json!(meshes.len() - 1);
        }
        let obj_node = nodes.len();
        nodes.push(node);
        node_of.insert(*id, obj_node);

        if let Some(sk) = skeleton {
            let rest: HashMap<BoneId, Mat4> = bones.iter().map(|b| (*b, sk.bones[b].rest_matrix())).collect();
            let mut info = SkinInfo { joint_node: HashMap::new(), local: HashMap::new() };
            let mut joints = Vec::new();
            let mut ibm: Vec<f32> = Vec::new();
            let mut obj_children = Vec::new();
            for bid in &bones {
                let bone = &sk.bones[bid];
                let parent = bone.parent.filter(|p| rest.contains_key(p));
                let parent_rest = parent.map_or(Mat4::IDENTITY, |p| rest[&p]);
                let (_, r, tr) = (parent_rest.inverse() * rest[bid]).to_scale_rotation_translation();
                let r = r.normalize();
                let n = nodes.len();
                nodes.push(json!({"name": bone.name, "translation": tr.to_array(), "rotation": r.to_array()}));
                match parent {
                    Some(p) => {
                        let pn = info.joint_node[&p];
                        let children = nodes[pn].as_object_mut().unwrap().entry("children").or_insert(json!([]));
                        children.as_array_mut().unwrap().push(json!(n));
                    }
                    None => obj_children.push(n),
                }
                info.joint_node.insert(*bid, n);
                info.local.insert(*bid, (tr, r));
                joints.push(n);
                ibm.extend(rest[bid].inverse().to_cols_array());
            }
            if needs_root {
                let n = nodes.len();
                nodes.push(json!({"name": format!("{}_root", o.name)}));
                obj_children.push(n);
                joints.push(n);
                ibm.extend(Mat4::IDENTITY.to_cols_array());
            }
            nodes[obj_node]["children"] = json!(obj_children);
            if nodes[obj_node].get("mesh").is_some() {
                let acc = b.float_accessor(&ibm, 16, false, None);
                skins.push(json!({"name": o.name, "joints": joints, "inverseBindMatrices": acc}));
                nodes[obj_node]["skin"] = json!(skins.len() - 1);
            }
            skin_info.insert(*id, info);
        }
    }
    let mut roots = Vec::new();
    for id in &selected {
        let n = node_of[id];
        match scene.objects[id].parent.and_then(|p| node_of.get(&p)) {
            Some(&p) => {
                let children = nodes[p].as_object_mut().unwrap().entry("children").or_insert(json!([]));
                children.as_array_mut().unwrap().push(json!(n));
            }
            None => roots.push(n),
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
            let interp = if interp == Interpolation::Step { "STEP" } else { "LINEAR" };
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
            match &ch.target {
                Target::BoneTranslation { bone, .. } => {
                    let Some((&n, (lt, lr))) = bone_target(bone) else { continue };
                    let v: Vec<f32> = ch.keys.iter().flat_map(|k| (lt + lr * k.value.truncate()).to_array()).collect();
                    add(&mut b, &times, &v, 3, ch.interpolation, n, "translation");
                }
                Target::BoneRotation { bone, .. } => {
                    let Some((&n, (_, lr))) = bone_target(bone) else { continue };
                    let v: Vec<f32> =
                        ch.keys.iter().flat_map(|k| (lr * Quat::from_vec4(k.value)).normalize().to_array()).collect();
                    add(&mut b, &times, &v, 4, ch.interpolation, n, "rotation");
                }
                Target::BoneScale { bone, .. } => {
                    let Some((&n, _)) = bone_target(bone) else { continue };
                    let v: Vec<f32> = ch.keys.iter().flat_map(|k| k.value.truncate().to_array()).collect();
                    add(&mut b, &times, &v, 3, ch.interpolation, n, "scale");
                }
                Target::ObjectTranslation { .. } => {
                    let v: Vec<f32> = ch.keys.iter().flat_map(|k| k.value.truncate().to_array()).collect();
                    add(&mut b, &times, &v, 3, ch.interpolation, obj_node, "translation");
                }
                Target::ObjectRotation { .. } => {
                    let v: Vec<f32> = ch.keys.iter().flat_map(|k| Quat::from_vec4(k.value).normalize().to_array()).collect();
                    add(&mut b, &times, &v, 4, ch.interpolation, obj_node, "rotation");
                }
                Target::ObjectScale { .. } => {
                    let v: Vec<f32> = ch.keys.iter().flat_map(|k| k.value.truncate().to_array()).collect();
                    add(&mut b, &times, &v, 3, ch.interpolation, obj_node, "scale");
                }
                Target::MorphWeight { morph, .. } => {
                    if has_morphs.contains(&object) && scene.objects[&object].mesh.morphs.iter().any(|m| &m.name == morph) {
                        morph_channels.entry(object).or_default().push(ch);
                    }
                }
            }
        }
        // glTF animates all morph weights of a node at once: merge per object.
        for (object, chans) in morph_channels {
            let o = &scene.objects[&object];
            let mut times: Vec<f32> = chans.iter().flat_map(|c| c.keys.iter().map(|k| k.time)).collect();
            times.sort_by(f32::total_cmp);
            times.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
            let mut values = Vec::with_capacity(times.len() * o.mesh.morphs.len());
            for &t in &times {
                for m in &o.mesh.morphs {
                    let ch = chans.iter().find(|c| matches!(&c.target, Target::MorphWeight { morph, .. } if *morph == m.name));
                    let w = match ch.and_then(|c| c.sample(t)) {
                        Some(v) => v.x,
                        None => o.morph_weights.get(&m.name).copied().unwrap_or(0.0),
                    };
                    values.push(w);
                }
            }
            let interp =
                if chans.iter().all(|c| c.interpolation == Interpolation::Step) { Interpolation::Step } else { Interpolation::Linear };
            add(&mut b, &times, &values, 1, interp, node_of[&object], "weights");
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
    let mut doc = json!({
        "asset": {"version": "2.0", "generator": "polly3d"},
        "scene": 0,
        "scenes": [{"nodes": roots}],
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
    if uses_unlit {
        doc["extensionsUsed"] = json!(["KHR_materials_unlit"]);
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
