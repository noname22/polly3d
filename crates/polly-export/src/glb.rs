//! Binary glTF 2.0 export. One node per object, one primitive per material,
//! textures embedded as PNG with nearest/linear samplers matching the
//! texture's filter setting.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use polly_core::scene::{Filter, ObjectId, Scene};
use serde_json::{Value, json};

const NEAREST: u32 = 9728;
const LINEAR: u32 = 9729;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
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

    fn floats(&mut self, data: &[f32], comps: usize, bounds: bool) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(ARRAY_BUFFER));
        let ty = match comps {
            2 => "VEC2",
            3 => "VEC3",
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
    idx: Vec<u32>,
    dedup: HashMap<[u32; 12], u32>,
}

impl Prim {
    fn vertex(&mut self, p: [f32; 3], n: [f32; 3], uv: [f32; 2], c: [f32; 4]) -> u32 {
        let mut key = [0u32; 12];
        for (k, v) in p.iter().chain(&n).chain(&uv).chain(&c[..1]).enumerate() {
            key[k] = v.to_bits();
        }
        key[9] = c[1].to_bits();
        key[10] = c[2].to_bits();
        key[11] = c[3].to_bits();
        if let Some(i) = self.dedup.get(&key) {
            return *i;
        }
        let i = (self.pos.len() / 3) as u32;
        self.pos.extend(p);
        self.nrm.extend(n);
        self.uv.extend(uv);
        self.col.extend(c);
        self.dedup.insert(key, i);
        i
    }
}

pub struct ExportOptions {
    /// Objects to export (with their descendants); `None` = all visible.
    pub objects: Option<Vec<ObjectId>>,
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
    let mut nodes = Vec::new();
    let mut node_of: BTreeMap<ObjectId, usize> = BTreeMap::new();
    for id in &selected {
        let o = &scene.objects[id];
        let mesh = &o.mesh;
        let normals = mesh.corner_normals();
        let mut prims: BTreeMap<u32, Prim> = BTreeMap::new();
        for (&fid, f) in &mesh.faces {
            let pts = mesh.face_positions(f);
            let prim = prims.entry(f.material).or_default();
            let vidx: Vec<u32> = (0..f.len())
                .map(|i| {
                    let c = &f.corners[i];
                    prim.vertex(pts[i].to_array(), normals[&(fid, i)].to_array(), c.uv.to_array(), c.color.to_array())
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
            pj.push(json!({"attributes": attrs, "indices": b.indices(&p.idx), "material": mat_index[&mid], "mode": 4}));
        }
        let t = &o.transform;
        let mut node = json!({
            "name": o.name,
            "translation": t.translation.to_array(),
            "rotation": t.rotation.to_array(),
            "scale": t.scale.to_array(),
        });
        if !pj.is_empty() {
            meshes.push(json!({"name": o.name, "primitives": pj}));
            node["mesh"] = json!(meshes.len() - 1);
        }
        nodes.push(node);
        node_of.insert(*id, nodes.len() - 1);
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
