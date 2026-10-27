//! Baking PBR maps (normal, ambient occlusion, curvature masks) from a
//! detailed surface onto an object's UV layout.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use polly_core::scene::{Filter, Slot, Texture};
use polly_paint::bake::{BakeOptions, CurvatureMask, Source};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![def(
        "bake_maps",
        "Bake PBR maps onto an object's UVs (its exported mesh): \"normal\" (tangent-space, MikkTSpace like the export, so Bevy/Blender decode it exactly), \"ao\" (ambient occlusion), and masks \"curvature\", \"convexity\" (edge wear) and \"cavity\" (dirt). The detail comes from `source`: \"subdivided\" (this object's smooth subdivision surface), other objects (a sculpted or detailed high-poly version placed over it), or \"self\". Writes textures <object>_<map> (linear data) and wires normal/ao into the material. Masks are for painting: pass one as `strength_map` to fill_faces/paint tools.",
        false,
        bake_maps,
    )]
}

#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
enum SourceArg {
    /// "self" or "subdivided".
    Mode(String),
    /// High-detail objects, placed over the target.
    Objects(Vec<String>),
}

#[derive(Deserialize, JsonSchema)]
struct BakeMaps {
    /// The target (game) object.
    object: Option<String>,
    /// Which maps: "normal", "ao", "curvature", "convexity", "cavity".
    maps: Vec<String>,
    /// Detail source (default "subdivided" for normal maps, "self" otherwise).
    source: Option<SourceArg>,
    /// Subdivision level for source "subdivided" (default: 2 above the object's export level).
    level: Option<u8>,
    /// Texture size [w, h] for new maps (default: the object's base texture size, else 1024).
    size: Option<[u32; 2]>,
    /// Material to wire normal/ao maps into (default: the material of the object's faces, if one).
    material: Option<String>,
    #[serde(flatten)]
    options: BakeOptions,
    /// Grow coverage past UV island edges, texels (default 2).
    grow_px: Option<u32>,
}

fn bake_maps(ctx: &mut Ctx, a: BakeMaps) -> Result<Out> {
    let s = &ctx.doc.scene;
    let id = resolve(s, &a.object)?;
    if a.maps.is_empty() {
        bail!("give maps, e.g. [\"normal\", \"ao\"]");
    }
    // Target: the mesh as exported (static pose, export subdivision level).
    let posed = polly_core::anim::pose_scene(s, &[]);
    let obj = &posed.objects[&id];
    let export_level = obj.subdivision.map_or(0, |sd| sd.export());
    let target = polly_core::subdiv::subdivided(&obj.mesh, export_level);
    if target.faces.values().all(|f| f.corners.iter().all(|c| c.uv == polly_core::glam::Vec2::ZERO)) {
        bail!("'{}' has no UVs yet; unwrap it first", obj.name);
    }
    let needs_normal = a.maps.iter().any(|m| m == "normal");
    let source_mode = match &a.source {
        None => if needs_normal { "subdivided".to_string() } else { "self".to_string() },
        Some(SourceArg::Mode(m)) if m == "self" || m == "subdivided" => m.clone(),
        Some(SourceArg::Mode(m)) => bail!("source must be \"self\", \"subdivided\" or a list of objects, not '{m}'"),
        Some(SourceArg::Objects(_)) => "objects".into(),
    };
    // Source surface in the target's object space.
    let source_mesh: Option<polly_core::mesh::Mesh> = match (&a.source, source_mode.as_str()) {
        (_, "self") => None,
        (_, "subdivided") => {
            let level = a.level.unwrap_or(export_level + 2).clamp(export_level + 1, polly_core::subdiv::MAX_LEVEL);
            Some((*polly_core::subdiv::subdivided(&obj.mesh, level)).clone())
        }
        (Some(SourceArg::Objects(names)), _) => {
            let inv = posed.world_matrix(id).inverse();
            let mut m = polly_core::mesh::Mesh::new();
            for n in names {
                let oid = s.resolve(Some(n))?;
                if oid == id {
                    bail!("the source objects must not include the target itself (use source \"self\")");
                }
                let o = &posed.objects[&oid];
                let lvl = o.subdivision.map_or(0, |sd| sd.export().max(sd.level));
                let mut part = (*polly_core::subdiv::subdivided(&o.mesh, lvl)).clone();
                part.transform(inv * posed.world_matrix(oid));
                m.append(&part);
            }
            Some(m)
        }
        _ => unreachable!(),
    };
    let mut source = source_mesh.as_ref().map(Source::new);
    let size = (target.bounds().map(|(lo, hi)| (hi - lo).length()).unwrap_or(1.0)).max(1e-4);
    let max_dist = a.options.max_distance.unwrap_or(size * 0.1);
    // Texture size: given, else the base texture's, else 1024.
    let base_size = target
        .faces
        .values()
        .find_map(|f| s.materials.get(&f.material)?.texture.and_then(|t| s.textures.get(&t)))
        .map(|t| [t.width, t.height]);
    let [w, h] = a.size.or(base_size).unwrap_or([1024, 1024]);
    if !(1..=polly_core::scene::MAX_TEXTURE_SIZE).contains(&w) || !(1..=polly_core::scene::MAX_TEXTURE_SIZE).contains(&h) {
        bail!("map size must be 1..{}", polly_core::scene::MAX_TEXTURE_SIZE);
    }
    let faces = target.faces.keys().copied().collect();
    let map = polly_paint::SurfaceMap::new(&target, &faces, w, h, a.grow_px.unwrap_or(2));
    let material = match &a.material {
        Some(m) => Some(s.resolve_material(m)?),
        None => {
            let mats: std::collections::BTreeSet<u32> = obj.mesh.faces.values().map(|f| f.material).collect();
            (mats.len() == 1).then(|| *mats.iter().next().unwrap())
        }
    };
    let tangents = if needs_normal { polly_export::corner_tangents(&target) } else { Default::default() };
    // Occluders for AO: the source (or target) plus nothing else.
    let occ = polly_core::bvh::Bvh::new(source_mesh.as_ref().unwrap_or(&target), polly_core::glam::Mat4::IDENTITY);
    let mut baked: BTreeMap<String, Texture> = BTreeMap::new();
    let mut report = serde_json::Map::new();
    for name in &a.maps {
        let tex_name = format!("{}_{name}", obj.name);
        let fill = if name == "normal" { [128, 128, 255, 255] } else { [255; 4] };
        let mut t = Texture::new(&tex_name, w, h, fill);
        t.linear = true;
        t.filter = Filter::Linear;
        let p = polly_paint::Paint::default();
        let r = match name.as_str() {
            "normal" => polly_paint::bake::bake_normal(&mut t, &map, &target, &tangents, source.as_ref(), max_dist, &p),
            "ao" => polly_paint::bake::bake_ao(&mut t, &map, source.as_ref(), &occ, &a.options, max_dist, &p),
            "curvature" | "convexity" | "cavity" => {
                let kind = match name.as_str() {
                    "curvature" => CurvatureMask::Curvature,
                    "convexity" => CurvatureMask::Convexity,
                    _ => CurvatureMask::Cavity,
                };
                polly_paint::bake::bake_curvature(&mut t, &map, &target, source.as_mut(), kind, &a.options, max_dist, &p)
            }
            other => bail!("unknown map '{other}' (normal, ao, curvature, convexity, cavity)"),
        };
        let mut j = json!({"texture": tex_name, "texels": r.changed});
        if r.missed > 0 {
            j["missed_texels"] = json!(r.missed);
            j["hint"] = json!("no source surface within max_distance of these texels (raise max_distance, or check the source covers the target)");
        }
        report.insert(name.clone(), j);
        baked.insert(name.clone(), t);
    }
    let wired = edit_scene(ctx, "bake maps", |s| {
        let mut wired = Vec::new();
        for (name, t) in baked {
            let tid = match s.resolve_texture(&t.name) {
                Ok(tid) => {
                    *s.texture_mut(tid)? = t;
                    tid
                }
                Err(_) => s.add_texture(t),
            };
            let slot = match name.as_str() {
                "normal" => Some(Slot::Normal),
                "ao" => Some(Slot::Occlusion),
                _ => None,
            };
            if let (Some(slot), Some(mid)) = (slot, material) {
                if let Some(m) = s.materials.get_mut(&mid) {
                    *m.slot_mut(slot) = Some(tid);
                    wired.push(format!("{} -> {}.{}", s.textures[&tid].name, m.name, slot.name()));
                }
            }
        }
        Ok(wired)
    })?;
    let mut j = json!({"object": ctx.doc.scene.objects[&id].name, "source": source_mode, "size": [w, h], "maps": report});
    if !wired.is_empty() {
        j["wired"] = json!(wired);
    } else if needs_normal || a.maps.iter().any(|m| m == "ao") {
        j["note"] = json!("not wired into a material (the object uses several materials): pass `material`, or update_material normal_texture / occlusion_texture");
    }
    Ok(Out::json(j))
}
