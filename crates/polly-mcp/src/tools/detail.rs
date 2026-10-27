//! Detail and LOD: sculpt-style region brushes, decimation, hair cards.

use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use polly_core::glam::Vec3;
use polly_core::ops::hair::HairCards;
use polly_core::ops::sculpt::{Brush, Stroke};
use polly_core::ops::transform::Falloff;
use polly_core::scene::{Anisotropy, Filter, Material, Texture};
use polly_paint::Rgba;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "brush",
            "Sculpt-style region brush along a stroke: mode smooth (relax lumps), inflate (swell; negative strength shrinks), crease (groove along the stroke; negative = ridge), displace (3-D noise along the normals: pores, bark, stone; `scale`, `seed`) or flatten. The stroke is `points` (object space) or `pixels` of the last render; vertices within `radius` move, fading out (falloff_curve). Works best on dense meshes (subdivide or a subdivision level first). `mirror` keeps it symmetric.",
            false,
            brush,
        ),
        def(
            "decimate",
            "Reduce an object's triangle count by quadric edge collapse (to `ratio` of its triangles or `target_tris`). Borders, UV seams, sharp edges, creases and material/part borders are kept, as are skin weights and morphs. Faces become triangles. For game LODs prefer export_glb `lods`, which leaves the editable mesh alone.",
            false,
            decimate,
        ),
        def(
            "add_hair_cards",
            "Grow hair cards from faces of a scalp object: a new object of textured strips (alpha-cutout strand texture with `variations` columns, double-sided, anisotropic highlights along the strands). Shape with length, width, lift, gravity, comb direction, random; the cards follow the scalp's rig (weights from the nearest scalp vertex). Repaint the texture, or use your own, for the final look.",
            false,
            add_hair_cards,
        ),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct BrushArgs {
    object: Option<String>,
    #[serde(flatten)]
    brush: Brush,
    /// Stroke points in object space (one point = a dab).
    points: Option<Vec<[f32; 3]>>,
    /// Stroke as pixels of the last render (snapped onto the object).
    pixels: Option<Vec<[f32; 2]>>,
    /// Brush radius in object units.
    radius: f32,
    /// Strength (default 0.5).
    strength: Option<f32>,
    #[serde(default)]
    falloff_curve: Falloff,
    /// Passes for smooth/crease (default 2).
    iterations: Option<u32>,
    /// Only these vertices may move (default all).
    elements: Option<Vec<String>>,
    #[serde(flatten)]
    mirror: Symmetry,
}

fn brush(ctx: &mut Ctx, a: BrushArgs) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let path: Vec<Vec3> = match (&a.points, &a.pixels) {
        (Some(p), None) => p.iter().map(|q| Vec3::from_array(*q)).collect(),
        (None, Some(px)) => {
            let s = super::view::last_scene(ctx);
            let inv = ctx.doc.scene.world_matrix(id).inverse();
            let mut pts = Vec::new();
            for &[x, y] in px {
                let (cam, x, y) = super::view::camera_at(ctx, x, y)?;
                let h = polly_render::pick(&s, &cam, x, y).with_context(|| format!("nothing under pixel [{x}, {y}]"))?;
                if h.object != id {
                    bail!("pixel [{x}, {y}] hits '{}', not the brushed object", h.object_name);
                }
                pts.push(inv.transform_point3(Vec3::from(h.position)));
            }
            pts
        }
        _ => bail!("give either `points` or `pixels`"),
    };
    let allowed = match &a.elements {
        Some(_) => Some(elements(&ctx.doc.scene, id, &a.elements)?.touched_verts(&ctx.doc.scene.object(id)?.mesh)),
        None => None,
    };
    let stroke = Stroke { path, radius: a.radius, strength: a.strength.unwrap_or(0.5), falloff: a.falloff_curve, iterations: a.iterations.unwrap_or(2) };
    let mut moved = 0;
    // Brushes act on whole regions: the stroke decides what moves, not the selection.
    let all = Some(vec!["all".to_string()]);
    let mut out = edit_mesh_mirrored(ctx, &a.object, Some(&all), "brush", a.mirror, |m, _| {
        moved = polly_core::ops::sculpt::apply(m, a.brush, &stroke, allowed.as_ref())?;
        Ok(Default::default())
    })?;
    out.json["moved_verts"] = json!(moved);
    if moved == 0 {
        out.json["note"] = json!("no vertices within the radius of the stroke");
    } else if moved < 8 {
        out.json["note"] = json!("only a few vertices were in reach: brushes need dense geometry (subdivide, or a larger radius)");
    }
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
struct DecimateArgs {
    object: Option<String>,
    /// Fraction of triangles to keep (0..1).
    ratio: Option<f32>,
    /// Or an absolute triangle budget.
    target_tris: Option<usize>,
}

fn decimate(ctx: &mut Ctx, a: DecimateArgs) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let before = ctx.doc.scene.object(id)?.mesh.triangle_count();
    let target = match (a.ratio, a.target_tris) {
        (Some(r), None) if r > 0.0 && r <= 1.0 => ((before as f32) * r).round() as usize,
        (None, Some(t)) => t,
        _ => bail!("give ratio (0..1) or target_tris"),
    };
    let mut out = edit_mesh(ctx, &a.object, None, "decimate", |m, _| {
        *m = polly_core::ops::decimate::decimate(m, target.max(1));
        Ok(Default::default())
    })?;
    let after = ctx.doc.scene.object(id)?.mesh.triangle_count();
    out.json["triangles"] = json!({"before": before, "after": after, "target": target});
    if after > target + target / 10 + 2 {
        out.json["note"] = json!("stopped above the target: the rest is locked by borders, seams, sharp edges or material borders");
    }
    // The full diff of a decimation is huge and not useful to paste back.
    out.json.as_object_mut().unwrap().remove("changes");
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
struct HairArgs {
    /// The scalp object (default: active).
    object: Option<String>,
    /// Scalp faces to grow from (default: its selection).
    elements: Option<Vec<String>>,
    /// Name of the new object (default "<scalp>_hair").
    name: Option<String>,
    #[serde(flatten)]
    cards: HairCards,
    /// Hair colour (default dark brown).
    color: Option<Rgba>,
    /// Strand texture size (default [256, 512]).
    texture_size: Option<[u32; 2]>,
    /// Soft alpha blending instead of a cutout (sorting artefacts possible).
    #[serde(default)]
    blend: bool,
}

/// Strand texture: per column, vertical strands with alpha, fading at the tips.
fn strand_texture(name: &str, w: u32, h: u32, cols: u32, color: Rgba, seed: u64) -> Texture {
    let mut t = Texture::new(name, w, h, [color[0], color[1], color[2], 0]);
    t.filter = Filter::Linear;
    let px = t.pixels_mut();
    let colw = (w / cols.max(1)).max(1);
    let r = |i: u64| {
        let mut x = seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 29;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    for c in 0..cols {
        let strands = (colw as f32 * 0.6) as u64 + 3;
        for k in 0..strands {
            let i = c as u64 * 1000 + k;
            let x0 = c * colw + (r(i) * colw as f32 * 0.9 + colw as f32 * 0.05) as u32;
            let len = 0.55 + 0.45 * r(i + 7);
            let shade = 0.75 + 0.5 * r(i + 13);
            for y in 0..h {
                let v = y as f32 / h as f32;
                if v > len {
                    break;
                }
                // Slight sideways wave.
                let x = (x0 as f32 + (v * 9.0 + r(i + 3) * 6.0).sin() * 1.5).round().clamp((c * colw) as f32, ((c + 1) * colw - 1) as f32) as u32;
                let a = (1.0 - ((v - len * 0.75) / (len * 0.25)).clamp(0.0, 1.0)) * 255.0;
                let idx = ((y * w + x) * 4) as usize;
                let col: [u8; 3] = std::array::from_fn(|ch| ((color[ch] as f32 * shade).min(255.0)) as u8);
                px[idx..idx + 3].copy_from_slice(&col);
                px[idx + 3] = px[idx + 3].max(a as u8);
            }
        }
    }
    t
}

fn add_hair_cards(ctx: &mut Ctx, a: HairArgs) -> Result<Out> {
    let s = &ctx.doc.scene;
    let scalp_id = resolve(s, &a.object)?;
    let scalp = s.object(scalp_id)?.clone();
    let faces: BTreeSet<_> = elements(s, scalp_id, &a.elements)?.implied_faces(&scalp.mesh);
    let (mut mesh, roots) = polly_core::ops::hair::hair_cards(&scalp.mesh, &faces, &a.cards)?;
    // Follow the scalp's rig.
    if !scalp.mesh.weights.is_empty() {
        for (v, root) in &roots {
            if let Some(w) = scalp.mesh.weights.get(root) {
                mesh.weights.insert(*v, w.clone());
            }
        }
    }
    let name = a.name.clone().unwrap_or_else(|| format!("{}_hair", scalp.name));
    let [tw, th] = a.texture_size.unwrap_or([256, 512]);
    if !(16..=polly_core::scene::MAX_TEXTURE_SIZE).contains(&tw) || !(16..=polly_core::scene::MAX_TEXTURE_SIZE).contains(&th) {
        bail!("texture_size must be 16..{}", polly_core::scene::MAX_TEXTURE_SIZE);
    }
    let color = a.color.unwrap_or([70, 45, 30, 255]);
    let cols = a.cards.variations.clamp(1, 16);
    let tris = mesh.triangle_count();
    let (oid, tex_name) = edit_scene(ctx, "add hair cards", |s| {
        let tex_name = {
            let mut n = format!("{name}_strands");
            while s.textures.values().any(|t| t.name == n) {
                n.push('_');
            }
            n
        };
        let tid = s.add_texture(strand_texture(&tex_name, tw, th, cols, color, a.cards.seed));
        let mat = Material {
            texture: Some(tid),
            alpha_cutoff: (!a.blend).then_some(0.4),
            blend: a.blend,
            double_sided: true,
            roughness: 0.45,
            anisotropy: Some(Anisotropy { strength: 0.7, rotation: std::f32::consts::FRAC_PI_2 }),
            ..Material::new(format!("{name}_hair"), [1.0; 4])
        };
        let mid = s.add_material(mat);
        for f in mesh.faces.values_mut() {
            f.material = mid;
        }
        let oid = s.add_object(&name, std::mem::take(&mut mesh));
        let o = s.object_mut(oid)?;
        o.transform = scalp.transform;
        o.parent = scalp.parent;
        o.parent_bone = scalp.parent_bone;
        o.skin = match (scalp.skin, &scalp.skeleton) {
            (Some(b), _) => Some(b),
            (None, Some(_)) => Some(polly_core::scene::SkinBinding { armature: scalp_id }),
            _ => None,
        };
        Ok((oid, tex_name))
    })?;
    let o = ctx.doc.scene.object(oid)?;
    let j = json!({
        "object": o.name,
        "cards": a.cards.count,
        "triangles": tris,
        "texture": tex_name,
        "material": format!("{name}_hair"),
        "note": "alpha-cutout, double-sided, anisotropic material; check with render, then repaint the strands texture or tweak length/gravity/lift",
    });
    crate::tools::view::maybe_preview(ctx, oid, Out::json(j))
}
