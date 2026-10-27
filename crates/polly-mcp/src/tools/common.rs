use std::sync::Arc;

use anyhow::{Result, bail};
use polly_core::document::mesh_diff;
use polly_core::glam::Vec3;
use polly_core::ids::Selection;
use polly_core::mesh::{FaceId, Mesh, VertId};
use polly_core::scene::{ObjectId, Scene};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::registry::{Ctx, Out};

pub fn resolve(scene: &Scene, object: &Option<String>) -> Result<ObjectId> {
    Ok(scene.resolve(object.as_deref())?)
}

/// Parse element ids against an object; `None` means the object's current selection.
pub fn elements(scene: &Scene, id: ObjectId, ids: &Option<Vec<String>>) -> Result<Selection> {
    let o = scene.object(id)?;
    let default = vec!["selection".to_string()];
    let ids = ids.as_ref().unwrap_or(&default);
    let s = Selection::parse(ids, &o.mesh, &o.selection)?;
    if s.is_empty() {
        bail!("no elements given and nothing is selected on '{}'; pass element ids or use select", o.name);
    }
    Ok(s)
}

pub fn faces_of(mesh: &Mesh, s: &Selection) -> std::collections::BTreeSet<FaceId> {
    s.implied_faces(mesh)
}

pub fn verts_of(mesh: &Mesh, s: &Selection) -> std::collections::BTreeSet<VertId> {
    s.touched_verts(mesh)
}

/// Run a mesh edit on one object as an undoable step and report what changed.
/// The closure gets the mesh and the parsed elements; the returned selection
/// becomes the object's new selection (if non-empty).
pub fn edit_mesh(
    ctx: &mut Ctx,
    object: &Option<String>,
    ids: Option<&Option<Vec<String>>>,
    label: &str,
    f: impl FnOnce(&mut Mesh, &Selection) -> polly_core::Result<Selection>,
) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, object)?;
    let sel = match ids {
        Some(ids) => elements(&ctx.doc.scene, id, ids)?,
        None => ctx.doc.scene.object(id)?.selection.clone(),
    };
    let before = ctx.doc.scene.object(id)?.mesh.clone();
    let result = ctx.doc.edit(label, |scene| {
        let o = scene.object_mut(id)?;
        let mesh = Arc::make_mut(&mut o.mesh);
        let r = f(mesh, &sel)?;
        if r.is_empty() {
            o.selection.prune(mesh);
        } else {
            o.selection = r.clone();
        }
        Ok(r)
    })?;
    let o = ctx.doc.scene.object(id)?;
    let mut report = json!({
        "object": o.name,
        "changes": mesh_diff(&before, &o.mesh),
    });
    if !result.is_empty() {
        report["result"] = result.to_json();
        report["note"] = json!("result elements are now the object's selection");
    }
    // Only problems this edit introduced; older ones are summarised in one line.
    let (was, now) = (polly_core::mesh::Mesh::problems_cached(&before), polly_core::mesh::Mesh::problems_cached(&o.mesh));
    let warnings = now.new_since(&was);
    if !warnings.is_empty() {
        report["warnings"] = json!(warnings);
    }
    if !was.is_empty() && !now.is_empty() {
        report["earlier_warnings"] = json!("the mesh still has earlier problems (describe_mesh lists them)");
    }
    let out = Out::json(report);
    crate::tools::view::maybe_preview(ctx, id, out)
}

/// Symmetric editing options shared by the vertex tools.
#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema)]
pub struct Symmetry {
    /// Symmetric edit: each moved vertex's mirror partner across the object's x = 0 plane follows (the +X side leads if both move; vertices on the plane stay on it).
    #[serde(default)]
    pub mirror: bool,
    /// How far (object units) a vertex may be from the exact mirror position and still pair (default: 0.1% of the mesh size).
    pub mirror_tolerance: Option<f32>,
}

/// [`edit_mesh`] that optionally makes the edit symmetric across the object's
/// X = 0 plane: moved vertices' mirror partners follow (see `ops::symmetry`).
pub fn edit_mesh_mirrored(
    ctx: &mut Ctx,
    object: &Option<String>,
    ids: Option<&Option<Vec<String>>>,
    label: &str,
    mirror: Symmetry,
    f: impl FnOnce(&mut Mesh, &Selection) -> polly_core::Result<Selection>,
) -> Result<Out> {
    let (mirror, tol) = (mirror.mirror, mirror.mirror_tolerance);
    let report = std::cell::RefCell::new(None);
    let mut out = edit_mesh(ctx, object, ids, label, |m, s| {
        let before = mirror.then(|| m.clone());
        let r = f(m, s)?;
        if let Some(b) = before {
            *report.borrow_mut() = Some(polly_core::ops::symmetry::mirror_edit(&b, m, tol));
        }
        Ok(r)
    })?;
    if let Some(r) = report.into_inner() {
        out.json["mirror"] = json!({"partners_moved": r.mirrored});
        if !r.unpaired.is_empty() {
            out.json["mirror"]["unpaired"] = json!(format!(
                "{} edited vertices have no mirror partner across x = 0 (their other side didn't follow; a larger mirror_tolerance may pair them): {}",
                r.unpaired.len(),
                polly_core::ids::compress("v", r.unpaired.iter().copied())
            ));
        }
    }
    Ok(out)
}

/// Run a scene-level edit as an undoable step.
pub fn edit_scene<T>(ctx: &mut Ctx, label: &str, f: impl FnOnce(&mut Scene) -> polly_core::Result<T>) -> Result<T> {
    Ok(ctx.doc.edit(label, f)?)
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Pivot {
    /// "center" (bounding-box center of the elements, default), "centroid" (vertex average), "origin" (object origin)
    Named(String),
    Point([f32; 3]),
}

pub fn pivot(mesh: &Mesh, verts: &std::collections::BTreeSet<VertId>, p: &Option<Pivot>) -> Result<Vec3> {
    Ok(match p {
        None => polly_core::ops::transform::bounds_center(mesh, verts),
        Some(Pivot::Point(p)) => Vec3::from_array(*p),
        Some(Pivot::Named(n)) => match n.as_str() {
            "center" | "bounds" => polly_core::ops::transform::bounds_center(mesh, verts),
            "centroid" | "median" => polly_core::ops::transform::centroid(mesh, verts),
            "origin" => Vec3::ZERO,
            other => bail!("unknown pivot '{other}' (center, centroid, origin, or [x,y,z])"),
        },
    })
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Scalar3 {
    Uniform(f32),
    Vec([f32; 3]),
}

impl Scalar3 {
    pub fn vec(&self) -> Vec3 {
        match self {
            Scalar3::Uniform(s) => Vec3::splat(*s),
            Scalar3::Vec(v) => Vec3::from_array(*v),
        }
    }
}

pub fn round3(v: Vec3) -> [f32; 3] {
    let r = |x: f32| (x * 10000.0).round() / 10000.0 + 0.0;
    [r(v.x), r(v.y), r(v.z)]
}

pub fn ok(msg: impl Into<String>) -> Out {
    Out::json(Value::String(msg.into()))
}

/// `null`, `""` or `false` -> `Some(None)` (remove); a value -> `Some(Some(v))`;
/// absent -> `None` (unchanged, via `#[serde(default)]`).
pub fn nullable<'de, D: serde::Deserializer<'de>, T: serde::de::DeserializeOwned>(d: D) -> Result<Option<Option<T>>, D::Error> {
    let v = <Value as Deserialize>::deserialize(d)?;
    match v {
        Value::Null | Value::Bool(false) => Ok(Some(None)),
        Value::String(s) if s.is_empty() => Ok(Some(None)),
        v => serde_json::from_value(v).map(|x| Some(Some(x))).map_err(serde::de::Error::custom),
    }
}


/// Where paint goes beyond the texture: a material slot, a subset of channels,
/// and an optional per-texel strength map. Flattened into the paint tools.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct PaintTarget {
    /// Material slot whose texture is painted when `texture` is omitted: base (default), metallic_roughness, normal, occlusion, emissive.
    pub slot: Option<String>,
    /// Write only these channels: "r", "g", "b", "a" or combinations ("rgb"), or a name: roughness (g), metallic (b), occlusion / ao (r).
    pub channel: Option<String>,
    /// With `channel`: the value 0..1 written to those channels (instead of the colour's).
    pub value: Option<f32>,
    /// Paint strength per texel from a channel of another texture (e.g. a baked cavity or curvature map).
    pub strength_map: Option<StrengthArg>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct StrengthArg {
    pub texture: String,
    /// "r" (default), "g", "b" or "a".
    pub channel: Option<String>,
    #[serde(default)]
    pub invert: bool,
    /// Multiplier (default 1).
    pub amount: Option<f32>,
}

fn channel_index(c: char) -> Option<usize> {
    "rgba".find(c)
}

impl PaintTarget {
    pub fn slot(&self) -> Result<polly_core::scene::Slot> {
        match self.slot.as_deref() {
            None => Ok(polly_core::scene::Slot::Base),
            Some(n) => polly_core::scene::Slot::parse(n).ok_or_else(|| anyhow::anyhow!("unknown slot '{n}' (base, metallic_roughness, normal, occlusion, emissive)")),
        }
    }

    pub fn channels(&self) -> Result<[bool; 4]> {
        let Some(c) = self.channel.as_deref() else { return Ok([true; 4]) };
        let mut m = [false; 4];
        match c {
            "roughness" => m[1] = true,
            "metallic" => m[2] = true,
            "occlusion" | "ao" => m[0] = true,
            other => {
                for ch in other.chars() {
                    m[channel_index(ch).ok_or_else(|| anyhow::anyhow!("unknown channel '{other}' (r, g, b, a, rgb..., roughness, metallic, occlusion)"))?] = true;
                }
            }
        }
        Ok(m)
    }

    /// The colour to paint: `value` in the selected channels, if given.
    pub fn color(&self, c: polly_paint::Rgba) -> Result<polly_paint::Rgba> {
        let Some(v) = self.value else { return Ok(c) };
        if self.channel.is_none() {
            bail!("`value` needs `channel`");
        }
        let b = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let m = self.channels()?;
        Ok(std::array::from_fn(|i| if m[i] { b } else { c[i] }))
    }

    /// Apply channel selection and strength map to a paint.
    pub fn paint(&self, scene: &Scene, p: polly_paint::Paint) -> Result<polly_paint::Paint> {
        let mut p = p.with_channels(self.channels()?);
        p.color = self.color(p.color)?;
        if self.channel.is_some() {
            // Palette snapping is for colours, not data channels.
            p.palette = None;
        }
        if let Some(sm) = &self.strength_map {
            let t = &scene.textures[&scene.resolve_texture(&sm.texture)?];
            let ch = match sm.channel.as_deref() {
                None => 0,
                Some(c) => c.chars().next().and_then(channel_index).filter(|_| c.len() == 1).ok_or_else(|| anyhow::anyhow!("strength_map channel must be r, g, b or a"))?,
            };
            p = p.with_strength(Some(polly_paint::StrengthMap {
                width: t.width,
                height: t.height,
                pixels: t.pixels.clone(),
                channel: ch,
                invert: sm.invert,
                amount: sm.amount.unwrap_or(1.0),
            }));
        }
        Ok(p)
    }
}

/// The texture in `slot` of the materials of `faces`, when they agree.
pub fn texture_of_faces_slot(scene: &Scene, obj: u32, faces: &std::collections::BTreeSet<polly_core::mesh::FaceId>, slot: polly_core::scene::Slot) -> Result<polly_core::scene::TextureId> {
    let o = scene.object(obj)?;
    let mats: std::collections::BTreeSet<u32> = faces.iter().filter_map(|f| o.mesh.faces.get(f)).map(|f| f.material).collect();
    let texs: std::collections::BTreeSet<_> = mats.iter().filter_map(|m| scene.materials.get(m)?.slot(slot)).collect();
    match texs.len() {
        1 => Ok(*texs.iter().next().unwrap()),
        0 => bail!("these faces' material has no {} texture; pass `texture`, or create_texture with `material`, `slot` and `object`", slot.name()),
        _ => bail!("faces use several {} textures; pass `texture` explicitly", slot.name()),
    }
}
