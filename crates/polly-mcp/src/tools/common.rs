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
    let (was, now) = (before.problems(), o.mesh.problems());
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

/// [`edit_mesh`] that optionally makes the edit symmetric across the object's
/// X = 0 plane: moved vertices' mirror partners follow (see `ops::symmetry`).
pub fn edit_mesh_mirrored(
    ctx: &mut Ctx,
    object: &Option<String>,
    ids: Option<&Option<Vec<String>>>,
    label: &str,
    mirror: bool,
    f: impl FnOnce(&mut Mesh, &Selection) -> polly_core::Result<Selection>,
) -> Result<Out> {
    let report = std::cell::RefCell::new(None);
    let mut out = edit_mesh(ctx, object, ids, label, |m, s| {
        let before = mirror.then(|| m.clone());
        let r = f(m, s)?;
        if let Some(b) = before {
            *report.borrow_mut() = Some(polly_core::ops::symmetry::mirror_edit(&b, m));
        }
        Ok(r)
    })?;
    if let Some(r) = report.into_inner() {
        out.json["mirror"] = json!({"partners_moved": r.mirrored});
        if !r.unpaired.is_empty() {
            out.json["mirror"]["unpaired"] = json!(format!(
                "{} edited vertices have no mirror partner across x = 0 (their other side didn't follow): {}",
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
