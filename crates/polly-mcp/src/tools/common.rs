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
    let warnings = o.mesh.validate();
    if !warnings.is_empty() {
        report["warnings"] = json!(warnings);
    }
    let out = Out::json(report);
    crate::tools::view::maybe_preview(ctx, id, out)
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
