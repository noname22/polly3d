use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::Result;
use polly_core::mesh::FaceId;
use polly_core::scene::{ObjectId, Scene};
use polly_uv::{PackOptions, Projection, TransformUv, UnwrapOptions};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def("mark_seams", "Mark (or clear) UV seams on edges. Faces/verts given are converted to their edges.", false, mark_seams),
        def(
            "auto_seams",
            "Mark seams on edges sharper than an angle, plus whatever extra cuts make every chart unwrappable (cylinders, spheres, tori).",
            false,
            auto_seams,
        ),
        def(
            "project_uv",
            "Assign UVs by projection: planar, box (per-face dominant axis), cylindrical, spherical, or per_face (each face fills the whole texture — tiles/trim sheets). Result is fit into 0..1.",
            false,
            project_uv,
        ),
        def(
            "unwrap",
            "Unwrap faces along seams (LSCM) and pack the charts into the texture with pixel padding and texel snapping. Returns islands with pixel rects, stretch and overlap report.",
            false,
            unwrap,
        ),
        def("pack_uvs", "Re-pack existing UV islands (keeps their shapes) into the texture.", false, pack_uvs),
        def("uv_report", "Report UV islands, pixel rects, coverage, stretch, overlaps and texel density without changing anything.", true, uv_report),
        def("transform_uvs", "Move/scale/rotate the UVs of faces (e.g. to place an island on a specific texture area).", false, transform_uvs),
    ]
}

/// Size of the faces' texture (via material), else 64x64.
fn texture_size(scene: &Scene, id: ObjectId, faces: &BTreeSet<FaceId>) -> [u32; 2] {
    let o = &scene.objects[&id];
    faces
        .iter()
        .filter_map(|f| o.mesh.faces.get(f))
        .find_map(|f| scene.materials.get(&f.material)?.texture.and_then(|t| scene.textures.get(&t)))
        .map_or([64, 64], |t| [t.width, t.height])
}

fn all_default(e: &Option<Vec<String>>) -> Option<Vec<String>> {
    Some(e.clone().unwrap_or_else(|| vec!["all".into()]))
}

#[derive(Deserialize, JsonSchema)]
struct Seams {
    object: Option<String>,
    elements: Option<Vec<String>>,
    #[serde(default = "yes")]
    on: bool,
}

fn yes() -> bool {
    true
}

fn mark_seams(ctx: &mut Ctx, a: Seams) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "mark seams", |m, s| {
        let mut e = s.implied_edges(m);
        for f in &s.faces {
            e.extend(m.faces[f].edges());
        }
        polly_uv::mark_seams(m, &e, a.on);
        Ok(Default::default())
    })
}

#[derive(Deserialize, JsonSchema)]
struct AutoSeams {
    object: Option<String>,
    /// Faces to consider (default all).
    elements: Option<Vec<String>>,
    /// Dihedral angle threshold in degrees (default 60).
    angle: Option<f32>,
}

fn auto_seams(ctx: &mut Ctx, a: AutoSeams) -> Result<Out> {
    let ids = all_default(&a.elements);
    let mut n = 0;
    let mut out = edit_mesh(ctx, &a.object, Some(&ids), "auto seams", |m, s| {
        n = polly_uv::auto_seams(m, &s.implied_faces(m), a.angle.unwrap_or(60.0));
        Ok(Default::default())
    })?;
    out.json["seams_marked"] = json!(n);
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
struct Project {
    object: Option<String>,
    elements: Option<Vec<String>>,
    projection: Projection,
}

fn project_uv(ctx: &mut Ctx, a: Project) -> Result<Out> {
    let ids = all_default(&a.elements);
    edit_mesh(ctx, &a.object, Some(&ids), "project uv", |m, s| {
        polly_uv::project(m, &s.implied_faces(m), &a.projection)?;
        Ok(Default::default())
    })
}

#[derive(Deserialize, JsonSchema)]
struct Unwrap {
    object: Option<String>,
    /// Faces to unwrap (default all).
    elements: Option<Vec<String>>,
    /// Texture size used for padding/snapping (default: the faces' texture size, else 64x64).
    texture_size: Option<[u32; 2]>,
    #[serde(default)]
    options: UnwrapOptions,
}

fn uv_edit(
    ctx: &mut Ctx,
    object: &Option<String>,
    ids: &Option<Vec<String>>,
    label: &str,
    size: Option<[u32; 2]>,
    f: impl FnOnce(&mut polly_core::mesh::Mesh, &BTreeSet<FaceId>, [u32; 2]) -> polly_core::Result<polly_uv::UvReport>,
) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, object)?;
    let sel = elements(&ctx.doc.scene, id, &all_default(ids))?;
    let faces = sel.implied_faces(&ctx.doc.scene.object(id)?.mesh);
    let size = size.unwrap_or_else(|| texture_size(&ctx.doc.scene, id, &faces));
    let report = edit_scene(ctx, label, |s| {
        let o = s.object_mut(id)?;
        f(Arc::make_mut(&mut o.mesh), &faces, size)
    })?;
    let mut j = serde_json::to_value(&report)?;
    j["texture_size"] = json!(size);
    j["hint"] = json!("render_uv_layout shows the islands over the texture");
    Ok(Out::json(j))
}

fn unwrap(ctx: &mut Ctx, a: Unwrap) -> Result<Out> {
    let mut o = a.options.clone();
    uv_edit(ctx, &a.object, &a.elements, "unwrap", a.texture_size, |m, faces, size| {
        o.pack.texture_size = size;
        polly_uv::unwrap(m, faces, &o)
    })
}

#[derive(Deserialize, JsonSchema)]
struct Pack {
    object: Option<String>,
    elements: Option<Vec<String>>,
    texture_size: Option<[u32; 2]>,
    #[serde(default)]
    options: PackOptions,
}

fn pack_uvs(ctx: &mut Ctx, a: Pack) -> Result<Out> {
    let mut o = a.options.clone();
    uv_edit(ctx, &a.object, &a.elements, "pack uvs", a.texture_size, |m, faces, size| {
        o.texture_size = size;
        polly_uv::pack(m, faces, &o)
    })
}

#[derive(Deserialize, JsonSchema)]
struct Report {
    object: Option<String>,
    elements: Option<Vec<String>>,
    texture_size: Option<[u32; 2]>,
}

fn uv_report(ctx: &mut Ctx, a: Report) -> Result<Out> {
    let s = &ctx.doc.scene;
    let id = resolve(s, &a.object)?;
    let sel = elements(s, id, &all_default(&a.elements))?;
    let mesh = &s.object(id)?.mesh;
    let faces = sel.implied_faces(mesh);
    let size = a.texture_size.unwrap_or_else(|| texture_size(s, id, &faces));
    let mut j = serde_json::to_value(polly_uv::report(mesh, &faces, size))?;
    j["texture_size"] = json!(size);
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct Transform {
    object: Option<String>,
    elements: Option<Vec<String>>,
    #[serde(flatten)]
    transform: TransformUv,
}

fn transform_uvs(ctx: &mut Ctx, a: Transform) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "transform uvs", |m, s| {
        polly_uv::transform_uvs(m, &s.implied_faces(m), &a.transform)?;
        Ok(Default::default())
    })
}
