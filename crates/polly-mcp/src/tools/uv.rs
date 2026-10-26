use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::Result;
use polly_core::mesh::FaceId;
use polly_core::scene::{ObjectId, Scene};
use polly_uv::{PackOptions, Projection, TransformUv, UnwrapOptions};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def("mark_seams", "Mark (or clear) UV seams on edges. Faces/verts given are converted to their edges.", false, mark_seams),
        def(
            "seam_path",
            "Mark (or clear) a seam along the shortest edge path through the given vertices in order, e.g. [\"v12\", \"v40\", \"v77\"] from the armpit down the inside of a sleeve. Waypoints keep the path where you want it.",
            false,
            seam_path,
        ),
        def(
            "seam_plane",
            "Mark (or clear) seams where the mesh crosses a plane: the edges between faces whose centres lie on opposite sides. Good for least-visible cuts on tube shapes: centre back (plane x=0, `within` the back half: point [0,0,0] normal [0,0,-1]), inner leg, under the arm. `within` (a half-space) keeps only part of the crossing.",
            false,
            seam_plane,
        ),
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
        def(
            "pack_uvs",
            "Re-pack existing UV islands (keeps their shapes) into the texture. By default islands get even texel density; `density` gives parts more or less ({\"head\": 2}); options.keep_relative keeps the islands' current relative sizes.",
            false,
            pack_uvs,
        ),
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
struct SeamPath {
    object: Option<String>,
    /// Two or more vertex ids; the path goes through them in order.
    verts: Vec<String>,
    /// Close the path back to the first vertex.
    #[serde(default)]
    closed: bool,
    #[serde(default = "yes")]
    on: bool,
}

fn seam_path(ctx: &mut Ctx, a: SeamPath) -> Result<Out> {
    let mut vs = super::edit::vert_ids(&a.verts)?;
    if vs.len() < 2 {
        anyhow::bail!("give at least two vertices");
    }
    if a.closed {
        vs.push(vs[0]);
    }
    let mut n = 0;
    let mut out = edit_mesh(ctx, &a.object, None, "seam path", |m, _| {
        let mut edges = BTreeSet::new();
        for w in vs.windows(2) {
            for v in w {
                if !m.verts.contains_key(v) {
                    return Err(polly_core::Error::Invalid(format!("no vertex v{v}")));
                }
            }
            let path = polly_uv::shortest_path(m, w[0], w[1])
                .ok_or_else(|| polly_core::Error::Invalid(format!("v{} and v{} are not connected", w[0], w[1])))?;
            edges.extend(path);
        }
        n = edges.len();
        polly_uv::mark_seams(m, &edges, a.on);
        Ok(Default::default())
    })?;
    out.json["seam_edges"] = json!(n);
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
struct HalfSpace {
    point: [f32; 3],
    normal: [f32; 3],
}

#[derive(Deserialize, JsonSchema)]
struct SeamPlane {
    object: Option<String>,
    /// Faces to consider (default all).
    elements: Option<Vec<String>>,
    /// A point on the plane (object space).
    point: [f32; 3],
    /// The plane's normal.
    normal: [f32; 3],
    /// Only crossing edges on the positive side of this half-space.
    within: Option<HalfSpace>,
    #[serde(default = "yes")]
    on: bool,
}

fn seam_plane(ctx: &mut Ctx, a: SeamPlane) -> Result<Out> {
    use polly_core::glam::Vec3;
    let ids = all_default(&a.elements);
    let mut n = 0;
    let within = a.within.as_ref().map(|h| (Vec3::from_array(h.point), Vec3::from_array(h.normal).normalize_or_zero()));
    let mut out = edit_mesh(ctx, &a.object, Some(&ids), "seam plane", |m, s| {
        let edges = polly_uv::plane_seams(m, &s.implied_faces(m), Vec3::from_array(a.point), Vec3::from_array(a.normal), within);
        n = edges.len();
        polly_uv::mark_seams(m, &edges, a.on);
        Ok(Default::default())
    })?;
    out.json["seam_edges"] = json!(n);
    if n == 0 {
        out.json["note"] = json!("the plane crosses no edges between the given faces (check point/normal, or within)");
    }
    Ok(out)
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
    /// List every island (default: the 30 largest).
    #[serde(default)]
    detail: bool,
    /// Texel density per part (or element ids): {"head": 2} gives the head twice the
    /// texels per unit along each axis (4x the area); {"f40..f60": 0.5} halves it.
    density: Option<BTreeMap<String, f32>>,
}

/// Islands listed in a report unless `detail` is set.
const MAX_ISLANDS: usize = 30;

/// Shrink a UV report for the agent: face id lists become ranges
/// ("f400..f507"), and only the largest islands are listed unless `detail`.
fn compact_report(j: &mut Value, detail: bool) {
    let ids = |v: &Value| -> Vec<u32> {
        v.as_array().into_iter().flatten().filter_map(|s| s.as_str()?.strip_prefix('f')?.parse().ok()).collect()
    };
    let compress = |v: &Value| {
        let mut list = ids(v);
        list.sort_unstable();
        json!(polly_core::ids::compress("f", list))
    };
    if let Some(islands) = j.get_mut("islands").and_then(Value::as_array_mut) {
        for isl in islands.iter_mut() {
            let n = ids(&isl["faces"]).len();
            isl["faces"] = compress(&isl["faces"]);
            isl["face_count"] = json!(n);
        }
        let total = islands.len();
        if !detail && total > MAX_ISLANDS {
            islands.sort_by_key(|i| std::cmp::Reverse(i["face_count"].as_u64().unwrap_or(0)));
            islands.truncate(MAX_ISLANDS);
            j["islands_not_listed"] = json!(format!("{} smaller islands (pass detail: true to list all)", total - MAX_ISLANDS));
        }
        j["island_count"] = json!(total);
    }
    for key in ["overlapping_faces", "out_of_bounds_faces", "flipped_faces"] {
        if let Some(v) = j.get(key).cloned() {
            j[key] = compress(&v);
        }
    }
    if let Some(w) = j.get_mut("worst_faces").and_then(Value::as_array_mut) {
        w.truncate(8);
    }
}

fn uv_edit(
    ctx: &mut Ctx,
    object: &Option<String>,
    ids: &Option<Vec<String>>,
    label: &str,
    size: Option<[u32; 2]>,
    detail: bool,
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
    compact_report(&mut j, detail);
    j["texture_size"] = json!(size);
    j["hint"] = json!("render_uv_layout shows the islands over the texture");
    Ok(Out::json(j))
}

/// Per-face density multipliers from {part or element ids: weight}.
fn face_density(scene: &Scene, id: ObjectId, density: &Option<BTreeMap<String, f32>>) -> Result<BTreeMap<FaceId, f32>> {
    let mut out = BTreeMap::new();
    let o = scene.object(id)?;
    for (key, w) in density.iter().flatten() {
        if *w <= 0.0 {
            anyhow::bail!("density for '{key}' must be > 0");
        }
        let by_part: Vec<FaceId> = o.mesh.faces.iter().filter(|(_, f)| f.part.as_deref() == Some(key.as_str())).map(|(i, _)| *i).collect();
        let faces = if by_part.is_empty() {
            polly_core::ids::Selection::parse(std::slice::from_ref(key), &o.mesh, &o.selection)
                .map_err(|e| anyhow::anyhow!("density key '{key}' is neither a part nor element ids: {e}"))?
                .implied_faces(&o.mesh)
                .into_iter()
                .collect()
        } else {
            by_part
        };
        out.extend(faces.into_iter().map(|f| (f, *w)));
    }
    Ok(out)
}

fn unwrap(ctx: &mut Ctx, a: Unwrap) -> Result<Out> {
    let mut o = a.options.clone();
    let id = resolve(&ctx.doc.scene, &a.object)?;
    o.pack.face_density = face_density(&ctx.doc.scene, id, &a.density)?;
    uv_edit(ctx, &a.object, &a.elements, "unwrap", a.texture_size, a.detail, |m, faces, size| {
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
    /// List every island (default: the 30 largest).
    #[serde(default)]
    detail: bool,
    /// Texel density per part (or element ids): {"head": 2} gives the head twice the
    /// texels per unit along each axis (4x the area); {"f40..f60": 0.5} halves it.
    density: Option<BTreeMap<String, f32>>,
}

fn pack_uvs(ctx: &mut Ctx, a: Pack) -> Result<Out> {
    let mut o = a.options.clone();
    let id = resolve(&ctx.doc.scene, &a.object)?;
    o.face_density = face_density(&ctx.doc.scene, id, &a.density)?;
    uv_edit(ctx, &a.object, &a.elements, "pack uvs", a.texture_size, a.detail, |m, faces, size| {
        o.texture_size = size;
        polly_uv::pack(m, faces, &o)
    })
}

#[derive(Deserialize, JsonSchema)]
struct Report {
    object: Option<String>,
    elements: Option<Vec<String>>,
    texture_size: Option<[u32; 2]>,
    /// List every island (default: the 30 largest).
    #[serde(default)]
    detail: bool,
}

fn uv_report(ctx: &mut Ctx, a: Report) -> Result<Out> {
    let s = &ctx.doc.scene;
    let id = resolve(s, &a.object)?;
    let sel = elements(s, id, &all_default(&a.elements))?;
    let mesh = &s.object(id)?.mesh;
    let faces = sel.implied_faces(mesh);
    let size = a.texture_size.unwrap_or_else(|| texture_size(s, id, &faces));
    let mut j = serde_json::to_value(polly_uv::report(mesh, &faces, size))?;
    compact_report(&mut j, a.detail);
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
