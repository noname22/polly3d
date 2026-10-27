//! Subdivision surfaces: Catmull–Clark as a modifier over the editable cage.

use anyhow::{Result, bail};
use polly_core::mesh::Edge;
use polly_core::subdiv::{MAX_LEVEL, Subdivision};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "set_subdivision",
            "Smooth an object with Catmull–Clark subdivision as a modifier: the mesh stays the low-poly cage you edit by id (overlays, labels, pick and paint use the cage), renders and the viewer show the smooth surface, and export_glb writes the subdivided mesh (weights, morphs, UVs and parts carried over). level 0 = off. Each level multiplies the faces by 4; use creases (set_crease) for hard edges.",
            false,
            set_subdivision,
        ),
        def(
            "set_crease",
            "Set subdivision crease sharpness on edges: 0 = smooth (removes), 1 = sharp for one level, >= the level = fully sharp, fractions in between. Boundary edges are always sharp. Separate from mark_sharp (which only affects shading normals).",
            false,
            set_crease,
        ),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct SetSubdivision {
    object: Option<String>,
    /// Level for renders and the viewer (0 = off, max 4).
    level: u8,
    /// Level written by export_glb (default: same as `level`).
    export_level: Option<u8>,
}

/// Triangles after `level` levels of Catmull–Clark (all quads after the first).
pub fn subdivided_tris(mesh: &polly_core::mesh::Mesh, level: u8) -> usize {
    if level == 0 {
        return mesh.triangle_count();
    }
    let quads: usize = mesh.faces.values().map(|f| f.len()).sum();
    quads * 4usize.pow(level as u32 - 1) * 2
}

fn set_subdivision(ctx: &mut Ctx, a: SetSubdivision) -> Result<Out> {
    if a.level > MAX_LEVEL || a.export_level.is_some_and(|l| l > MAX_LEVEL) {
        bail!("subdivision levels go up to {MAX_LEVEL}");
    }
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let sub = (a.level > 0 || a.export_level.is_some_and(|l| l > 0)).then_some(Subdivision { level: a.level, export_level: a.export_level.filter(|l| *l != a.level) });
    edit_scene(ctx, "set subdivision", |s| {
        s.object_mut(id)?.subdivision = sub;
        Ok(())
    })?;
    let o = ctx.doc.scene.object(id)?;
    let shown = sub.map_or(0, |s| s.level);
    let exported = sub.map_or(0, |s| s.export());
    let mut j = json!({
        "object": o.name,
        "level": shown,
        "export_level": exported,
        "cage_tris": o.mesh.triangle_count(),
        "shown_tris": subdivided_tris(&o.mesh, shown),
        "exported_tris": subdivided_tris(&o.mesh, exported),
    });
    if o.mesh.faces.values().any(|f| f.len() != 4) {
        j["note"] = json!("non-quad faces subdivide into irregular patches (small pinches); quads give the cleanest surface");
    }
    if shown > 0 && o.mesh.faces.values().any(|f| !f.smooth) {
        j["shading"] = json!("some faces are flat-shaded, so the surface shows facets: set_smooth for a smooth look");
    }
    if subdivided_tris(&o.mesh, exported) > 200_000 {
        j["warning"] = json!("over 200k exported triangles: lower export_level for a game-ready mesh");
    }
    let out = Out::json(j);
    crate::tools::view::maybe_preview(ctx, id, out)
}

#[derive(Deserialize, JsonSchema)]
struct SetCrease {
    object: Option<String>,
    /// Edges (ids like "e3-9", "selection", or faces/verts whose edges to use).
    elements: Option<Vec<String>>,
    /// Sharpness (0 removes the crease).
    value: f32,
}

fn set_crease(ctx: &mut Ctx, a: SetCrease) -> Result<Out> {
    if !(0.0..=10.0).contains(&a.value) {
        bail!("crease sharpness must be 0..10");
    }
    let mut count = 0;
    let out = edit_mesh(ctx, &a.object, Some(&a.elements), "set crease", |m, sel| {
        let edges: Vec<Edge> = sel.implied_edges(m).into_iter().collect();
        for e in edges {
            count += 1;
            if a.value > 0.0 {
                m.creases.insert(e, a.value);
            } else {
                m.creases.remove(&e);
            }
        }
        Ok(Default::default())
    })?;
    let mut out = out;
    out.json["creased_edges"] = json!(count);
    Ok(out)
}
