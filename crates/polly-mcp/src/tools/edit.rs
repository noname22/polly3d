use std::collections::BTreeSet;

use anyhow::Result;
use polly_core::error::invalid;
use polly_core::glam::{Quat, Vec3, Vec4};
use polly_core::ids::Selection;
use polly_core::ops::{bevel, cut, extrude, topology, transform};
use polly_core::select::edge_arg;
use schemars::JsonSchema;
use serde::Deserialize;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def("move", "Translate elements (verts/edges/faces) by an offset in object space.", false, move_),
        def(
            "rotate",
            "Rotate elements by Euler XYZ degrees, or by `angle` degrees around `axis`, about a pivot.",
            false,
            rotate,
        ),
        def("scale", "Scale elements about a pivot (uniform number or [x,y,z]). Use 0 on an axis to flatten.", false, scale),
        def("set_position", "Set absolute coordinates of elements' vertices; omitted axes are unchanged (e.g. flatten to y=0).", false, set_position),
        def("snap", "Snap vertex positions to a grid step.", false, snap),
        def(
            "extrude",
            "Extrude faces (as one region, or individually) along their normal or a direction; or extrude boundary edges by an offset. The extruded faces/edges become the selection.",
            false,
            extrude_,
        ),
        def("inset", "Inset faces by a thickness (in their plane), optionally pushing inner faces by depth (negative = into the mesh). Inner faces become the selection.", false, inset),
        def("bevel", "Bevel/chamfer edges with a width and segment count (1 = flat chamfer, 2+ = rounded).", false, bevel_),
        def("loop_cut", "Cut a loop through the quad ring crossing an edge. cuts>1 spaces evenly; with cuts=1, fraction picks the position along the ring edges.", false, loop_cut),
        def("split_edges", "Insert evenly spaced vertices along edges.", false, split_edges),
        def("connect_verts", "Split faces by connecting a path of vertices (each consecutive pair must share a face). Use after split_edges to knife-cut.", false, connect_verts),
        def("subdivide", "Split faces into smaller faces (midpoint subdivision, repeated `levels` times). Neighbours stay connected.", false, subdivide),
        def("delete", "Delete verts (and their faces), edges (and their faces) or faces.", false, delete),
        def("merge", "Merge vertices: mode \"center\" collapses all to one vertex at their center (or `target`), mode \"distance\" welds vertices closer than `distance`.", false, merge),
        def("dissolve", "Dissolve edges (merge the two faces), vertices (merge surrounding faces / remove 2-edge verts) or faces (merge a connected face region into one n-gon).", false, dissolve),
        def("flip_normals", "Reverse face winding.", false, flip),
        def("recalc_normals", "Make face winding consistent and pointing outward (or inward).", false, recalc),
        def("triangulate", "Triangulate faces (ear clipping; handles concave n-gons).", false, triangulate),
        def("fill_holes", "Fill boundary holes with n-gons (only holes touching the given edges, or all holes).", false, fill_holes),
        def("bridge", "Remove two faces with the same vertex count and connect their outlines with quads (tunnels, handles, joining parts).", false, bridge),
        def("mirror", "Mirror the whole mesh across the object's local plane axis=0, welding vertices on the plane. Model one half, then mirror.", false, mirror),
        def("duplicate", "Copy faces (as new disconnected geometry) with an offset. The copies become the selection.", false, duplicate),
        def("add_vertex", "Add loose vertices (connect them with add_face).", false, add_vertex),
        def("add_face", "Create a face from existing vertex ids (CCW seen from outside).", false, add_face),
        def("set_smooth", "Set faces to smooth or flat shading.", false, set_smooth),
        def("mark_sharp", "Mark/clear edges as sharp (shading split on smooth faces).", false, mark_sharp),
        def("set_vertex_color", "Set vertex colors (per face corner) for elements: corners of selected faces, or all corners at selected vertices.", false, set_vertex_color),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct Move {
    /// Object name (default: active object).
    object: Option<String>,
    /// Element ids like ["f1","v3","e2-5"], "all", or "selection" (default: current selection).
    elements: Option<Vec<String>>,
    offset: [f32; 3],
}

fn move_(ctx: &mut Ctx, a: Move) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "move", |m, s| {
        transform::translate(m, &s.touched_verts(m), Vec3::from_array(a.offset))?;
        Ok(s.clone())
    })
}

#[derive(Deserialize, JsonSchema)]
struct Rotate {
    object: Option<String>,
    elements: Option<Vec<String>>,
    /// Euler XYZ degrees.
    rotation: Option<[f32; 3]>,
    axis: Option<[f32; 3]>,
    /// Degrees around `axis`.
    angle: Option<f32>,
    /// "center" (default), "centroid", "origin", or [x,y,z].
    pivot: Option<Pivot>,
}

fn rotate(ctx: &mut Ctx, a: Rotate) -> Result<Out> {
    let q = match (a.rotation, a.axis, a.angle) {
        (Some(r), _, _) => polly_core::scene::quat_from_euler_degrees(r),
        (None, Some(ax), Some(ang)) => Quat::from_axis_angle(Vec3::from_array(ax).normalize(), ang.to_radians()),
        _ => anyhow::bail!("give `rotation` [x,y,z] degrees, or `axis` and `angle`"),
    };
    let piv = a.pivot.clone();
    edit_mesh(ctx, &a.object, Some(&a.elements), "rotate", |m, s| {
        let vs = s.touched_verts(m);
        let p = pivot(m, &vs, &piv).map_err(|e| polly_core::Error::Invalid(e.to_string()))?;
        transform::rotate(m, &vs, q, p)?;
        Ok(s.clone())
    })
}

#[derive(Deserialize, JsonSchema)]
struct Scale {
    object: Option<String>,
    elements: Option<Vec<String>>,
    factor: Scalar3,
    pivot: Option<Pivot>,
}

fn scale(ctx: &mut Ctx, a: Scale) -> Result<Out> {
    let piv = a.pivot.clone();
    edit_mesh(ctx, &a.object, Some(&a.elements), "scale", |m, s| {
        let vs = s.touched_verts(m);
        let p = pivot(m, &vs, &piv).map_err(|e| polly_core::Error::Invalid(e.to_string()))?;
        transform::scale(m, &vs, a.factor.vec(), p)?;
        Ok(s.clone())
    })
}

#[derive(Deserialize, JsonSchema)]
struct SetPosition {
    object: Option<String>,
    elements: Option<Vec<String>>,
    x: Option<f32>,
    y: Option<f32>,
    z: Option<f32>,
}

fn set_position(ctx: &mut Ctx, a: SetPosition) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "set position", |m, s| {
        transform::set_position(m, &s.touched_verts(m), [a.x, a.y, a.z])?;
        Ok(s.clone())
    })
}

#[derive(Deserialize, JsonSchema)]
struct Snap {
    object: Option<String>,
    elements: Option<Vec<String>>,
    step: f32,
}

fn snap(ctx: &mut Ctx, a: Snap) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "snap", |m, s| {
        transform::snap(m, &s.touched_verts(m), a.step)?;
        Ok(s.clone())
    })
}

#[derive(Deserialize, JsonSchema)]
struct Extrude {
    object: Option<String>,
    /// Faces, or edges (for edge extrusion). Default: current selection.
    elements: Option<Vec<String>>,
    /// Distance along the normal (or along `direction`). For edges, the offset is direction*distance.
    distance: f32,
    /// Optional direction instead of the face normal (required for edges unless the edges border faces).
    direction: Option<[f32; 3]>,
    /// Extrude each face separately along its own normal.
    #[serde(default)]
    individual: bool,
}

fn extrude_(ctx: &mut Ctx, a: Extrude) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "extrude", |m, s| {
        let faces = s.implied_faces(m);
        if !faces.is_empty() {
            return extrude::extrude_faces(m, &faces, a.distance, a.direction.map(Vec3::from_array), a.individual);
        }
        let edges = s.implied_edges(m);
        if edges.is_empty() {
            return invalid("select faces or edges to extrude");
        }
        let dir = match a.direction {
            Some(d) => Vec3::from_array(d).normalize_or_zero(),
            None => {
                // Outward along the faces the edges border, in their plane.
                let ef = m.edge_faces();
                let mut d = Vec3::ZERO;
                for e in &edges {
                    if let Some(f) = ef.get(e).and_then(|f| f.first()) {
                        let face = &m.faces[f];
                        let mid = (m.pos(e.0) + m.pos(e.1)) / 2.0;
                        d += (mid - m.face_center(face)).normalize_or_zero();
                    }
                }
                d.normalize_or_zero()
            }
        };
        extrude::extrude_edges(m, &edges, dir * a.distance)
    })
}

#[derive(Deserialize, JsonSchema)]
struct Inset {
    object: Option<String>,
    elements: Option<Vec<String>>,
    thickness: f32,
    #[serde(default)]
    depth: f32,
    /// Inset each face separately (default false = inset the region border).
    #[serde(default)]
    individual: bool,
}

fn inset(ctx: &mut Ctx, a: Inset) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "inset", |m, s| {
        extrude::inset_faces(m, &s.implied_faces(m), a.thickness, a.depth, a.individual)
    })
}

#[derive(Deserialize, JsonSchema)]
struct Bevel {
    object: Option<String>,
    /// Edges (faces/verts are converted to their edges). Default: current selection.
    elements: Option<Vec<String>>,
    width: f32,
    #[serde(default = "one")]
    segments: u32,
}

fn one() -> u32 {
    1
}

fn edges_of(m: &polly_core::mesh::Mesh, s: &Selection) -> BTreeSet<polly_core::mesh::Edge> {
    let mut e = s.implied_edges(m);
    for f in &s.faces {
        e.extend(m.faces[f].edges());
    }
    e
}

fn bevel_(ctx: &mut Ctx, a: Bevel) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "bevel", |m, s| bevel::bevel_edges(m, &edges_of(m, s), a.width, a.segments))
}

#[derive(Deserialize, JsonSchema)]
struct LoopCut {
    object: Option<String>,
    /// An edge crossed by the loop, e.g. "e3-7".
    edge: String,
    #[serde(default = "one")]
    cuts: u32,
    /// Position along the ring edges when cuts = 1 (0..1, default 0.5).
    fraction: Option<f32>,
}

fn loop_cut(ctx: &mut Ctx, a: LoopCut) -> Result<Out> {
    edit_mesh(ctx, &a.object, None, "loop cut", |m, _| {
        let e = edge_arg(m, &a.edge)?;
        cut::loop_cut(m, e, a.cuts, a.fraction.unwrap_or(0.5))
    })
}

#[derive(Deserialize, JsonSchema)]
struct SplitEdges {
    object: Option<String>,
    elements: Option<Vec<String>>,
    #[serde(default = "one")]
    cuts: u32,
}

fn split_edges(ctx: &mut Ctx, a: SplitEdges) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "split edges", |m, s| cut::split_edges(m, &s.implied_edges(m), a.cuts))
}

#[derive(Deserialize, JsonSchema)]
struct ConnectVerts {
    object: Option<String>,
    /// Ordered vertex ids, e.g. ["v3","v9","v12"].
    path: Vec<String>,
}

fn vert_ids(path: &[String]) -> Result<Vec<u32>> {
    path.iter()
        .map(|s| match polly_core::ids::parse(s)? {
            polly_core::ids::Element::Vert(v) => Ok(v),
            _ => Err(anyhow::anyhow!("'{s}' is not a vertex id")),
        })
        .collect()
}

fn connect_verts(ctx: &mut Ctx, a: ConnectVerts) -> Result<Out> {
    let path = vert_ids(&a.path)?;
    edit_mesh(ctx, &a.object, None, "connect verts", |m, _| cut::connect_verts(m, &path))
}

#[derive(Deserialize, JsonSchema)]
struct Subdivide {
    object: Option<String>,
    elements: Option<Vec<String>>,
    #[serde(default = "one")]
    levels: u32,
}

fn subdivide(ctx: &mut Ctx, a: Subdivide) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "subdivide", |m, s| {
        let mut faces = s.implied_faces(m);
        let mut out = Selection::default();
        for _ in 0..a.levels.clamp(1, 4) {
            out = cut::subdivide(m, &faces)?;
            faces = out.faces.clone();
        }
        Ok(out)
    })
}

#[derive(Deserialize, JsonSchema)]
struct Delete {
    object: Option<String>,
    elements: Option<Vec<String>>,
    /// Keep vertices left without faces (default false).
    #[serde(default)]
    keep_loose: bool,
}

fn delete(ctx: &mut Ctx, a: Delete) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "delete", |m, s| {
        topology::delete(m, s, a.keep_loose)?;
        Ok(Selection::default())
    })
}

#[derive(Deserialize, JsonSchema)]
struct Merge {
    object: Option<String>,
    elements: Option<Vec<String>>,
    /// "center" (default) or "distance".
    mode: Option<String>,
    /// Weld distance for mode=distance (default 0.001).
    distance: Option<f32>,
    /// Target position for mode=center.
    target: Option<[f32; 3]>,
}

fn merge(ctx: &mut Ctx, a: Merge) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "merge", |m, s| {
        let vs = s.touched_verts(m);
        match a.mode.as_deref().unwrap_or("center") {
            "distance" => {
                topology::merge_by_distance(m, &vs, a.distance.unwrap_or(0.001))?;
                Ok(Selection::default())
            }
            "center" | "collapse" => {
                let v = topology::collapse(m, &vs, a.target.map(Vec3::from_array))?;
                Ok(Selection { verts: [v].into(), ..Default::default() })
            }
            other => invalid(format!("unknown merge mode '{other}'")),
        }
    })
}

#[derive(Deserialize, JsonSchema)]
struct Elements {
    object: Option<String>,
    /// Element ids, "all" or "selection" (default).
    elements: Option<Vec<String>>,
}

fn dissolve(ctx: &mut Ctx, a: Elements) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "dissolve", |m, s| {
        if !s.faces.is_empty() {
            let f = topology::merge_faces(m, &s.faces)?;
            return Ok(Selection { faces: [f].into(), ..Default::default() });
        }
        if !s.edges.is_empty() {
            return topology::dissolve_edges(m, &s.edges);
        }
        topology::dissolve_verts(m, &s.verts)
    })
}

fn flip(ctx: &mut Ctx, a: Elements) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "flip normals", |m, s| {
        topology::flip(m, &s.implied_faces(m))?;
        Ok(s.clone())
    })
}

#[derive(Deserialize, JsonSchema)]
struct Recalc {
    object: Option<String>,
    /// Default: all faces.
    elements: Option<Vec<String>>,
    #[serde(default)]
    inside: bool,
}

fn recalc(ctx: &mut Ctx, a: Recalc) -> Result<Out> {
    let ids = Some(a.elements.unwrap_or_else(|| vec!["all".into()]));
    edit_mesh(ctx, &a.object, Some(&ids), "recalc normals", |m, s| {
        topology::recalc_normals(m, &s.implied_faces(m), a.inside)?;
        Ok(Selection::default())
    })
}

fn triangulate(ctx: &mut Ctx, a: Elements) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "triangulate", |m, s| topology::triangulate(m, &s.implied_faces(m)))
}

#[derive(Deserialize, JsonSchema)]
struct FillHoles {
    object: Option<String>,
    /// Boundary edges identifying holes (default: fill all holes).
    edges: Option<Vec<String>>,
}

fn fill_holes(ctx: &mut Ctx, a: FillHoles) -> Result<Out> {
    let ids = a.edges.clone();
    edit_mesh(ctx, &a.object, None, "fill holes", |m, cur| {
        let edges = match &ids {
            None => BTreeSet::new(),
            Some(ids) => Selection::parse(ids, m, cur)?.implied_edges(m),
        };
        topology::fill_holes(m, &edges)
    })
}

#[derive(Deserialize, JsonSchema)]
struct Bridge {
    object: Option<String>,
    /// Exactly two face ids.
    faces: [String; 2],
}

fn bridge(ctx: &mut Ctx, a: Bridge) -> Result<Out> {
    let ids = Some(a.faces.to_vec());
    edit_mesh(ctx, &a.object, Some(&ids), "bridge", |m, s| {
        let f: Vec<_> = s.faces.iter().copied().collect();
        if f.len() != 2 {
            return invalid("give exactly two distinct faces");
        }
        topology::bridge_faces(m, f[0], f[1])
    })
}

#[derive(Deserialize, JsonSchema)]
struct Mirror {
    object: Option<String>,
    /// "x", "y" or "z".
    axis: String,
    /// Weld distance for vertices on the mirror plane (default 0.001; 0 disables).
    merge_distance: Option<f32>,
}

fn mirror(ctx: &mut Ctx, a: Mirror) -> Result<Out> {
    let axis = match a.axis.to_lowercase().as_str() {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        _ => anyhow::bail!("axis must be x, y or z"),
    };
    edit_mesh(ctx, &a.object, None, "mirror", |m, _| topology::mirror(m, axis, a.merge_distance.unwrap_or(0.001)))
}

#[derive(Deserialize, JsonSchema)]
struct Duplicate {
    object: Option<String>,
    elements: Option<Vec<String>>,
    #[serde(default)]
    offset: [f32; 3],
}

fn duplicate(ctx: &mut Ctx, a: Duplicate) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "duplicate", |m, s| {
        topology::duplicate(m, &s.implied_faces(m), Vec3::from_array(a.offset))
    })
}

#[derive(Deserialize, JsonSchema)]
struct AddVertex {
    object: Option<String>,
    positions: Vec<[f32; 3]>,
}

fn add_vertex(ctx: &mut Ctx, a: AddVertex) -> Result<Out> {
    edit_mesh(ctx, &a.object, None, "add vertex", |m, _| {
        let verts = a.positions.iter().map(|p| m.add_vert(Vec3::from_array(*p))).collect();
        Ok(Selection { verts, ..Default::default() })
    })
}

#[derive(Deserialize, JsonSchema)]
struct AddFace {
    object: Option<String>,
    /// Vertex ids in CCW order seen from outside.
    verts: Vec<String>,
    /// Material name or index for the new face.
    material: Option<String>,
}

fn add_face(ctx: &mut Ctx, a: AddFace) -> Result<Out> {
    let verts = vert_ids(&a.verts)?;
    let mat = a.material.as_deref().map(|n| ctx.doc.scene.resolve_material(n)).transpose()?.unwrap_or(0);
    edit_mesh(ctx, &a.object, None, "add face", |m, _| {
        let f = m.add_face(verts)?;
        m.faces.get_mut(&f).unwrap().material = mat;
        Ok(Selection { faces: [f].into(), ..Default::default() })
    })
}

#[derive(Deserialize, JsonSchema)]
struct SetSmooth {
    object: Option<String>,
    elements: Option<Vec<String>>,
    smooth: bool,
}

fn set_smooth(ctx: &mut Ctx, a: SetSmooth) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "set smooth", |m, s| {
        for f in s.implied_faces(m) {
            m.faces.get_mut(&f).unwrap().smooth = a.smooth;
        }
        Ok(Selection::default())
    })
}

#[derive(Deserialize, JsonSchema)]
struct MarkEdges {
    object: Option<String>,
    elements: Option<Vec<String>>,
    #[serde(default = "yes")]
    on: bool,
}

fn yes() -> bool {
    true
}

fn mark_sharp(ctx: &mut Ctx, a: MarkEdges) -> Result<Out> {
    edit_mesh(ctx, &a.object, Some(&a.elements), "mark sharp", |m, s| {
        for e in edges_of(m, s) {
            if a.on {
                m.sharp.insert(e);
            } else {
                m.sharp.remove(&e);
            }
        }
        Ok(Selection::default())
    })
}

#[derive(Deserialize, JsonSchema)]
struct VertexColor {
    object: Option<String>,
    elements: Option<Vec<String>>,
    color: [u8; 4],
}

fn set_vertex_color(ctx: &mut Ctx, a: VertexColor) -> Result<Out> {
    let c = Vec4::from_array(super::scene::to_f32(a.color));
    edit_mesh(ctx, &a.object, Some(&a.elements), "vertex color", |m, s| {
        let faces = s.faces.clone();
        let verts = s.verts.clone();
        for (fid, f) in m.faces.iter_mut() {
            for (i, v) in f.verts.iter().enumerate() {
                if faces.contains(fid) || verts.contains(v) {
                    f.corners[i].color = c;
                }
            }
        }
        Ok(Selection::default())
    })
}
