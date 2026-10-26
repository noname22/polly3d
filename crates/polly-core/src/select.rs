//! Selection queries so the agent can address geometry without listing ids.

use std::collections::{BTreeSet, VecDeque};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::error::{Result, invalid};
use crate::ids::{self, Element, Selection};
use crate::mesh::{Edge, FaceId, Mesh, VertId};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum Query {
    /// Everything.
    All,
    /// Explicit ids, e.g. ["f1", "v3", "e2-5"].
    Ids { ids: Vec<String> },
    /// Faces whose normal is within `max_angle` degrees of `direction`.
    Normal {
        direction: [f32; 3],
        #[serde(default = "thirty")]
        max_angle: f32,
    },
    /// Vertices inside the axis-aligned box, faces whose center is inside, edges with both ends inside.
    Box { min: [f32; 3], max: [f32; 3] },
    /// Elements on the positive side of a plane (dot(p - point, normal) > threshold).
    PlaneSide {
        point: [f32; 3],
        normal: [f32; 3],
        #[serde(default)]
        threshold: f32,
    },
    /// Edge loop through an edge (continues straight through 4-valent vertices).
    EdgeLoop { edge: String },
    /// Edge ring across quads starting at an edge.
    EdgeRing { edge: String },
    /// Faces of the quad strip crossing an edge (the faces a loop cut would split).
    FaceLoop { edge: String },
    /// All faces connected to the seed elements.
    Linked { seeds: Vec<String> },
    /// Faces using a material index.
    Material { index: u32 },
    /// Open boundary edges.
    Boundary,
    /// Edges whose adjacent faces meet at more than `min_angle` degrees.
    Sharp {
        #[serde(default = "thirty")]
        min_angle: f32,
    },
    /// Vertex / face nearest to a point.
    Nearest { point: [f32; 3] },
    /// Expand the current selection by one ring of adjacent faces.
    Grow,
    /// Remove faces at the border of the current face selection.
    Shrink,
}

fn thirty() -> f32 {
    30.0
}

pub fn edge_arg(mesh: &Mesh, s: &str) -> Result<Edge> {
    match ids::parse(s)? {
        Element::Edge(e) if mesh.edges().contains(&e) => Ok(e),
        Element::Edge(e) => Err(crate::Error::NotFound { kind: "edge", id: ids::edge(e) }),
        _ => invalid(format!("'{s}' is not an edge id (expected e<a>-<b>)")),
    }
}

pub fn run(mesh: &Mesh, q: &Query, current: &Selection) -> Result<Selection> {
    let mut s = Selection::default();
    match q {
        Query::All => s = Selection::all(mesh),
        Query::Ids { ids } => s = Selection::parse(ids, mesh, current)?,
        Query::Normal { direction, max_angle } => {
            let d = Vec3::from_array(*direction).normalize_or_zero();
            let cos = max_angle.to_radians().cos();
            for (&id, f) in &mesh.faces {
                if mesh.face_normal(f).dot(d) >= cos - 1e-6 {
                    s.faces.insert(id);
                }
            }
        }
        Query::Box { min, max } => {
            let (lo, hi) = (Vec3::from_array(*min), Vec3::from_array(*max));
            let inside = |p: Vec3| p.cmpge(lo - 1e-6).all() && p.cmple(hi + 1e-6).all();
            s.verts = mesh.verts.iter().filter(|(_, v)| inside(v.pos)).map(|(i, _)| *i).collect();
            s.faces = mesh.faces.iter().filter(|(_, f)| inside(mesh.face_center(f))).map(|(i, _)| *i).collect();
            s.edges = mesh.edges().into_iter().filter(|e| s.verts.contains(&e.0) && s.verts.contains(&e.1)).collect();
        }
        Query::PlaneSide { point, normal, threshold } => {
            let (p0, n) = (Vec3::from_array(*point), Vec3::from_array(*normal).normalize_or_zero());
            let side = |p: Vec3| (p - p0).dot(n) > *threshold;
            s.verts = mesh.verts.iter().filter(|(_, v)| side(v.pos)).map(|(i, _)| *i).collect();
            s.faces = mesh.faces.iter().filter(|(_, f)| f.verts.iter().all(|v| s.verts.contains(v))).map(|(i, _)| *i).collect();
            s.edges = mesh.edges().into_iter().filter(|e| s.verts.contains(&e.0) && s.verts.contains(&e.1)).collect();
        }
        Query::EdgeLoop { edge } => s.edges = edge_loop(mesh, edge_arg(mesh, edge)?),
        Query::EdgeRing { edge } => {
            let (ring, _) = crate::ops::cut::edge_ring(mesh, edge_arg(mesh, edge)?);
            s.edges = ring.into_iter().map(|(a, b)| Edge::new(a, b)).collect();
        }
        Query::FaceLoop { edge } => {
            let (_, quads) = crate::ops::cut::edge_ring(mesh, edge_arg(mesh, edge)?);
            s.faces = quads.into_iter().collect();
        }
        Query::Linked { seeds } => {
            let seed = Selection::parse(seeds, mesh, current)?;
            s.faces = linked_faces(mesh, &seed);
        }
        Query::Material { index } => {
            s.faces = mesh.faces.iter().filter(|(_, f)| f.material == *index).map(|(i, _)| *i).collect();
        }
        Query::Boundary => s.edges = mesh.boundary_edges(),
        Query::Sharp { min_angle } => {
            let cos = min_angle.to_radians().cos();
            for (e, fs) in mesh.edge_faces() {
                if fs.len() == 2 {
                    let (a, b) = (mesh.face_normal(&mesh.faces[&fs[0]]), mesh.face_normal(&mesh.faces[&fs[1]]));
                    if a.dot(b) < cos {
                        s.edges.insert(e);
                    }
                }
            }
        }
        Query::Nearest { point } => {
            let p = Vec3::from_array(*point);
            if let Some((&v, _)) = mesh.verts.iter().min_by(|a, b| a.1.pos.distance(p).total_cmp(&b.1.pos.distance(p))) {
                s.verts.insert(v);
            }
            if let Some((&f, _)) = mesh
                .faces
                .iter()
                .min_by(|a, b| mesh.face_center(a.1).distance(p).total_cmp(&mesh.face_center(b.1).distance(p)))
            {
                s.faces.insert(f);
            }
        }
        Query::Grow => {
            let verts = current.touched_verts(mesh);
            s = current.clone();
            for (&id, f) in &mesh.faces {
                if f.verts.iter().any(|v| verts.contains(v)) {
                    s.faces.insert(id);
                }
            }
            s.verts = s.touched_verts(mesh);
        }
        Query::Shrink => {
            let faces = current.implied_faces(mesh);
            let vf = mesh.vert_faces();
            s.faces = faces
                .iter()
                .copied()
                .filter(|f| mesh.faces[f].verts.iter().all(|v| vf[v].iter().all(|g| faces.contains(g))))
                .collect();
        }
    }
    Ok(s)
}

pub fn linked_faces(mesh: &Mesh, seed: &Selection) -> BTreeSet<FaceId> {
    let start_verts: BTreeSet<VertId> = seed.touched_verts(mesh);
    let vf = mesh.vert_faces();
    let mut out: BTreeSet<FaceId> = BTreeSet::new();
    let mut q: VecDeque<FaceId> = start_verts.iter().flat_map(|v| vf.get(v).cloned().unwrap_or_default()).collect();
    while let Some(f) = q.pop_front() {
        if !out.insert(f) {
            continue;
        }
        for v in &mesh.faces[&f].verts {
            for g in &vf[v] {
                if !out.contains(g) {
                    q.push_back(*g);
                }
            }
        }
    }
    out
}

pub fn edge_loop(mesh: &Mesh, start: Edge) -> BTreeSet<Edge> {
    let ve = mesh.vert_edges();
    let ef = mesh.edge_faces();
    let mut out = BTreeSet::from([start]);
    for (from, mut at) in [(start.0, start.1), (start.1, start.0)] {
        let mut cur = Edge::new(from, at);
        loop {
            let es = &ve[&at];
            if es.len() != 4 {
                break;
            }
            let faces_of_cur = &ef[&cur];
            let next = es.iter().find(|e| {
                **e != cur && !ef[e].iter().any(|f| faces_of_cur.contains(f))
            });
            let Some(&next) = next else { break };
            if !out.insert(next) {
                break;
            }
            at = next.other(at);
            cur = next;
        }
    }
    out
}

/// Convert a selection to a single element kind.
pub fn convert(mesh: &Mesh, s: &Selection, to: &str) -> Result<Selection> {
    Ok(match to {
        "verts" | "vertices" => Selection { verts: s.touched_verts(mesh), ..Default::default() },
        "faces" => Selection { faces: s.implied_faces(mesh), ..Default::default() },
        "edges" => {
            let mut e = s.implied_edges(mesh);
            for f in &s.faces {
                e.extend(mesh.faces[f].edges());
            }
            Selection { edges: e, ..Default::default() }
        }
        "any" | "" => s.clone(),
        other => return invalid(format!("unknown element kind '{other}' (verts, edges, faces, any)")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;

    #[test]
    fn queries() {
        let m = cube(Vec3::ONE);
        let none = Selection::default();
        let top = run(&m, &Query::Normal { direction: [0., 1., 0.], max_angle: 10.0 }, &none).unwrap();
        assert_eq!(top.faces.len(), 1);
        let grown = run(&m, &Query::Grow, &top).unwrap();
        assert_eq!(grown.faces.len(), 5);
        let e = *m.edges().iter().next().unwrap();
        assert_eq!(edge_loop(&m, e).len(), 1); // cube vertices are 3-valent
        let ring = run(&m, &Query::EdgeRing { edge: ids::edge(e) }, &none).unwrap();
        assert_eq!(ring.edges.len(), 4);
        let sharp = run(&m, &Query::Sharp { min_angle: 30.0 }, &none).unwrap();
        assert_eq!(sharp.edges.len(), 12);
    }

    #[test]
    fn edge_loop_on_grid() {
        let m = crate::primitives::plane(glam::Vec2::ONE, 4, 4);
        // A boundary-free interior edge in the middle row: loops stop at boundary (valence 3).
        let e = *m.edges().iter().find(|e| {
            let (a, b) = (m.pos(e.0), m.pos(e.1));
            a.z.abs() < 1e-6 && b.z.abs() < 1e-6 && a.x.abs() < 0.3 && b.x.abs() < 0.3
        }).unwrap();
        assert_eq!(edge_loop(&m, e).len(), 4);
    }
}
