use std::collections::{BTreeSet, HashMap};

use glam::Vec3;

use crate::error::{Result, invalid};
use crate::ids::Selection;
use crate::mesh::{Edge, FaceId, Mesh, VertId};

/// Duplicate the boundary of a face region and stitch it to the original with
/// side quads. Region faces are rewired to the duplicates. Returns the
/// duplicate map (original -> new) and the boundary half-edges (in original ids,
/// as traversed by region faces).
fn detach_region(mesh: &mut Mesh, region: &BTreeSet<FaceId>) -> (HashMap<VertId, VertId>, Vec<(VertId, VertId, FaceId)>) {
    let mut count: HashMap<Edge, u32> = HashMap::new();
    for f in region {
        for e in mesh.faces[f].edges() {
            *count.entry(e).or_default() += 1;
        }
    }
    let mut boundary = Vec::new();
    for &f in region {
        for (a, b) in mesh.faces[&f].half_edges() {
            if count[&Edge::new(a, b)] == 1 {
                boundary.push((a, b, f));
            }
        }
    }
    let mut dup = HashMap::new();
    for &(a, b, _) in &boundary {
        for v in [a, b] {
            if !dup.contains_key(&v) {
                let p = mesh.pos(v);
                dup.insert(v, mesh.add_vert(p));
            }
        }
    }
    let mut sides = Vec::new();
    for &(a, b, f) in &boundary {
        let face = &mesh.faces[&f];
        let (ia, ib) = (face.index_of(a).unwrap(), face.index_of(b).unwrap());
        let (ca, cb) = (face.corners[ia], face.corners[ib]);
        sides.push((vec![a, b, dup[&b], dup[&a]], vec![ca, cb, cb, ca], face.clone()));
    }
    for f in region {
        let face = mesh.faces.get_mut(f).unwrap();
        for v in face.verts.iter_mut() {
            if let Some(d) = dup.get(v) {
                *v = *d;
            }
        }
    }
    for (verts, mut corners, template) in sides {
        for c in corners.iter_mut() {
            c.uv = glam::Vec2::ZERO;
        }
        mesh.add_face_from(verts, corners, &template).unwrap();
    }
    (dup, boundary)
}

/// Average normal of region faces around each vertex of the region.
fn region_vertex_normals(mesh: &Mesh, region: &BTreeSet<FaceId>) -> HashMap<VertId, Vec3> {
    let mut n: HashMap<VertId, Vec3> = HashMap::new();
    for f in region {
        let face = &mesh.faces[f];
        let fnorm = mesh.face_normal(face);
        for &v in &face.verts {
            *n.entry(v).or_default() += fnorm;
        }
    }
    n.values_mut().for_each(|v| *v = v.normalize_or_zero());
    n
}

fn check_faces(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Result<()> {
    if faces.is_empty() {
        return invalid("no faces given");
    }
    for f in faces {
        mesh.face(*f)?;
    }
    Ok(())
}

/// Extrude faces. `direction = None` moves along the (region-averaged) normal.
/// Returns the extruded (moved) faces.
pub fn extrude_faces(
    mesh: &mut Mesh,
    faces: &BTreeSet<FaceId>,
    distance: f32,
    direction: Option<Vec3>,
    individual: bool,
) -> Result<Selection> {
    check_faces(mesh, faces)?;
    let groups: Vec<BTreeSet<FaceId>> =
        if individual { faces.iter().map(|f| BTreeSet::from([*f])).collect() } else { vec![faces.clone()] };
    for region in &groups {
        let normals = region_vertex_normals(mesh, region);
        let (dup, _) = detach_region(mesh, region);
        let moved: BTreeSet<VertId> = region.iter().flat_map(|f| mesh.faces[f].verts.clone()).collect();
        let inv: HashMap<VertId, VertId> = dup.iter().map(|(a, b)| (*b, *a)).collect();
        for v in moved {
            let orig = inv.get(&v).copied().unwrap_or(v);
            let d = match direction {
                Some(d) => d.normalize_or_zero(),
                None => normals.get(&orig).copied().unwrap_or(Vec3::ZERO),
            };
            mesh.verts.get_mut(&v).unwrap().pos += d * distance;
        }
    }
    Ok(Selection { faces: faces.clone(), ..Default::default() })
}

/// Extrude edges into new quads, offsetting the new edge copies by `offset`.
pub fn extrude_edges(mesh: &mut Mesh, edges: &BTreeSet<Edge>, offset: Vec3) -> Result<Selection> {
    if edges.is_empty() {
        return invalid("no edges given");
    }
    let ef = mesh.edge_faces();
    let mut dup: HashMap<VertId, VertId> = HashMap::new();
    let mut out = Selection::default();
    for e in edges {
        let Some(fs) = ef.get(e) else { return invalid(format!("edge {} does not exist", crate::ids::edge(*e))) };
        let face = mesh.faces[&fs[0]].clone();
        // Traverse opposite to the existing face.
        let (a, b) = if face.has_half_edge(e.0, e.1) { (e.0, e.1) } else { (e.1, e.0) };
        for v in [a, b] {
            if !dup.contains_key(&v) {
                let p = mesh.pos(v) + offset;
                dup.insert(v, mesh.add_vert(p));
            }
        }
        mesh.add_face_like(vec![b, a, dup[&a], dup[&b]], &face)?;
        out.edges.insert(Edge::new(dup[&a], dup[&b]));
    }
    Ok(out)
}

/// Inset faces by `thickness` (in the face plane) and optionally push the inner
/// faces by `depth` along their normal. Returns the inner faces.
pub fn inset_faces(
    mesh: &mut Mesh,
    faces: &BTreeSet<FaceId>,
    thickness: f32,
    depth: f32,
    individual: bool,
) -> Result<Selection> {
    check_faces(mesh, faces)?;
    let groups: Vec<BTreeSet<FaceId>> =
        if individual { faces.iter().map(|f| BTreeSet::from([*f])).collect() } else { vec![faces.clone()] };
    for region in &groups {
        let normals = region_vertex_normals(mesh, region);
        let face_normals: HashMap<FaceId, Vec3> =
            region.iter().map(|f| (*f, mesh.face_normal(&mesh.faces[f]))).collect();
        let (dup, boundary) = detach_region(mesh, region);
        // Boundary loop connectivity in original ids.
        let mut incoming: HashMap<VertId, (VertId, FaceId)> = HashMap::new();
        let mut outgoing: HashMap<VertId, (VertId, FaceId)> = HashMap::new();
        for &(a, b, f) in &boundary {
            outgoing.entry(a).or_insert((b, f));
            incoming.entry(b).or_insert((a, f));
        }
        let inward = |a: Vec3, b: Vec3, n: Vec3| n.cross(b - a).normalize_or_zero();
        let mut new_pos = Vec::new();
        for (&v, &d) in &dup {
            let pv = mesh.pos(v);
            let (Some(&(p, fp)), Some(&(n, fnx))) = (incoming.get(&v), outgoing.get(&v)) else { continue };
            let d1 = inward(mesh.pos(p), pv, face_normals[&fp]);
            let d2 = inward(pv, mesh.pos(n), face_normals[&fnx]);
            let sum = d1 + d2;
            let off = if sum.length() < 1e-4 {
                d1 * thickness
            } else {
                let m = sum.normalize();
                m * thickness / m.dot(d1).max(0.25)
            };
            new_pos.push((d, pv + off));
        }
        for (d, p) in new_pos {
            mesh.verts.get_mut(&d).unwrap().pos = p;
        }
        if depth != 0.0 {
            let inv: HashMap<VertId, VertId> = dup.iter().map(|(a, b)| (*b, *a)).collect();
            let moved: BTreeSet<VertId> = region.iter().flat_map(|f| mesh.faces[f].verts.clone()).collect();
            for v in moved {
                let orig = inv.get(&v).copied().unwrap_or(v);
                let n = normals.get(&orig).copied().unwrap_or(Vec3::ZERO);
                mesh.verts.get_mut(&v).unwrap().pos += n * depth;
            }
        }
    }
    Ok(Selection { faces: faces.clone(), ..Default::default() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;

    fn top_face(m: &Mesh) -> FaceId {
        *m.faces.iter().find(|(_, f)| m.face_normal(f).y > 0.9).unwrap().0
    }

    #[test]
    fn extrude_top_of_cube() {
        let mut m = cube(Vec3::ONE);
        let top = top_face(&m);
        extrude_faces(&mut m, &BTreeSet::from([top]), 1.0, None, false).unwrap();
        assert_eq!((m.verts.len(), m.faces.len()), (12, 10));
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert!((m.bounds().unwrap().1.y - 1.5).abs() < 1e-6);
        // Euler characteristic of a closed genus-0 surface.
        assert_eq!(m.verts.len() as i64 - m.edges().len() as i64 + m.faces.len() as i64, 2);
    }

    #[test]
    fn extrude_two_adjacent_faces_as_region() {
        let mut m = cube(Vec3::ONE);
        let top = top_face(&m);
        let side = *m.faces.iter().find(|(_, f)| m.face_normal(f).x > 0.9).unwrap().0;
        extrude_faces(&mut m, &BTreeSet::from([top, side]), 0.5, None, false).unwrap();
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert_eq!(m.verts.len() as i64 - m.edges().len() as i64 + m.faces.len() as i64, 2);
    }

    #[test]
    fn inset_individual_and_region() {
        let mut m = cube(Vec3::ONE);
        let top = top_face(&m);
        inset_faces(&mut m, &BTreeSet::from([top]), 0.1, 0.0, true).unwrap();
        assert_eq!(m.faces.len(), 10);
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        let inner = &m.faces[&top];
        for p in m.face_positions(inner) {
            assert!((p.x.abs() - 0.4).abs() < 1e-5 && (p.z.abs() - 0.4).abs() < 1e-5);
        }
        let mut m = crate::primitives::plane(glam::Vec2::ONE, 2, 2);
        let all: BTreeSet<_> = m.faces.keys().copied().collect();
        inset_faces(&mut m, &all, 0.1, 0.1, false).unwrap();
        assert_eq!(m.faces.len(), 4 + 8);
        assert!(m.validate().is_empty(), "{:?}", m.validate());
    }

    #[test]
    fn extrude_boundary_edges() {
        let mut m = crate::primitives::plane(glam::Vec2::ONE, 1, 1);
        let edges = m.boundary_edges();
        extrude_edges(&mut m, &edges, Vec3::Y).unwrap();
        assert_eq!(m.faces.len(), 5);
        assert!(m.validate().is_empty(), "{:?}", m.validate());
    }
}
