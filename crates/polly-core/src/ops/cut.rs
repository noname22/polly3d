use std::collections::{BTreeSet, HashMap};

use crate::error::{Result, invalid};
use crate::ids::Selection;
use crate::mesh::{Corner, Edge, Face, FaceId, Mesh, VertId};

/// Insert `cuts` evenly spaced vertices on each edge. Returns the new vertices.
pub fn split_edges(mesh: &mut Mesh, edges: &BTreeSet<Edge>, cuts: u32) -> Result<Selection> {
    if edges.is_empty() || cuts == 0 {
        return invalid("need at least one edge and cuts >= 1");
    }
    let ef = mesh.edge_faces();
    let mut out = Selection::default();
    for e in edges {
        let Some(faces) = ef.get(e) else { return invalid(format!("edge {} does not exist", crate::ids::edge(*e))) };
        let (pa, pb) = (mesh.pos(e.0), mesh.pos(e.1));
        let pts: Vec<(VertId, f32)> = (1..=cuts)
            .map(|k| {
                let t = k as f32 / (cuts + 1) as f32;
                (mesh.add_vert(pa.lerp(pb, t)), t)
            })
            .collect();
        for f in faces {
            mesh.insert_on_edge(*f, e.0, e.1, &pts);
        }
        out.verts.extend(pts.iter().map(|p| p.0));
    }
    Ok(out)
}

/// Split a face by connecting two of its (non-adjacent) vertices.
pub fn connect_pair(mesh: &mut Mesh, a: VertId, b: VertId) -> Result<FaceId> {
    let found = mesh.faces.iter().find_map(|(&id, f)| {
        let (ia, ib) = (f.index_of(a)?, f.index_of(b)?);
        let n = f.len();
        let adjacent = (ia + 1) % n == ib || (ib + 1) % n == ia;
        (!adjacent).then_some((id, ia.min(ib), ia.max(ib)))
    });
    let Some((id, i, j)) = found else {
        return invalid(format!(
            "no face contains both {} and {} as non-adjacent vertices",
            crate::ids::vert(a),
            crate::ids::vert(b)
        ));
    };
    let f = mesh.faces[&id].clone();
    let n = f.len();
    let first: Vec<usize> = (i..=j).collect();
    let second: Vec<usize> = (j..n).chain(0..=i).collect();
    let pick = |idx: &[usize]| Face {
        verts: idx.iter().map(|k| f.verts[*k]).collect(),
        corners: idx.iter().map(|k| f.corners[*k]).collect(),
        material: f.material,
        smooth: f.smooth,
        part: f.part.clone(),
    };
    mesh.faces.insert(id, pick(&first));
    let s = pick(&second);
    mesh.add_face_from(s.verts.clone(), s.corners.clone(), &s)
}

/// Connect a path of vertices pairwise (v0-v1, v1-v2, ...), splitting faces.
pub fn connect_verts(mesh: &mut Mesh, path: &[VertId]) -> Result<Selection> {
    if path.len() < 2 {
        return invalid("need at least two vertices");
    }
    let mut out = Selection::default();
    for w in path.windows(2) {
        connect_pair(mesh, w[0], w[1])?;
        out.edges.insert(Edge::new(w[0], w[1]));
    }
    Ok(out)
}

/// Walk the ring of quads across `start`. Returns oriented ring edges and the quads.
pub fn edge_ring(mesh: &Mesh, start: Edge) -> (Vec<(VertId, VertId)>, Vec<FaceId>) {
    let ef = mesh.edge_faces();
    let mut ring = vec![(start.0, start.1)];
    let mut visited = BTreeSet::from([start]);
    let mut quads: Vec<FaceId> = Vec::new();
    for &f0 in ef.get(&start).map(|v| v.as_slice()).unwrap_or(&[]) {
        let mut cur = (start.0, start.1);
        let mut face = f0;
        loop {
            if quads.contains(&face) {
                break;
            }
            let q = &mesh.faces[&face].verts;
            if q.len() != 4 {
                break;
            }
            let Some(i) = (0..4).find(|&i| Edge::new(q[i], q[(i + 1) % 4]) == Edge::new(cur.0, cur.1)) else { break };
            quads.push(face);
            let opp = if (q[i], q[(i + 1) % 4]) == cur {
                (q[(i + 3) % 4], q[(i + 2) % 4])
            } else {
                (q[(i + 2) % 4], q[(i + 3) % 4])
            };
            let e = Edge::new(opp.0, opp.1);
            if !visited.insert(e) {
                break;
            }
            ring.push(opp);
            let Some(next) = ef[&e].iter().find(|&&g| g != face) else { break };
            face = *next;
            cur = opp;
        }
    }
    (ring, quads)
}

/// Loop cut through the quad ring crossing `edge`. With `cuts == 1`, the cut
/// is placed at `fraction` (0..1) along the ring edges. Returns the new edges.
pub fn loop_cut(mesh: &mut Mesh, edge: Edge, cuts: u32, fraction: f32) -> Result<Selection> {
    if cuts == 0 {
        return invalid("cuts must be >= 1");
    }
    if !mesh.edges().contains(&edge) {
        return invalid(format!("edge {} does not exist", crate::ids::edge(edge)));
    }
    let (ring, quads) = edge_ring(mesh, edge);
    if quads.is_empty() {
        return invalid("edge is not part of any quad; loop cut needs quads");
    }
    let fracs: Vec<f32> = if cuts == 1 {
        vec![fraction.clamp(0.01, 0.99)]
    } else {
        (1..=cuts).map(|k| k as f32 / (cuts + 1) as f32).collect()
    };
    // Edge -> (start vertex, [(new vertex, fraction from start)])
    let mut pts: HashMap<Edge, (VertId, Vec<(VertId, f32)>)> = HashMap::new();
    for &(s, t) in &ring {
        let (ps, pt) = (mesh.pos(s), mesh.pos(t));
        let list = fracs.iter().map(|&f| (mesh.add_vert(ps.lerp(pt, f)), f)).collect();
        pts.insert(Edge::new(s, t), (s, list));
    }
    // Points on edge x->y ordered from x with fractions from x.
    let along = |x: VertId, y: VertId| -> Vec<(VertId, f32)> {
        let (s, list) = &pts[&Edge::new(x, y)];
        if *s == x { list.clone() } else { list.iter().rev().map(|&(v, f)| (v, 1.0 - f)).collect() }
    };
    let quad_set: BTreeSet<FaceId> = quads.iter().copied().collect();
    // Non-ring faces sharing a ring edge get the points inserted.
    let ef = mesh.edge_faces();
    for e in pts.keys() {
        for f in &ef[e] {
            if !quad_set.contains(f) {
                let p = along(e.0, e.1);
                mesh.insert_on_edge(*f, e.0, e.1, &p);
            }
        }
    }
    let mut out = Selection::default();
    for qid in quads {
        let f = mesh.faces[&qid].clone();
        let q = &f.verts;
        let Some(i) = (0..4).find(|&i| {
            pts.contains_key(&Edge::new(q[i], q[(i + 1) % 4])) && pts.contains_key(&Edge::new(q[(i + 2) % 4], q[(i + 3) % 4]))
        }) else {
            continue;
        };
        let (ia, ib, ic, id) = (i, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4);
        let rail = |x: usize, y: usize| -> Vec<(VertId, Corner)> {
            let mut r = vec![(q[x], f.corners[x])];
            r.extend(along(q[x], q[y]).into_iter().map(|(v, t)| (v, f.corners[x].lerp(&f.corners[y], t))));
            r.push((q[y], f.corners[y]));
            r
        };
        let ab = rail(ia, ib);
        let dc = rail(id, ic);
        mesh.remove_face(qid);
        for j in 0..ab.len() - 1 {
            let loopv = [ab[j], ab[j + 1], dc[j + 1], dc[j]];
            mesh.add_face_from(loopv.iter().map(|x| x.0).collect(), loopv.iter().map(|x| x.1).collect(), &f)?;
            if j > 0 {
                out.edges.insert(Edge::new(ab[j].0, dc[j].0));
            }
        }
    }
    Ok(out)
}

/// Midpoint subdivision of the given faces (one level). Neighbouring faces get
/// the edge midpoints inserted so the mesh stays watertight.
pub fn subdivide(mesh: &mut Mesh, faces: &BTreeSet<FaceId>) -> Result<Selection> {
    if faces.is_empty() {
        return invalid("no faces given");
    }
    let mut mid: HashMap<Edge, VertId> = HashMap::new();
    for f in faces {
        for e in mesh.face(*f)?.edges().collect::<Vec<_>>() {
            if !mid.contains_key(&e) {
                let p = (mesh.pos(e.0) + mesh.pos(e.1)) / 2.0;
                mid.insert(e, mesh.add_vert(p));
            }
        }
    }
    let ef = mesh.edge_faces();
    for (e, m) in &mid {
        for f in &ef[e] {
            if !faces.contains(f) {
                mesh.insert_on_edge(*f, e.0, e.1, &[(*m, 0.5)]);
            }
        }
    }
    let mut out = Selection::default();
    for fid in faces {
        let f = mesh.remove_face(*fid).unwrap();
        let n = f.len();
        let v = |i: usize| (f.verts[i % n], f.corners[i % n]);
        let m = |i: usize| {
            let (a, b) = (f.verts[i % n], f.verts[(i + 1) % n]);
            (mid[&Edge::new(a, b)], f.corners[i % n].lerp(&f.corners[(i + 1) % n], 0.5))
        };
        let mut add = |l: Vec<(VertId, Corner)>, mesh: &mut Mesh| -> Result<()> {
            let id = mesh.add_face_from(l.iter().map(|x| x.0).collect(), l.iter().map(|x| x.1).collect(), &f)?;
            out.faces.insert(id);
            Ok(())
        };
        if n == 3 {
            for i in 0..3 {
                add(vec![v(i), m(i), m(i + 2)], mesh)?;
            }
            add(vec![m(0), m(1), m(2)], mesh)?;
        } else {
            let cp = mesh.face_center(&f);
            let c = mesh.add_vert(cp);
            let cc = f.corners.iter().fold(Corner { uv: glam::Vec2::ZERO, color: glam::Vec4::ZERO }, |a, b| Corner {
                uv: a.uv + b.uv / n as f32,
                color: a.color + b.color / n as f32,
            });
            for i in 0..n {
                add(vec![v(i), m(i), (c, cc), m(i + n - 1)], mesh)?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;
    use glam::Vec3;

    fn euler(m: &Mesh) -> i64 {
        m.verts.len() as i64 - m.edges().len() as i64 + m.faces.len() as i64
    }

    #[test]
    fn loop_cut_cube() {
        let mut m = cube(Vec3::ONE);
        let e = *m.edges().iter().find(|e| (m.pos(e.0) - m.pos(e.1)).y.abs() > 0.5).unwrap();
        let r = loop_cut(&mut m, e, 1, 0.5).unwrap();
        assert_eq!(r.edges.len(), 4);
        assert_eq!((m.verts.len(), m.faces.len()), (12, 10));
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert_eq!(euler(&m), 2);
        let mut m2 = cube(Vec3::ONE);
        loop_cut(&mut m2, e, 3, 0.5).unwrap();
        assert_eq!(m2.faces.len(), 6 + 4 * 3);
        assert!(m2.validate().is_empty());
    }

    #[test]
    fn subdivide_one_face_keeps_watertight() {
        let mut m = cube(Vec3::ONE);
        let f = *m.faces.keys().next().unwrap();
        subdivide(&mut m, &BTreeSet::from([f])).unwrap();
        assert_eq!(m.faces.len(), 9);
        assert!(m.boundary_edges().is_empty());
        assert!(m.validate().iter().all(|w| !w.contains("winding")));
        assert_eq!(euler(&m), 2);
    }

    #[test]
    fn split_and_connect() {
        let mut m = cube(Vec3::ONE);
        let f = m.faces[&0].clone();
        let e1 = Edge::new(f.verts[0], f.verts[1]);
        let e2 = Edge::new(f.verts[2], f.verts[3]);
        let s = split_edges(&mut m, &BTreeSet::from([e1, e2]), 1).unwrap();
        let vs: Vec<_> = s.verts.into_iter().collect();
        connect_verts(&mut m, &vs).unwrap();
        assert_eq!(m.faces.len(), 7);
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert_eq!(euler(&m), 2);
    }
}
