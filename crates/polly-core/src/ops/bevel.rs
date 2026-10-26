//! Edge bevel / chamfer.
//!
//! Each corner of a face adjacent to a bevelled edge is replaced by a point
//! offset `width` away from the bevelled edge(s) inside that face. Corners
//! next to one bevelled edge slide along their other (unbevelled) edge; that
//! slide point is shared with the face across the unbevelled edge. Strips
//! connect the offset points along each bevelled edge, and any remaining
//! corner holes are capped with n-gons.

use std::collections::{BTreeSet, HashMap};

use glam::Vec3;

use crate::error::{Result, invalid};
use crate::ids::Selection;
use crate::mesh::{Corner, Edge, FaceId, Mesh, VertId};

pub fn bevel_edges(mesh: &mut Mesh, edges: &BTreeSet<Edge>, width: f32, segments: u32) -> Result<Selection> {
    if edges.is_empty() {
        return invalid("no edges given");
    }
    if width <= 0.0 {
        return invalid("width must be > 0");
    }
    let segments = segments.clamp(1, 16);
    let ef = mesh.edge_faces();
    for e in edges {
        match ef.get(e) {
            Some(f) if f.len() == 2 => {}
            Some(_) => return invalid(format!("edge {} must be shared by exactly two faces", crate::ids::edge(*e))),
            None => return invalid(format!("edge {} does not exist", crate::ids::edge(*e))),
        }
    }
    let touched: BTreeSet<VertId> = edges.iter().flat_map(|e| [e.0, e.1]).collect();
    let face_ids: Vec<FaceId> =
        mesh.faces.iter().filter(|(_, f)| f.verts.iter().any(|v| touched.contains(v))).map(|(i, _)| *i).collect();

    enum Kind {
        Miter(Vec3),
        Slide(Edge), // slide along this unbevelled edge
        None,
    }
    let mut kinds: HashMap<(FaceId, VertId), Kind> = HashMap::new();
    let mut slide_dist: HashMap<(Edge, VertId), Vec<f32>> = HashMap::new();
    for &fid in &face_ids {
        let f = &mesh.faces[&fid];
        let nf = mesh.face_normal(f);
        let n = f.len();
        for i in 0..n {
            let v = f.verts[i];
            if !touched.contains(&v) {
                continue;
            }
            let (p, nx) = (f.verts[(i + n - 1) % n], f.verts[(i + 1) % n]);
            let (pv, pp, pn) = (mesh.pos(v), mesh.pos(p), mesh.pos(nx));
            let (e_in, e_out) = (Edge::new(p, v), Edge::new(v, nx));
            let kind = match (edges.contains(&e_in), edges.contains(&e_out)) {
                (true, true) => {
                    let d1 = nf.cross(pv - pp).normalize_or_zero();
                    let d2 = nf.cross(pn - pv).normalize_or_zero();
                    let s = d1 + d2;
                    let off = if s.length() < 1e-4 {
                        d1 * width
                    } else {
                        let m = s.normalize();
                        m * width / m.dot(d1).max(0.2)
                    };
                    Kind::Miter(pv + off)
                }
                (false, true) | (true, false) => {
                    let bevelled_dir = if edges.contains(&e_out) { pn - pv } else { pp - pv };
                    let (slide_edge, other) = if edges.contains(&e_out) { (e_in, pp) } else { (e_out, pn) };
                    let along = (other - pv).normalize_or_zero();
                    let sin = along.cross(bevelled_dir.normalize_or_zero()).length().max(0.05);
                    let d = (width / sin).min((other - pv).length() * 0.95);
                    slide_dist.entry((slide_edge, v)).or_default().push(d);
                    Kind::Slide(slide_edge)
                }
                (false, false) => Kind::None,
            };
            kinds.insert((fid, v), kind);
        }
    }
    let mut new_verts: BTreeSet<VertId> = BTreeSet::new();
    let mut slide_vert: HashMap<(Edge, VertId), VertId> = HashMap::new();
    for ((e, v), ds) in &slide_dist {
        let d = ds.iter().sum::<f32>() / ds.len() as f32;
        let pv = mesh.pos(*v);
        let dir = (mesh.pos(e.other(*v)) - pv).normalize_or_zero();
        let id = mesh.add_vert(pv + dir * d);
        new_verts.insert(id);
        slide_vert.insert((*e, *v), id);
    }
    // Corner replacement point per (face, vertex) for faces touching a bevelled edge.
    let mut point: HashMap<(FaceId, VertId), VertId> = HashMap::new();
    for (&(fid, v), k) in &kinds {
        match k {
            Kind::Miter(p) => {
                let id = mesh.add_vert(*p);
                new_verts.insert(id);
                point.insert((fid, v), id);
            }
            Kind::Slide(e) => {
                point.insert((fid, v), slide_vert[&(*e, v)]);
            }
            Kind::None => {}
        }
    }
    // For each bevelled edge: (edge, f, g, a, b) where f traverses a -> b.
    let strips: Vec<(Edge, FaceId, FaceId, VertId, VertId)> = edges
        .iter()
        .map(|e| {
            let fs = &ef[e];
            let (f, g) = if mesh.faces[&fs[0]].has_half_edge(e.0, e.1) { (fs[0], fs[1]) } else { (fs[1], fs[0]) };
            (*e, f, g, e.0, e.1)
        })
        .collect();
    // Rebuild face loops.
    for &fid in &face_ids {
        let f = mesh.faces[&fid].clone();
        let n = f.len();
        let mut verts = Vec::new();
        let mut corners = Vec::new();
        for i in 0..n {
            let v = f.verts[i];
            let c = f.corners[i];
            if let Some(pt) = point.get(&(fid, v)) {
                verts.push(*pt);
                corners.push(c);
                continue;
            }
            if !touched.contains(&v) {
                verts.push(v);
                corners.push(c);
                continue;
            }
            let (p, nx) = (f.verts[(i + n - 1) % n], f.verts[(i + 1) % n]);
            let s_in = slide_vert.get(&(Edge::new(p, v), v));
            let s_out = slide_vert.get(&(Edge::new(v, nx), v));
            if let Some(s) = s_in {
                verts.push(*s);
                corners.push(c);
            }
            if s_in.is_none() || s_out.is_none() {
                verts.push(v);
                corners.push(c);
            }
            if let Some(s) = s_out {
                verts.push(*s);
                corners.push(c);
            }
        }
        let face = mesh.faces.get_mut(&fid).unwrap();
        face.verts = verts;
        face.corners = corners;
    }
    // Strips.
    let mut out = Selection::default();
    for &(_, f, g, a, b) in &strips {
        let rail = |mesh: &mut Mesh, v: VertId, new_verts: &mut BTreeSet<VertId>| -> Vec<VertId> {
            let (p0, p1) = (point[&(f, v)], point[&(g, v)]);
            let mut r = vec![p0];
            if segments > 1 {
                let other = if v == a { b } else { a };
                let dir = (mesh.pos(other) - mesh.pos(v)).normalize_or_zero();
                let mid = (mesh.pos(p0) + mesh.pos(p1)) / 2.0;
                let c = mesh.pos(v) + dir * (mid - mesh.pos(v)).dot(dir);
                let (q0, q1) = (mesh.pos(p0), mesh.pos(p1));
                for j in 1..segments {
                    let t = j as f32 / segments as f32;
                    let pos = q0 * (1.0 - t) * (1.0 - t) + c * 2.0 * t * (1.0 - t) + q1 * t * t;
                    let id = mesh.add_vert(pos);
                    new_verts.insert(id);
                    r.push(id);
                }
            }
            r.push(p1);
            r
        };
        let ra = rail(mesh, a, &mut new_verts);
        let rb = rail(mesh, b, &mut new_verts);
        // Put arc points into any face that spans the rail ends directly.
        if segments > 1 {
            for r in [&ra, &rb] {
                let (s, t) = (r[0], *r.last().unwrap());
                let inner: Vec<(VertId, f32)> =
                    r[1..r.len() - 1].iter().enumerate().map(|(k, v)| (*v, (k + 1) as f32 / segments as f32)).collect();
                let host = mesh.faces.iter().find(|(_, fc)| fc.edge_index(Edge::new(s, t)).is_some()).map(|(i, _)| *i);
                if let Some(h) = host {
                    mesh.insert_on_edge(h, s, t, &inner);
                }
            }
        }
        let template = mesh.faces[&f].clone();
        let c = Corner { uv: glam::Vec2::ZERO, color: template.corners[0].color };
        for j in 0..segments as usize {
            let quad = vec![rb[j], ra[j], ra[j + 1], rb[j + 1]];
            let id = mesh.add_face_from(quad, vec![c; 4], &template)?;
            out.faces.insert(id);
        }
    }
    // Cap corner holes made only of new / touched vertices.
    cap_holes(mesh, &new_verts, &touched);
    mesh.remove_loose_verts();
    out.prune(mesh);
    Ok(out)
}

fn cap_holes(mesh: &mut Mesh, new_verts: &BTreeSet<VertId>, touched: &BTreeSet<VertId>) {
    let boundary = mesh.boundary_edges();
    let mut next: HashMap<VertId, VertId> = HashMap::new();
    let mut template: HashMap<VertId, FaceId> = HashMap::new();
    let ok = |v: &VertId| new_verts.contains(v) || touched.contains(v);
    for (&fid, f) in &mesh.faces {
        for (a, b) in f.half_edges() {
            if boundary.contains(&Edge::new(a, b)) && ok(&a) && ok(&b) {
                next.insert(b, a);
                template.insert(b, fid);
            }
        }
    }
    let mut done = BTreeSet::new();
    let starts: Vec<VertId> = next.keys().copied().collect();
    for s in starts {
        if done.contains(&s) {
            continue;
        }
        let mut lp = vec![s];
        let mut cur = match next.get(&s) {
            Some(c) => *c,
            None => continue,
        };
        let mut closed = true;
        while cur != s {
            if lp.len() > next.len() || done.contains(&cur) {
                closed = false;
                break;
            }
            lp.push(cur);
            match next.get(&cur) {
                Some(n) => cur = *n,
                None => {
                    closed = false;
                    break;
                }
            }
        }
        done.extend(lp.iter().copied());
        if closed && lp.len() >= 3 && lp.iter().any(|v| new_verts.contains(v)) {
            let t = mesh.faces[&template[&s]].clone();
            let _ = mesh.add_face_like(lp, &t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;

    fn euler(m: &Mesh) -> i64 {
        m.verts.len() as i64 - m.edges().len() as i64 + m.faces.len() as i64
    }

    #[test]
    fn bevel_single_cube_edge() {
        let mut m = cube(Vec3::ONE);
        let e = *m.edges().iter().next().unwrap();
        let r = bevel_edges(&mut m, &BTreeSet::from([e]), 0.1, 1).unwrap();
        assert_eq!(r.faces.len(), 1);
        assert_eq!((m.verts.len(), m.faces.len()), (10, 7));
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert_eq!(euler(&m), 2);
    }

    #[test]
    fn bevel_all_cube_edges() {
        let mut m = cube(Vec3::ONE);
        let all = m.edges();
        bevel_edges(&mut m, &all, 0.1, 1).unwrap();
        // 6 shrunk faces + 12 strips + 8 corner triangles
        assert_eq!(m.faces.len(), 26);
        assert_eq!(m.verts.len(), 24);
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert_eq!(euler(&m), 2);
    }

    #[test]
    fn rounded_bevel() {
        let mut m = cube(Vec3::ONE);
        let e = *m.edges().iter().next().unwrap();
        bevel_edges(&mut m, &BTreeSet::from([e]), 0.2, 3).unwrap();
        assert_eq!((m.verts.len(), m.faces.len()), (14, 9));
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert_eq!(euler(&m), 2);
        let all = cube(Vec3::ONE).edges();
        let mut m = cube(Vec3::ONE);
        bevel_edges(&mut m, &all, 0.1, 2).unwrap();
        assert!(m.boundary_edges().is_empty());
        assert_eq!(euler(&m), 2);
    }
}
