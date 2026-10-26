//! Building and joining tubes: bridge two vertex loops (or open chains) of
//! any sizes, and loft a tube through cross-sections.

use std::collections::{BTreeSet, HashMap};

use glam::Vec3;

use crate::error::{Result, invalid};
use crate::mesh::{Edge, FaceId, Mesh, VertId};

/// Normalised arc-length parameter of each point along a loop or chain.
fn params(p: &[Vec3], closed: bool) -> Vec<f32> {
    let n = p.len();
    let segs = if closed { n } else { n - 1 };
    let mut acc = vec![0.0];
    for i in 0..segs {
        acc.push(acc[i] + p[i].distance(p[(i + 1) % n]));
    }
    let total = acc.last().copied().unwrap_or(0.0).max(1e-12);
    acc.iter().map(|x| x / total).collect()
}

/// Rotate / reverse closed loop `b` so it starts near `a[0]` and runs the same way.
fn align(mesh: &Mesh, a: &[VertId], b: &[VertId]) -> Vec<VertId> {
    let pos = |v: &VertId| mesh.verts[v].pos;
    let start = (0..b.len()).min_by(|i, j| pos(&b[*i]).distance(pos(&a[0])).total_cmp(&pos(&b[*j]).distance(pos(&a[0])))).unwrap();
    let forward: Vec<VertId> = (0..b.len()).map(|k| b[(start + k) % b.len()]).collect();
    let backward: Vec<VertId> = (0..b.len()).map(|k| b[(start + b.len() - k) % b.len()]).collect();
    // Same direction: a quarter of the way round, the loops are close.
    let probe = |l: &[VertId]| {
        let (pa, pb) = (params(&a.iter().map(pos).collect::<Vec<_>>(), true), params(&l.iter().map(pos).collect::<Vec<_>>(), true));
        let at = |l: &[VertId], p: &[f32], t: f32| l[p.iter().position(|x| *x >= t).unwrap_or(0).min(l.len() - 1)];
        pos(&at(a, &pa, 0.25)).distance(pos(&at(l, &pb, 0.25)))
    };
    if probe(&backward) < probe(&forward) { backward } else { forward }
}

/// Connect loop (or chain) `a` to `b` with a band of faces: quads when they
/// have the same number of vertices, otherwise triangles spread evenly by
/// arc length. Loops are aligned automatically (nearest start, same
/// direction); chains are joined end to end as given. New faces are wound to
/// match existing faces along `a` or `b`.
pub fn bridge_loops(mesh: &mut Mesh, a: &[VertId], b: &[VertId], closed: bool) -> Result<BTreeSet<FaceId>> {
    let min = if closed { 3 } else { 2 };
    if a.len() < min || b.len() < min {
        return invalid(format!("each {} needs at least {min} vertices", if closed { "loop" } else { "chain" }));
    }
    for v in a.iter().chain(b) {
        if !mesh.verts.contains_key(v) {
            return invalid(format!("no vertex v{v}"));
        }
    }
    if a.iter().any(|v| b.contains(v)) {
        return invalid("the two loops share vertices");
    }
    let b: Vec<VertId> = if closed { align(mesh, a, b) } else { b.to_vec() };
    let pos = |v: &VertId| mesh.verts[v].pos;
    let mut faces: Vec<Vec<VertId>> = Vec::new();
    let (na, nb) = (a.len(), b.len());
    let wrap = |l: &[VertId], i: usize| l[i % l.len()];
    if na == nb {
        let segs = if closed { na } else { na - 1 };
        for i in 0..segs {
            faces.push(vec![wrap(a, i), wrap(a, i + 1), wrap(&b, i + 1), wrap(&b, i)]);
        }
    } else {
        // Zipper: always advance along the loop whose next point comes first.
        let pa = params(&a.iter().map(pos).collect::<Vec<_>>(), closed);
        let pb = params(&b.iter().map(pos).collect::<Vec<_>>(), closed);
        let (ea, eb) = if closed { (na, nb) } else { (na - 1, nb - 1) };
        let (mut i, mut j) = (0, 0);
        while i < ea || j < eb {
            let next_a = if i < ea { pa[i + 1] } else { f32::MAX };
            let next_b = if j < eb { pb[j + 1] } else { f32::MAX };
            if next_a <= next_b {
                faces.push(vec![wrap(a, i), wrap(a, i + 1), wrap(&b, j)]);
                i += 1;
            } else {
                faces.push(vec![wrap(a, i), wrap(&b, j + 1), wrap(&b, j)]);
                j += 1;
            }
        }
    }
    // Winding: a new face must run against existing faces along the loops.
    let existing: BTreeSet<(VertId, VertId)> = mesh.faces.values().flat_map(|f| f.half_edges()).collect();
    let mut votes = 0i32;
    for f in &faces {
        for k in 0..f.len() {
            let (x, y) = (f[k], f[(k + 1) % f.len()]);
            if existing.contains(&(x, y)) {
                votes -= 1;
            } else if existing.contains(&(y, x)) {
                votes += 1;
            }
        }
    }
    let flip = votes < 0;
    let mut out = BTreeSet::new();
    for mut f in faces {
        if flip {
            f.reverse();
        }
        out.insert(mesh.add_face(f)?);
    }
    Ok(out)
}

/// The open-boundary loop through a boundary edge, in order.
pub fn boundary_loop(mesh: &Mesh, start: Edge) -> Result<Vec<VertId>> {
    let ef = mesh.edge_faces();
    if ef.get(&start).is_none_or(|f| f.len() != 1) {
        return invalid(format!("e{}-{} is not on an open border", start.0, start.1));
    }
    let mut next: HashMap<VertId, Vec<VertId>> = HashMap::new();
    for (e, f) in &ef {
        if f.len() == 1 {
            next.entry(e.0).or_default().push(e.1);
            next.entry(e.1).or_default().push(e.0);
        }
    }
    let mut out = vec![start.0, start.1];
    loop {
        let (prev, cur) = (out[out.len() - 2], out[out.len() - 1]);
        let Some(&n) = next[&cur].iter().find(|n| **n != prev) else { return invalid("the border is not a simple loop") };
        if n == out[0] {
            return Ok(out);
        }
        if out.contains(&n) || out.len() > mesh.verts.len() {
            return invalid("the border is not a simple loop");
        }
        out.push(n);
    }
}

/// A tube through closed cross-sections (each a loop of points, any number
/// of points per section), optionally capped at both ends, facing outward.
pub fn loft(sections: &[Vec<Vec3>], caps: bool) -> Result<Mesh> {
    if sections.len() < 2 {
        return invalid("a loft needs at least two sections");
    }
    let mut m = Mesh::new();
    let rings: Vec<Vec<VertId>> = sections.iter().map(|s| s.iter().map(|p| m.add_vert(*p)).collect()).collect();
    for w in rings.windows(2) {
        bridge_loops(&mut m, &w[0], &w[1], true)?;
    }
    if caps {
        m.add_face(rings[0].clone())?;
        m.add_face(rings.last().unwrap().clone())?;
    }
    let all: BTreeSet<FaceId> = m.faces.keys().copied().collect();
    crate::ops::topology::recalc_normals(&mut m, &all, false)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize, r: f32, y: f32, phase: f32) -> Vec<Vec3> {
        (0..n).map(|i| {
            let t = phase + i as f32 / n as f32 * std::f32::consts::TAU;
            Vec3::new(r * t.cos(), y, r * t.sin())
        }).collect()
    }

    #[test]
    fn loft_with_different_section_sizes_is_closed_and_outward() {
        let m = loft(&[ring(8, 1.0, 0.0, 0.0), ring(5, 0.8, 1.0, 0.3), ring(8, 0.5, 2.0, 0.1)], true).unwrap();
        assert!(m.validate().iter().all(|w| w.contains("non-planar")), "{:?}", m.validate());
        // Closed: every edge has two faces. Outward: positive volume.
        assert!(m.edge_faces().values().all(|f| f.len() == 2));
        let vol: f32 = m.faces.values().map(|f| {
            let p = m.face_positions(f);
            let c = p.iter().copied().sum::<Vec3>() / p.len() as f32;
            crate::mesh::newell_vector(&p).dot(c - Vec3::new(0.0, 1.0, 0.0))
        }).sum();
        assert!(vol > 0.0);
        // 8 -> 5 (13 triangles), 5 -> 8 (13), 2 caps.
        assert_eq!(m.faces.len(), 13 + 13 + 2);
    }

    #[test]
    fn bridging_open_borders_and_chains() {
        // Two separate open tubes facing each other; bridge their borders.
        let mut m = loft(&[ring(6, 1.0, 0.0, 0.0), ring(6, 1.0, 1.0, 0.0)], false).unwrap();
        let low = loft(&[ring(9, 1.0, 2.0, 0.2), ring(9, 1.0, 3.0, 0.2)], false).unwrap();
        let mut ids = HashMap::new();
        for (id, v) in &low.verts {
            ids.insert(*id, m.add_vert(v.pos));
        }
        for f in low.faces.values() {
            m.add_face(f.verts.iter().map(|v| ids[v]).collect()).unwrap();
        }
        let ef = m.edge_faces();
        let border_at = |y: f32| {
            let e = *ef.iter().find(|(e, f)| f.len() == 1 && (m.verts[&e.0].pos.y - y).abs() < 1e-4 && (m.verts[&e.1].pos.y - y).abs() < 1e-4).unwrap().0;
            boundary_loop(&m, e).unwrap()
        };
        let (a, b) = (border_at(1.0), border_at(2.0));
        assert_eq!((a.len(), b.len()), (6, 9));
        let new = bridge_loops(&mut m, &a, &b, true).unwrap();
        assert_eq!(new.len(), 15);
        // Consistent winding with the tubes: no edge is traversed twice the same way.
        assert!(!m.validate().iter().any(|w| w.contains("inconsistent")), "{:?}", m.validate());
        // Open chains: a strip.
        let mut s = Mesh::new();
        let c1: Vec<VertId> = (0..4).map(|i| s.add_vert(Vec3::new(i as f32, 0.0, 0.0))).collect();
        let c2: Vec<VertId> = (0..3).map(|i| s.add_vert(Vec3::new(i as f32 * 1.5, 1.0, 0.0))).collect();
        assert_eq!(bridge_loops(&mut s, &c1, &c2, false).unwrap().len(), 5);
    }
}
