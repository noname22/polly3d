use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use glam::Vec3;

use crate::error::{Result, invalid};
use crate::ids::Selection;
use crate::mesh::{Edge, FaceId, Mesh, VertId};

// ---- delete -----------------------------------------------------------

pub fn delete(mesh: &mut Mesh, sel: &Selection, keep_loose_verts: bool) -> Result<()> {
    if sel.is_empty() {
        return invalid("nothing to delete");
    }
    for f in &sel.faces {
        mesh.remove_face(*f);
    }
    for e in &sel.edges {
        mesh.faces.retain(|_, f| f.edge_index(*e).is_none());
    }
    for v in &sel.verts {
        mesh.remove_vert(*v);
    }
    if !keep_loose_verts {
        mesh.remove_loose_verts();
    }
    mesh.prune_edge_sets();
    Ok(())
}

// ---- welding ------------------------------------------------------------

/// Replace vertex ids according to `map`, fixing up loops; drops faces that degenerate.
fn remap(mesh: &mut Mesh, map: &HashMap<VertId, VertId>) {
    let mut dead = Vec::new();
    for (&id, f) in mesh.faces.iter_mut() {
        let mut verts = Vec::with_capacity(f.len());
        let mut corners = Vec::with_capacity(f.len());
        for (v, c) in f.verts.iter().zip(&f.corners) {
            let v = *map.get(v).unwrap_or(v);
            if verts.last() != Some(&v) {
                verts.push(v);
                corners.push(*c);
            }
        }
        while verts.len() > 1 && verts.first() == verts.last() {
            verts.pop();
            corners.pop();
        }
        let unique: BTreeSet<_> = verts.iter().collect();
        if verts.len() < 3 || unique.len() != verts.len() {
            dead.push(id);
        } else {
            f.verts = verts;
            f.corners = corners;
        }
    }
    for id in dead {
        mesh.remove_face(id);
    }
    for v in map.keys() {
        if map[v] != *v {
            mesh.verts.remove(v);
        }
    }
    let remap_set = |s: &BTreeSet<Edge>| -> BTreeSet<Edge> {
        s.iter()
            .map(|e| Edge::new(*map.get(&e.0).unwrap_or(&e.0), *map.get(&e.1).unwrap_or(&e.1)))
            .filter(|e| e.0 != e.1)
            .collect()
    };
    mesh.seams = remap_set(&mesh.seams);
    mesh.sharp = remap_set(&mesh.sharp);
    mesh.prune_edge_sets();
}

/// Merge vertices closer than `distance`. Returns the number of removed vertices.
pub fn merge_by_distance(mesh: &mut Mesh, verts: &BTreeSet<VertId>, distance: f32) -> Result<usize> {
    let ids: Vec<VertId> = verts.iter().copied().collect();
    let mut parent: Vec<usize> = (0..ids.len()).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        p[i] = r;
        r
    }
    for i in 0..ids.len() {
        for j in i + 1..ids.len() {
            if mesh.pos(ids[i]).distance(mesh.pos(ids[j])) <= distance {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[b] = a;
                }
            }
        }
    }
    let mut clusters: BTreeMap<usize, Vec<VertId>> = BTreeMap::new();
    for i in 0..ids.len() {
        let r = find(&mut parent, i);
        clusters.entry(r).or_default().push(ids[i]);
    }
    let mut map = HashMap::new();
    let mut removed = 0;
    for members in clusters.values().filter(|m| m.len() > 1) {
        let keep = members[0];
        let c = members.iter().map(|v| mesh.pos(*v)).sum::<Vec3>() / members.len() as f32;
        mesh.verts.get_mut(&keep).unwrap().pos = c;
        for v in members {
            map.insert(*v, keep);
        }
        removed += members.len() - 1;
    }
    remap(mesh, &map);
    Ok(removed)
}

/// Collapse vertices into one at their centroid (or `target`).
pub fn collapse(mesh: &mut Mesh, verts: &BTreeSet<VertId>, target: Option<Vec3>) -> Result<VertId> {
    let Some(&keep) = verts.iter().next() else { return invalid("no vertices to collapse") };
    let c = target.unwrap_or_else(|| super::transform::centroid(mesh, verts));
    mesh.verts.get_mut(&keep).unwrap().pos = c;
    let map: HashMap<_, _> = verts.iter().map(|v| (*v, keep)).collect();
    remap(mesh, &map);
    Ok(keep)
}

// ---- dissolve / merge faces ---------------------------------------------

/// Merge a connected region of faces into one n-gon.
pub fn merge_faces(mesh: &mut Mesh, region: &BTreeSet<FaceId>) -> Result<FaceId> {
    if region.len() < 2 {
        return invalid("need at least two faces to merge");
    }
    let mut count: HashMap<Edge, u32> = HashMap::new();
    for f in region {
        for e in mesh.face(*f)?.edges() {
            *count.entry(e).or_default() += 1;
        }
    }
    let mut next: HashMap<VertId, VertId> = HashMap::new();
    let mut corner_of = HashMap::new();
    for f in region {
        let face = &mesh.faces[f];
        for (i, (a, b)) in face.half_edges().enumerate() {
            corner_of.entry(a).or_insert(face.corners[i]);
            if count[&Edge::new(a, b)] == 1 && next.insert(a, b).is_some() {
                return invalid("faces touch at a single vertex; cannot merge into one polygon");
            }
        }
    }
    let Some(&start) = next.keys().min() else { return invalid("region has no boundary") };
    let mut loopv = vec![start];
    let mut cur = next[&start];
    while cur != start {
        if loopv.len() > next.len() {
            return invalid("broken boundary");
        }
        loopv.push(cur);
        cur = *next.get(&cur).ok_or_else(|| crate::error::Error::Invalid("open boundary".into()))?;
    }
    if loopv.len() != next.len() {
        return invalid("region has holes or is not connected; it cannot become a single face");
    }
    let template = mesh.faces[region.iter().next().unwrap()].clone();
    let interior: BTreeSet<VertId> = region
        .iter()
        .flat_map(|f| mesh.faces[f].verts.clone())
        .filter(|v| !next.contains_key(v))
        .collect();
    for f in region {
        mesh.remove_face(*f);
    }
    let corners = loopv.iter().map(|v| corner_of[v]).collect();
    let id = mesh.add_face_with(loopv, corners, template.material, template.smooth)?;
    let used: BTreeSet<VertId> = mesh.faces.values().flat_map(|f| f.verts.iter().copied()).collect();
    for v in interior {
        if !used.contains(&v) {
            mesh.verts.remove(&v);
        }
    }
    mesh.prune_edge_sets();
    Ok(id)
}

/// Remove a vertex that has exactly two edges from all face loops.
fn dissolve_valence2(mesh: &mut Mesh, v: VertId) -> bool {
    let fs: Vec<FaceId> = mesh.faces.iter().filter(|(_, f)| f.verts.contains(&v)).map(|(i, _)| *i).collect();
    if fs.iter().any(|f| mesh.faces[f].len() <= 3) {
        return false;
    }
    for f in fs {
        let face = mesh.faces.get_mut(&f).unwrap();
        let i = face.index_of(v).unwrap();
        face.verts.remove(i);
        face.corners.remove(i);
    }
    mesh.verts.remove(&v);
    true
}

fn is_collinear_valence2(mesh: &Mesh, v: VertId) -> bool {
    let ve = mesh.vert_edges();
    let Some(es) = ve.get(&v) else { return false };
    if es.len() != 2 {
        return false;
    }
    let mut it = es.iter();
    let (a, b) = (it.next().unwrap().other(v), it.next().unwrap().other(v));
    let p = mesh.pos(v);
    (mesh.pos(a) - p).normalize_or_zero().dot((mesh.pos(b) - p).normalize_or_zero()) < -0.9995
}

/// Dissolve edges, merging the faces on either side. Leftover collinear
/// two-edge vertices are removed too.
pub fn dissolve_edges(mesh: &mut Mesh, edges: &BTreeSet<Edge>) -> Result<Selection> {
    let mut out = Selection::default();
    for e in edges {
        let ef = mesh.edge_faces();
        let Some(fs) = ef.get(e) else { continue }; // already gone
        if fs.len() != 2 {
            return invalid(format!("edge {} is not shared by exactly two faces", crate::ids::edge(*e)));
        }
        let id = merge_faces(mesh, &fs.iter().copied().collect())?;
        out.faces.insert(id);
        for v in [e.0, e.1] {
            if mesh.verts.contains_key(&v) && is_collinear_valence2(mesh, v) {
                dissolve_valence2(mesh, v);
            }
        }
    }
    out.prune(mesh);
    Ok(out)
}

pub fn dissolve_verts(mesh: &mut Mesh, verts: &BTreeSet<VertId>) -> Result<Selection> {
    let mut out = Selection::default();
    for &v in verts {
        if !mesh.verts.contains_key(&v) {
            continue;
        }
        let ve = mesh.vert_edges();
        if ve.get(&v).map_or(0, |e| e.len()) == 2 {
            if !dissolve_valence2(mesh, v) {
                return invalid(format!("cannot dissolve {}: a triangle would collapse", crate::ids::vert(v)));
            }
            continue;
        }
        let fs: BTreeSet<FaceId> = mesh.faces.iter().filter(|(_, f)| f.verts.contains(&v)).map(|(i, _)| *i).collect();
        let id = merge_faces(mesh, &fs)?;
        out.faces.insert(id);
    }
    out.prune(mesh);
    Ok(out)
}

// ---- normals ---------------------------------------------------------------

pub fn flip(mesh: &mut Mesh, faces: &BTreeSet<FaceId>) -> Result<()> {
    for f in faces {
        mesh.face_mut(*f)?.reverse();
    }
    Ok(())
}

/// Make winding consistent within each connected component of `faces`, then
/// point each component outward.
pub fn recalc_normals(mesh: &mut Mesh, faces: &BTreeSet<FaceId>, inside: bool) -> Result<()> {
    let ef = mesh.edge_faces();
    let mut seen = BTreeSet::new();
    for &seed in faces {
        if !seen.insert(seed) {
            continue;
        }
        let mut comp = vec![seed];
        let mut q = VecDeque::from([seed]);
        while let Some(f) = q.pop_front() {
            let face = mesh.faces[&f].clone();
            for (a, b) in face.half_edges() {
                for &g in &ef[&Edge::new(a, b)] {
                    if g == f || !faces.contains(&g) || !seen.insert(g) {
                        continue;
                    }
                    if mesh.faces[&g].has_half_edge(a, b) {
                        mesh.faces.get_mut(&g).unwrap().reverse();
                    }
                    comp.push(g);
                    q.push_back(g);
                }
            }
        }
        let pts: Vec<Vec3> = comp.iter().flat_map(|f| mesh.face_positions(&mesh.faces[f])).collect();
        let centroid = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
        let vol: f32 = comp
            .iter()
            .map(|f| {
                let p = mesh.face_positions(&mesh.faces[f]);
                let c = p.iter().copied().sum::<Vec3>() / p.len() as f32;
                crate::mesh::newell_vector(&p).dot(c - centroid)
            })
            .sum();
        if (vol < 0.0) != inside {
            for f in &comp {
                mesh.faces.get_mut(f).unwrap().reverse();
            }
        }
    }
    Ok(())
}

// ---- triangulate / fill / bridge -------------------------------------------

pub fn triangulate(mesh: &mut Mesh, faces: &BTreeSet<FaceId>) -> Result<Selection> {
    let mut out = Selection::default();
    for fid in faces {
        let f = mesh.face(*fid)?.clone();
        if f.len() == 3 {
            out.faces.insert(*fid);
            continue;
        }
        let tris = crate::geom::triangulate_polygon(&mesh.face_positions(&f));
        mesh.remove_face(*fid);
        for [a, b, c] in tris {
            let id = mesh.add_face_with(
                vec![f.verts[a], f.verts[b], f.verts[c]],
                vec![f.corners[a], f.corners[b], f.corners[c]],
                f.material,
                f.smooth,
            )?;
            out.faces.insert(id);
        }
    }
    Ok(out)
}

/// Fill boundary holes with n-gons. If `edges` is non-empty only holes
/// containing one of them are filled.
pub fn fill_holes(mesh: &mut Mesh, edges: &BTreeSet<Edge>) -> Result<Selection> {
    let boundary = mesh.boundary_edges();
    // Hole loops run opposite to the faces around them.
    let mut next: HashMap<VertId, VertId> = HashMap::new();
    let mut template = HashMap::new();
    for (&fid, f) in &mesh.faces {
        for (a, b) in f.half_edges() {
            if boundary.contains(&Edge::new(a, b)) {
                next.insert(b, a);
                template.insert(b, fid);
            }
        }
    }
    let mut out = Selection::default();
    let mut done = BTreeSet::new();
    let starts: Vec<VertId> = next.keys().copied().collect();
    for s in starts {
        if done.contains(&s) {
            continue;
        }
        let mut lp = vec![s];
        let mut cur = next[&s];
        let mut ok = true;
        while cur != s {
            if lp.len() > next.len() || done.contains(&cur) {
                ok = false;
                break;
            }
            lp.push(cur);
            match next.get(&cur) {
                Some(n) => cur = *n,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        done.extend(lp.iter().copied());
        if !ok || lp.len() < 3 {
            continue;
        }
        let n = lp.len();
        let wanted = edges.is_empty() || (0..n).any(|i| edges.contains(&Edge::new(lp[i], lp[(i + 1) % n])));
        if wanted {
            let t = mesh.faces[&template[&s]].clone();
            out.faces.insert(mesh.add_face_like(lp, &t)?);
        }
    }
    if out.faces.is_empty() {
        return invalid("no matching boundary holes found");
    }
    Ok(out)
}

/// Remove two faces with equal vertex counts and connect their loops with quads.
pub fn bridge_faces(mesh: &mut Mesh, f1: FaceId, f2: FaceId) -> Result<Selection> {
    let a = mesh.face(f1)?.clone();
    let b = mesh.face(f2)?.clone();
    if a.len() != b.len() {
        return invalid(format!("faces have different vertex counts ({} vs {})", a.len(), b.len()));
    }
    if a.verts.iter().any(|v| b.verts.contains(v)) {
        return invalid("faces share vertices; cannot bridge");
    }
    let n = a.len();
    let pa: Vec<Vec3> = a.verts.iter().map(|v| mesh.pos(*v)).collect();
    let pb: Vec<Vec3> = b.verts.iter().map(|v| mesh.pos(*v)).collect();
    // a_i pairs with b_{(offset - i) mod n}; pick the offset with least twist.
    let (ca, cb) = (pa.iter().sum::<Vec3>() / n as f32, pb.iter().sum::<Vec3>() / n as f32);
    let m = |off: usize, i: usize| (off + n - i % n) % n;
    let best = (0..n)
        .min_by(|&x, &y| {
            let cost = |off: usize| (0..n).map(|i| ((pa[i] - ca) - (pb[m(off, i)] - cb)).length()).sum::<f32>();
            cost(x).total_cmp(&cost(y))
        })
        .unwrap();
    mesh.remove_face(f1);
    mesh.remove_face(f2);
    let mut out = Selection::default();
    for i in 0..n {
        let j = (i + 1) % n;
        let quad = vec![a.verts[i], a.verts[j], b.verts[m(best, j)], b.verts[m(best, i)]];
        out.faces.insert(mesh.add_face_like(quad, &a)?);
    }
    Ok(out)
}

// ---- mirror / duplicate ---------------------------------------------------

/// Mirror the mesh across the plane `axis = 0` (0 = X, 1 = Y, 2 = Z). With
/// `merge > 0`, vertices on the plane are welded and faces lying on the
/// plane are removed.
pub fn mirror(mesh: &mut Mesh, axis: usize, merge: f32) -> Result<Selection> {
    if axis > 2 {
        return invalid("axis must be x, y or z");
    }
    let src = mesh.clone();
    let mut on_plane = BTreeSet::new();
    if merge > 0.0 {
        for (&id, v) in mesh.verts.iter_mut() {
            if v.pos[axis].abs() <= merge {
                v.pos[axis] = 0.0;
                on_plane.insert(id);
            }
        }
    }
    let mut vmap = HashMap::new();
    for (&id, v) in &src.verts {
        if on_plane.contains(&id) {
            vmap.insert(id, id);
        } else {
            let mut p = v.pos;
            p[axis] = -p[axis];
            vmap.insert(id, mesh.add_vert(p));
        }
    }
    let mut out = Selection::default();
    let mut dead = Vec::new();
    for (&fid, f) in &src.faces {
        if f.verts.iter().all(|v| on_plane.contains(v)) {
            dead.push(fid); // interior after mirroring
            continue;
        }
        let mut verts: Vec<VertId> = f.verts.iter().map(|v| vmap[v]).collect();
        let mut corners = f.corners.clone();
        verts.reverse();
        corners.reverse();
        out.faces.insert(mesh.add_face_with(verts, corners, f.material, f.smooth)?);
    }
    for f in dead {
        mesh.remove_face(f);
    }
    for e in &src.seams {
        mesh.seams.insert(Edge::new(vmap[&e.0], vmap[&e.1]));
    }
    for e in &src.sharp {
        mesh.sharp.insert(Edge::new(vmap[&e.0], vmap[&e.1]));
    }
    mesh.remove_loose_verts();
    Ok(out)
}

/// Copy faces (with their own new vertices), offset by `offset`.
pub fn duplicate(mesh: &mut Mesh, faces: &BTreeSet<FaceId>, offset: Vec3) -> Result<Selection> {
    let mut vmap = HashMap::new();
    let mut out = Selection::default();
    for fid in faces {
        let f = mesh.face(*fid)?.clone();
        let verts = f
            .verts
            .iter()
            .map(|v| {
                *vmap.entry(*v).or_insert_with(|| {
                    let p = mesh.pos(*v) + offset;
                    mesh.add_vert(p)
                })
            })
            .collect();
        out.faces.insert(mesh.add_face_with(verts, f.corners, f.material, f.smooth)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{cube, plane};
    use glam::Vec2;

    fn euler(m: &Mesh) -> i64 {
        m.verts.len() as i64 - m.edges().len() as i64 + m.faces.len() as i64
    }

    #[test]
    fn mirror_half_cube() {
        let mut m = cube(Vec3::ONE);
        let all: BTreeSet<_> = m.verts.keys().copied().collect();
        super::super::transform::translate(&mut m, &all, Vec3::new(0.5, 0.0, 0.0)).unwrap();
        mirror(&mut m, 0, 1e-4).unwrap();
        assert_eq!((m.verts.len(), m.faces.len()), (12, 10));
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert_eq!(euler(&m), 2);
    }

    #[test]
    fn dissolve_after_loop_cut() {
        let mut m = cube(Vec3::ONE);
        let e = *m.edges().iter().find(|e| (m.pos(e.0) - m.pos(e.1)).y.abs() > 0.5).unwrap();
        let cut = super::super::cut::loop_cut(&mut m, e, 1, 0.5).unwrap();
        dissolve_edges(&mut m, &cut.edges).unwrap();
        assert_eq!((m.verts.len(), m.faces.len()), (8, 6));
        assert!(m.validate().is_empty(), "{:?}", m.validate());
    }

    #[test]
    fn triangulate_fill_bridge_merge() {
        let mut m = cube(Vec3::ONE);
        let all: BTreeSet<_> = m.faces.keys().copied().collect();
        triangulate(&mut m, &all).unwrap();
        assert_eq!(m.faces.len(), 12);
        assert!(m.validate().is_empty());

        let mut m = cube(Vec3::ONE);
        m.remove_face(0);
        fill_holes(&mut m, &BTreeSet::new()).unwrap();
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert!(m.boundary_edges().is_empty());

        let mut m = cube(Vec3::ONE);
        let f = |n: Vec3| *m.faces.iter().find(|(_, f)| m.face_normal(f).dot(n) > 0.9).unwrap().0;
        let (top, bottom) = (f(Vec3::Y), f(-Vec3::Y));
        super::super::extrude::inset_faces(&mut m, &BTreeSet::from([top, bottom]), 0.2, 0.0, true).unwrap();
        bridge_faces(&mut m, top, bottom).unwrap();
        assert_eq!(m.faces.len(), 4 + 8 + 4);
        assert!(m.validate().is_empty(), "{:?}", m.validate());
        assert_eq!(euler(&m), 0); // torus-like

        let mut m = plane(Vec2::ONE, 2, 1);
        let fs: BTreeSet<_> = m.faces.keys().copied().collect();
        let dup = duplicate(&mut m, &fs, Vec3::ZERO).unwrap();
        assert_eq!(dup.faces.len(), 2);
        let all: BTreeSet<_> = m.verts.keys().copied().collect();
        assert_eq!(merge_by_distance(&mut m, &all, 1e-4).unwrap(), 6);
    }

    #[test]
    fn recalc_fixes_flipped_face() {
        let mut m = cube(Vec3::ONE);
        m.faces.get_mut(&2).unwrap().reverse();
        assert!(!m.validate().is_empty());
        let fs: BTreeSet<_> = m.faces.keys().copied().collect();
        recalc_normals(&mut m, &fs, false).unwrap();
        assert!(m.validate().is_empty());
        for f in m.faces.values() {
            assert!(m.face_normal(f).dot(m.face_center(f)) > 0.0);
        }
    }
}
