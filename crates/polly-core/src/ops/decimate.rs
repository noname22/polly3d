//! Mesh decimation by quadric error edge collapse (Garland–Heckbert), for
//! LODs and lighter game meshes. Faces are triangulated first. Vertices on
//! open borders, UV seams (anywhere UVs or colours differ between faces),
//! sharp edges and material or part borders are locked, so silhouettes,
//! texture layouts and hard edges survive. Skin weights and morph offsets of
//! a collapsed vertex blend like its position.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};

use glam::{DVec3, Vec3};

use crate::mesh::{Corner, Edge, Face, FaceId, Mesh, VertId};

#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: DVec3, d: f64, w: f64) -> Quadric {
        let (a, b, c) = (n.x, n.y, n.z);
        Quadric([a * a * w, a * b * w, a * c * w, a * d * w, b * b * w, b * c * w, b * d * w, c * c * w, c * d * w, d * d * w])
    }
    fn add(&mut self, o: &Quadric) {
        for i in 0..10 {
            self.0[i] += o.0[i];
        }
    }
    fn eval(&self, p: DVec3) -> f64 {
        let q = &self.0;
        let (x, y, z) = (p.x, p.y, p.z);
        q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y + q[7] * z * z + 2.0 * q[8] * z + q[9]
    }
}

#[derive(PartialEq, PartialOrd)]
struct Cost(f64);
impl Eq for Cost {}
impl Ord for Cost {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

fn tri_normal(m: &Mesh, f: &Face) -> Vec3 {
    let p: Vec<Vec3> = f.verts.iter().map(|v| m.verts[v].pos).collect();
    (p[1] - p[0]).cross(p[2] - p[0])
}

/// Split every n-gon into triangles (corners, material, smoothing and part kept).
pub fn triangulated(m: &Mesh) -> Mesh {
    let mut out = m.clone();
    let ids: Vec<FaceId> = out.faces.keys().copied().collect();
    for fid in ids {
        let f = out.faces[&fid].clone();
        if f.len() == 3 {
            continue;
        }
        let pts = out.face_positions(&f);
        out.faces.remove(&fid);
        for [a, b, c] in crate::geom::triangulate_polygon(&pts) {
            let _ = out.add_face_from(vec![f.verts[a], f.verts[b], f.verts[c]], vec![f.corners[a], f.corners[b], f.corners[c]], &f);
        }
    }
    out
}

/// Collapse edges until at most `target_tris` triangles remain (or nothing
/// more can be collapsed without breaking locked features).
pub fn decimate(input: &Mesh, target_tris: usize) -> Mesh {
    let mut m = triangulated(input);
    let mut vf: HashMap<VertId, Vec<FaceId>> = m.vert_faces();
    // Locked vertices.
    let mut locked: HashSet<VertId> = HashSet::new();
    let mut edge_faces: HashMap<Edge, Vec<FaceId>> = HashMap::new();
    for (fid, f) in &m.faces {
        for e in f.edges() {
            edge_faces.entry(e).or_default().push(*fid);
        }
    }
    for (e, fs) in &edge_faces {
        let feature = fs.len() != 2
            || m.seams.contains(e)
            || m.sharp.contains(e)
            || m.creases.contains_key(e)
            || m.faces[&fs[0]].material != m.faces[&fs[1]].material
            || m.faces[&fs[0]].part != m.faces[&fs[1]].part;
        if feature {
            locked.insert(e.0);
            locked.insert(e.1);
        }
    }
    // Attribute discontinuities (implicit seams).
    let corner_of = |m: &Mesh, f: FaceId, v: VertId| -> Corner {
        let face = &m.faces[&f];
        face.corners[face.index_of(v).unwrap()]
    };
    for (v, fs) in &vf {
        let c0 = corner_of(&m, fs[0], *v);
        if fs.iter().any(|f| {
            let c = corner_of(&m, *f, *v);
            c.uv.distance(c0.uv) > 1e-5 || c.color.distance(c0.color) > 1e-4
        }) {
            locked.insert(*v);
        }
    }
    // Quadrics.
    let mut q: HashMap<VertId, Quadric> = m.verts.keys().map(|v| (*v, Quadric::default())).collect();
    for f in m.faces.values() {
        let n = tri_normal(&m, f).as_dvec3();
        let area = n.length();
        if area < 1e-20 {
            continue;
        }
        let nn = n / area;
        let d = -nn.dot(m.verts[&f.verts[0]].pos.as_dvec3());
        let pq = Quadric::plane(nn, d, area);
        for v in &f.verts {
            q.get_mut(v).unwrap().add(&pq);
        }
    }
    let mut stamp: HashMap<VertId, u32> = m.verts.keys().map(|v| (*v, 0)).collect();
    let mut heap: BinaryHeap<(Reverse<Cost>, VertId, VertId, u32, u32, [u32; 3])> = BinaryHeap::new();
    // Candidate target for collapsing edge (a, b): which point and its cost.
    let candidate = |m: &Mesh, q: &HashMap<VertId, Quadric>, locked: &HashSet<VertId>, a: VertId, b: VertId| -> Option<(f64, Vec3)> {
        let (la, lb) = (locked.contains(&a), locked.contains(&b));
        if la && lb {
            return None;
        }
        let mut qq = q[&a];
        qq.add(&q[&b]);
        let (pa, pb) = (m.verts[&a].pos, m.verts[&b].pos);
        let opts: Vec<Vec3> = if la {
            vec![pa]
        } else if lb {
            vec![pb]
        } else {
            vec![pa, pb, (pa + pb) * 0.5]
        };
        opts.into_iter().map(|p| (qq.eval(p.as_dvec3()).max(0.0), p)).min_by(|x, y| x.0.total_cmp(&y.0))
    };
    let push = |heap: &mut BinaryHeap<_>, m: &Mesh, q: &HashMap<VertId, Quadric>, stamp: &HashMap<VertId, u32>, locked: &HashSet<VertId>, a: VertId, b: VertId| {
        if let Some((c, p)) = candidate(m, q, locked, a, b) {
            heap.push((Reverse(Cost(c)), a, b, stamp[&a], stamp[&b], p.to_array().map(f32::to_bits)));
        }
    };
    for e in edge_faces.keys() {
        push(&mut heap, &m, &q, &stamp, &locked, e.0, e.1);
    }
    let mut tris = m.faces.len();
    while tris > target_tris {
        let Some((_, a, b, sa, sb, pbits)) = heap.pop() else { break };
        if !m.verts.contains_key(&a) || !m.verts.contains_key(&b) || stamp[&a] != sa || stamp[&b] != sb {
            continue;
        }
        // The locked vertex (if any) survives with its id.
        let (a, b) = if locked.contains(&b) { (b, a) } else { (a, b) };
        let target = Vec3::from_array(pbits.map(f32::from_bits));
        let fa = vf.get(&a).cloned().unwrap_or_default();
        let fb = vf.get(&b).cloned().unwrap_or_default();
        let shared: Vec<FaceId> = fa.iter().filter(|f| fb.contains(f)).copied().collect();
        if shared.is_empty() {
            continue;
        }
        // Link condition: the only common neighbours are the shared triangles' third vertices.
        let nbrs = |fs: &[FaceId], v: VertId| -> BTreeSet<VertId> { fs.iter().flat_map(|f| m.faces[f].verts.iter().copied()).filter(|x| *x != v).collect() };
        let (na, nb) = (nbrs(&fa, a), nbrs(&fb, b));
        let common = na.intersection(&nb).count();
        if common != shared.len() {
            continue;
        }
        // No flipped or collapsed triangles around the new point.
        let ok = fa.iter().chain(fb.iter()).filter(|f| !shared.contains(f)).all(|f| {
            let face = &m.faces[f];
            let before = tri_normal(&m, face);
            let pts: Vec<Vec3> = face.verts.iter().map(|v| if *v == a || *v == b { target } else { m.verts[v].pos }).collect();
            let after = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
            after.length_squared() > before.length_squared() * 1e-4 && after.normalize_or_zero().dot(before.normalize_or_zero()) > 0.2
        });
        if !ok {
            continue;
        }
        // The kept vertex's attributes at the new point (unlocked => one value per vertex).
        let (pa, pb) = (m.verts[&a].pos, m.verts[&b].pos);
        let t = if (pb - pa).length_squared() > 0.0 { ((target - pa).dot(pb - pa) / (pb - pa).length_squared()).clamp(0.0, 1.0) } else { 0.0 };
        let ca = corner_of(&m, fa[0], a);
        let cb = corner_of(&m, fb[0], b);
        let new_corner = if locked.contains(&a) { ca } else if locked.contains(&b) { cb } else { ca.lerp(&cb, t) };
        // Weights and morphs blend like the position.
        if !m.weights.is_empty() {
            let wa = m.weights.get(&a).cloned().unwrap_or_default();
            let wb = m.weights.get(&b).cloned().unwrap_or_default();
            let mut acc: HashMap<crate::anim::BoneId, f32> = HashMap::new();
            for (bn, w) in &wa {
                *acc.entry(*bn).or_default() += w * (1.0 - t);
            }
            for (bn, w) in &wb {
                *acc.entry(*bn).or_default() += w * t;
            }
            let mut l: Vec<_> = acc.into_iter().filter(|(_, w)| *w > 1e-5).collect();
            l.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
            l.truncate(4);
            m.weights.insert(a, l);
            m.weights.remove(&b);
        }
        for mo in &mut m.morphs {
            let oa = mo.offsets.get(&a).copied().unwrap_or(Vec3::ZERO);
            let ob = mo.offsets.get(&b).copied().unwrap_or(Vec3::ZERO);
            let o = oa.lerp(ob, t);
            mo.offsets.remove(&b);
            if o.length_squared() > 0.0 {
                mo.offsets.insert(a, o);
            } else {
                mo.offsets.remove(&a);
            }
        }
        for f in &shared {
            m.faces.remove(f);
            tris -= 1;
        }
        for f in fb.iter().filter(|f| !shared.contains(f)) {
            let face = m.faces.get_mut(f).unwrap();
            let i = face.index_of(b).unwrap();
            face.verts[i] = a;
        }
        for f in fa.iter().chain(fb.iter()).filter(|f| !shared.contains(f)) {
            let face = m.faces.get_mut(f).unwrap();
            let i = face.index_of(a).unwrap();
            face.corners[i] = new_corner;
        }
        m.verts.get_mut(&a).unwrap().pos = target;
        m.verts.remove(&b);
        let mut fs: Vec<FaceId> = fa.iter().chain(fb.iter()).filter(|f| !shared.contains(f)).copied().collect();
        fs.sort();
        fs.dedup();
        vf.insert(a, fs.clone());
        vf.remove(&b);
        // The shared triangles' third vertices lose those faces.
        for f in &shared {
            for v in &na {
                if let Some(list) = vf.get_mut(v) {
                    list.retain(|x| x != f);
                }
            }
        }
        let qb = q[&b];
        q.get_mut(&a).unwrap().add(&qb);
        q.remove(&b);
        *stamp.get_mut(&a).unwrap() += 1;
        if locked.contains(&b) {
            locked.insert(a);
        }
        // New candidates around the kept vertex.
        let around: BTreeSet<VertId> = fs.iter().flat_map(|f| m.faces[f].verts.iter().copied()).filter(|v| *v != a).collect();
        for n in around {
            *stamp.get_mut(&n).unwrap() += 1;
            push(&mut heap, &m, &q, &stamp, &locked, a, n);
            let nfs = vf.get(&n).cloned().unwrap_or_default();
            let second: BTreeSet<VertId> = nfs.iter().flat_map(|f| m.faces[f].verts.iter().copied()).filter(|v| *v != n && *v != a).collect();
            for o in second {
                push(&mut heap, &m, &q, &stamp, &locked, n, o);
            }
        }
    }
    m.remove_loose_verts();
    m.prune_edge_sets();
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Primitive;

    #[test]
    fn halves_a_sphere_and_stays_clean() {
        let s = Primitive::UvSphere { radius: 1.0, segments: 32, rings: 16 }.build().unwrap();
        let tris = s.triangle_count();
        let d = decimate(&s, tris / 2);
        assert!(d.faces.len() <= tris / 2 + 2 && d.faces.len() > tris / 4, "{} -> {}", tris, d.faces.len());
        let p = d.problems();
        assert!(p.nonmanifold == 0 && p.flipped == 0 && p.loose == 0, "{:?}", d.validate());
        // Still a ball of about the same size.
        for v in d.verts.values() {
            assert!((v.pos.length() - 1.0).abs() < 0.05, "{}", v.pos.length());
        }
    }

    #[test]
    fn keeps_borders_and_uv_seams() {
        let mut g = crate::primitives::plane(glam::Vec2::splat(2.0), 8, 8);
        let border_before: BTreeSet<VertId> = g.boundary_edges().iter().flat_map(|e| [e.0, e.1]).collect();
        // A UV split down the middle (x = 0): left and right halves use different UV islands.
        let ids: Vec<FaceId> = g.faces.keys().copied().collect();
        for f in ids {
            let right = g.face_center(&g.faces[&f]).x > 0.0;
            let face = g.faces.get_mut(&f).unwrap();
            for c in &mut face.corners {
                c.uv = glam::Vec2::new(if right { 0.75 } else { 0.25 }, 0.5);
            }
        }
        let seam: Vec<VertId> = g.verts.iter().filter(|(_, v)| v.pos.x.abs() < 1e-5).map(|(i, _)| *i).collect();
        let d = decimate(&g, 10);
        for v in border_before.iter().chain(&seam) {
            assert!(d.verts.contains_key(v), "locked v{v} was collapsed");
        }
        assert!(d.faces.len() < g.triangle_count());
    }

    #[test]
    fn weights_stay_normalised() {
        let mut s = Primitive::UvSphere { radius: 1.0, segments: 16, rings: 8 }.build().unwrap();
        let vs: Vec<VertId> = s.verts.keys().copied().collect();
        for v in vs {
            let y = s.verts[&v].pos.y;
            s.weights.insert(v, vec![(0, 0.5 + y * 0.5), (1, 0.5 - y * 0.5)]);
        }
        let d = decimate(&s, s.triangle_count() / 3);
        for w in d.weights.values() {
            let sum: f32 = w.iter().map(|x| x.1).sum();
            assert!((sum - 1.0).abs() < 1e-3, "{w:?}");
        }
    }
}
