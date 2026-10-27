//! Bounding-volume hierarchy over a mesh's triangulated faces, for ray casts
//! (ambient occlusion) and closest-point queries (mirrored painting).

use glam::{Mat4, Vec2, Vec3};

use crate::mesh::{FaceId, Mesh};

/// One triangle of a (possibly n-gon) face.
#[derive(Debug, Clone, Copy)]
pub struct Tri {
    pub p: [Vec3; 3],
    pub face: FaceId,
    /// Corner indices into the face's loop (for interpolating uvs/colours).
    pub corners: [usize; 3],
}

impl Tri {
    pub fn normal(&self) -> Vec3 {
        (self.p[1] - self.p[0]).cross(self.p[2] - self.p[0]).normalize_or_zero()
    }

    /// Interpolate the face's corner uvs at barycentric `b`.
    pub fn uv(&self, mesh: &Mesh, b: Vec3) -> Vec2 {
        let f = &mesh.faces[&self.face];
        f.corners[self.corners[0]].uv * b.x + f.corners[self.corners[1]].uv * b.y + f.corners[self.corners[2]].uv * b.z
    }
}

#[derive(Debug, Clone, Copy)]
struct Node {
    lo: Vec3,
    hi: Vec3,
    /// Leaf: first triangle and count. Inner: left child index (right = left + 1), count 0.
    start: usize,
    count: usize,
}

pub struct Bvh {
    pub tris: Vec<Tri>,
    nodes: Vec<Node>,
}

const LEAF: usize = 4;

impl Bvh {
    /// Build over all faces of `mesh`, with positions transformed by `m`.
    pub fn new(mesh: &Mesh, m: Mat4) -> Self {
        let mut tris = Vec::new();
        for (&fid, f) in &mesh.faces {
            let pts: Vec<Vec3> = f.verts.iter().map(|v| m.transform_point3(mesh.pos(*v))).collect();
            for [a, b, c] in crate::geom::triangulate_polygon(&pts) {
                tris.push(Tri { p: [pts[a], pts[b], pts[c]], face: fid, corners: [a, b, c] });
            }
        }
        let mut bvh = Bvh { tris, nodes: Vec::new() };
        if !bvh.tris.is_empty() {
            bvh.nodes.push(Node { lo: Vec3::ZERO, hi: Vec3::ZERO, start: 0, count: 0 });
            let n = bvh.tris.len();
            bvh.build(0, 0, n);
        }
        bvh
    }

    fn build(&mut self, node: usize, start: usize, end: usize) {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for t in &self.tris[start..end] {
            for p in t.p {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if end - start <= LEAF {
            self.nodes[node] = Node { lo, hi, start, count: end - start };
            return;
        }
        let axis = (hi - lo).max_position();
        let centroid = |t: &Tri| (t.p[0] + t.p[1] + t.p[2])[axis];
        self.tris[start..end].sort_by(|a, b| centroid(a).total_cmp(&centroid(b)));
        let mid = (start + end) / 2;
        let left = self.nodes.len();
        self.nodes.push(Node { lo, hi, start: 0, count: 0 });
        self.nodes.push(Node { lo, hi, start: 0, count: 0 });
        self.nodes[node] = Node { lo, hi, start: left, count: 0 };
        self.build(left, start, mid);
        self.build(left + 1, mid, end);
    }

    /// Does a ray from `o` along unit `d` hit anything within (eps, `tmax`)?
    pub fn any_hit(&self, o: Vec3, d: Vec3, tmax: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv = d.recip();
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let n = self.nodes[i];
            if !ray_box(o, inv, n.lo, n.hi, tmax) {
                continue;
            }
            if n.count > 0 {
                if self.tris[n.start..n.start + n.count].iter().any(|t| ray_tri(o, d, t).is_some_and(|x| x > 1e-4 && x < tmax)) {
                    return true;
                }
            } else {
                stack.push(n.start);
                stack.push(n.start + 1);
            }
        }
        false
    }

    /// Nearest hit of a ray from `o` along unit `d` within (`tmin`, `tmax`):
    /// (triangle index, distance, barycentric weights of its three corners).
    pub fn first_hit(&self, o: Vec3, d: Vec3, tmin: f32, tmax: f32) -> Option<(usize, f32, Vec3)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = d.recip();
        let mut best: Option<(usize, f32)> = None;
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let n = self.nodes[i];
            let limit = best.map_or(tmax, |b| b.1);
            if !ray_box(o, inv, n.lo, n.hi, limit) {
                continue;
            }
            if n.count > 0 {
                for k in n.start..n.start + n.count {
                    if let Some(t) = ray_tri(o, d, &self.tris[k]) {
                        if t > tmin && t < best.map_or(tmax, |b| b.1) {
                            best = Some((k, t));
                        }
                    }
                }
            } else {
                stack.push(n.start);
                stack.push(n.start + 1);
            }
        }
        let (k, t) = best?;
        let p = o + d * t;
        let [a, b, c] = self.tris[k].p;
        let (v0, v1, v2) = (b - a, c - a, p - a);
        let (d00, d01, d11, d20, d21) = (v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1));
        let den = (d00 * d11 - d01 * d01).max(1e-20);
        let (v, w) = ((d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den);
        Some((k, t, Vec3::new(1.0 - v - w, v, w)))
    }

    /// Indices of triangles whose bounds overlap the box [lo, hi].
    pub fn overlapping(&self, lo: Vec3, hi: Vec3) -> Vec<usize> {
        let mut out = Vec::new();
        if self.nodes.is_empty() {
            return out;
        }
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let n = self.nodes[i];
            if n.lo.cmpgt(hi).any() || n.hi.cmplt(lo).any() {
                continue;
            }
            if n.count > 0 {
                for k in n.start..n.start + n.count {
                    let p = self.tris[k].p;
                    let (tlo, thi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
                    if !(tlo.cmpgt(hi).any() || thi.cmplt(lo).any()) {
                        out.push(k);
                    }
                }
            } else {
                stack.push(n.start);
                stack.push(n.start + 1);
            }
        }
        out
    }

    /// Nearest point on the surface to `p`: (triangle index, point, barycentric, distance).
    pub fn closest_point(&self, p: Vec3) -> Option<(usize, Vec3, Vec3, f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<(usize, Vec3, Vec3, f32)> = None;
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let n = self.nodes[i];
            let box_d = (n.lo - p).max(p - n.hi).max(Vec3::ZERO).length();
            if best.is_some_and(|b| box_d >= b.3) {
                continue;
            }
            if n.count > 0 {
                for k in n.start..n.start + n.count {
                    let (q, b) = closest_on_tri(p, &self.tris[k].p);
                    let d = q.distance(p);
                    if best.is_none_or(|bb| d < bb.3) {
                        best = Some((k, q, b, d));
                    }
                }
            } else {
                stack.push(n.start);
                stack.push(n.start + 1);
            }
        }
        best
    }
}

fn ray_box(o: Vec3, inv: Vec3, lo: Vec3, hi: Vec3, tmax: f32) -> bool {
    let t1 = (lo - o) * inv;
    let t2 = (hi - o) * inv;
    let tmin = t1.min(t2).max_element().max(0.0);
    let tfar = t1.max(t2).min_element().min(tmax);
    tmin <= tfar
}

/// Möller–Trumbore; returns the distance along the ray (two-sided).
pub(crate) fn ray_tri(o: Vec3, d: Vec3, t: &Tri) -> Option<f32> {
    let (e1, e2) = (t.p[1] - t.p[0], t.p[2] - t.p[0]);
    let h = d.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-12 {
        return None;
    }
    let f = 1.0 / a;
    let s = o - t.p[0];
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * d.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(f * e2.dot(q))
}

/// Closest point on a triangle and its barycentric coordinates (Ericson, RTCD 5.1.5).
pub fn closest_on_tri(p: Vec3, t: &[Vec3; 3]) -> (Vec3, Vec3) {
    let [a, b, c] = *t;
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Vec3::X);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Vec3::Y);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Vec3::new(1.0 - v, v, 0.0));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Vec3::Z);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Vec3::new(1.0 - w, 0.0, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, Vec3::new(0.0, 1.0 - w, w));
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    (a + ab * v + ac * w, Vec3::new(1.0 - v - w, v, w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_and_closest_points_on_a_cube() {
        let m = crate::primitives::cube(Vec3::ONE);
        let bvh = Bvh::new(&m, Mat4::IDENTITY);
        assert_eq!(bvh.tris.len(), 12);
        // From the centre every direction hits a wall within 0.9.
        for d in [Vec3::X, Vec3::NEG_Y, Vec3::new(1.0, 1.0, 1.0).normalize()] {
            assert!(bvh.any_hit(Vec3::ZERO, d, 0.9));
            assert!(!bvh.any_hit(Vec3::ZERO, d, 0.4));
        }
        // Outside, pointing away: no hit.
        assert!(!bvh.any_hit(Vec3::new(0.0, 2.0, 0.0), Vec3::Y, 10.0));
        let (_, q, _, d) = bvh.closest_point(Vec3::new(0.2, 3.0, -0.1)).unwrap();
        assert!((q - Vec3::new(0.2, 0.5, -0.1)).length() < 1e-5 && (d - 2.5).abs() < 1e-5);
    }
}
