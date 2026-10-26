//! Keeping meshes apart: push vertices out of another surface, and find
//! where two meshes cross.

use std::collections::BTreeSet;

use glam::Vec3;

use crate::bvh::{Bvh, Tri};
use crate::mesh::{FaceId, Mesh, VertId};

/// Move `verts` that are inside the target surface, or closer to it than
/// `clearance` on the outside, to `clearance` outside it (along the nearest
/// triangle's normal). The target must be closed and face outward; positions
/// are in the target BVH's space. Returns the moved vertices.
pub fn push_out(mesh: &mut Mesh, verts: &BTreeSet<VertId>, target: &Bvh, clearance: f32) -> Vec<VertId> {
    let mut moved = Vec::new();
    for v in verts {
        let Some(p) = mesh.verts.get(v).map(|x| x.pos) else { continue };
        let Some((tri, q, _, d)) = target.closest_point(p) else { continue };
        let n = target.tris[tri].normal();
        let inside = (p - q).dot(n) < 0.0;
        if inside || d < clearance {
            mesh.verts.get_mut(v).unwrap().pos = q + n * clearance;
            moved.push(*v);
        }
    }
    moved
}

/// Does segment a-b cross triangle t (excluding touching at the ends)?
fn segment_hits(a: Vec3, b: Vec3, t: &Tri) -> Option<Vec3> {
    let d = b - a;
    let len = d.length();
    if len < 1e-9 {
        return None;
    }
    let x = crate::bvh::ray_tri(a, d / len, t)?;
    (x > 1e-6 && x < len - 1e-6).then(|| a + d / len * x)
}

/// Do two triangles cross? Returns a point on the crossing.
fn tri_tri(a: &Tri, b: &Tri) -> Option<Vec3> {
    for (s, t) in [(a, b), (b, a)] {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            if let Some(p) = segment_hits(s.p[i], s.p[j], t) {
                return Some(p);
            }
        }
    }
    None
}

/// Where mesh `a` (triangles in a common space) crosses the surface in `b`:
/// (face of a, face of b, a point on the crossing).
pub fn intersections(a: &Bvh, b: &Bvh) -> Vec<(FaceId, FaceId, Vec3)> {
    let mut out = Vec::new();
    for ta in &a.tris {
        let lo = ta.p[0].min(ta.p[1]).min(ta.p[2]);
        let hi = ta.p[0].max(ta.p[1]).max(ta.p[2]);
        for k in b.overlapping(lo, hi) {
            if let Some(p) = tri_tri(ta, &b.tris[k]) {
                out.push((ta.face, b.tris[k].face, p));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;
    use glam::Mat4;

    #[test]
    fn push_out_and_intersections() {
        let body = cube(Vec3::splat(2.0));
        let bvh = Bvh::new(&body, Mat4::IDENTITY);
        // A smaller cube poking through the body's +X face.
        // Inner corners at x = 0.7, |y|, |z| = 0.2: nearest to the body's +X face.
        let mut hair = cube(Vec3::splat(0.4));
        hair.transform(Mat4::from_translation(Vec3::new(0.9, 0.0, 0.0)));
        let crossings = intersections(&Bvh::new(&hair, Mat4::IDENTITY), &bvh);
        assert!(!crossings.is_empty());
        assert!(crossings.iter().all(|(_, _, p)| (p.x - 1.0).abs() < 1e-4), "they cross on the body's x = 1 face");
        // Push it out: every vertex ends at least `clearance` outside.
        let all: BTreeSet<VertId> = hair.verts.keys().copied().collect();
        let moved = push_out(&mut hair, &all, &bvh, 0.05);
        assert_eq!(moved.len(), 4, "the four vertices inside the body");
        assert!(hair.verts.values().all(|v| v.pos.x >= 1.05 - 1e-4));
        assert!(intersections(&Bvh::new(&hair, Mat4::IDENTITY), &bvh).is_empty());
        // Separate meshes don't intersect.
        let mut far = cube(Vec3::ONE);
        far.transform(Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        assert!(intersections(&Bvh::new(&far, Mat4::IDENTITY), &bvh).is_empty());
    }
}
