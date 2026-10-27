//! Hair cards: strips grown from a scalp surface, each mapped to one column
//! of a strand texture (alpha cut out). The classic real-time hair setup.

use std::collections::BTreeSet;

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::error::{Result, invalid};
use crate::mesh::{Corner, FaceId, Mesh, VertId};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct HairCards {
    /// Number of cards.
    pub count: u32,
    /// Card length (object units; default: 40% of the scalp's size).
    pub length: Option<f32>,
    /// Card width at the root (default: length / 6).
    pub width: Option<f32>,
    /// Segments along each card (bend resolution).
    pub segments: u32,
    /// 0 = lie along the surface, 1 = stand straight out.
    pub lift: f32,
    /// How much cards droop downwards along their length (0..1).
    pub gravity: f32,
    /// Random variation of direction and length (0..1).
    pub random: f32,
    /// Width at the tip relative to the root.
    pub taper: f32,
    /// Texture columns: each card uses one (strand variations).
    pub variations: u32,
    pub seed: u64,
    /// Direction cards are combed towards along the surface (default down, [0,-1,0]).
    pub comb: [f32; 3],
}

impl Default for HairCards {
    fn default() -> Self {
        HairCards {
            count: 60,
            length: None,
            width: None,
            segments: 4,
            lift: 0.25,
            gravity: 0.5,
            random: 0.3,
            taper: 0.4,
            variations: 4,
            seed: 1,
            comb: [0.0, -1.0, 0.0],
        }
    }
}

fn rand(seed: u64, i: u64, k: u64) -> f32 {
    let mut h = seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Grow hair cards from `faces` of `scalp`. Returns the cards mesh and, per
/// card vertex, the scalp vertex nearest its root (to copy skin weights).
pub fn hair_cards(scalp: &Mesh, faces: &BTreeSet<FaceId>, o: &HairCards) -> Result<(Mesh, Vec<(VertId, VertId)>)> {
    if o.count == 0 || o.count > 20_000 {
        return invalid("count must be 1..20000");
    }
    // Area-weighted triangles to sample roots from.
    let mut tris: Vec<([Vec3; 3], [VertId; 3], f32)> = Vec::new();
    for f in faces {
        let Some(face) = scalp.faces.get(f) else { return invalid(format!("no face f{f}")) };
        let pts = scalp.face_positions(face);
        for [a, b, c] in crate::geom::triangulate_polygon(&pts) {
            let area = (pts[b] - pts[a]).cross(pts[c] - pts[a]).length() * 0.5;
            if area > 0.0 {
                tris.push(([pts[a], pts[b], pts[c]], [face.verts[a], face.verts[b], face.verts[c]], area));
            }
        }
    }
    if tris.is_empty() {
        return invalid("no faces to grow hair from");
    }
    let total: f32 = tris.iter().map(|t| t.2).sum();
    let (lo, hi) = tris.iter().flat_map(|t| t.0).fold((Vec3::MAX, Vec3::MIN), |(l, h), p| (l.min(p), h.max(p)));
    let size = (hi - lo).length().max(1e-4);
    let length = o.length.unwrap_or(size * 0.4).max(1e-4);
    let width = o.width.unwrap_or(length / 6.0).max(1e-5);
    let segs = o.segments.clamp(1, 32);
    let vars = o.variations.clamp(1, 16);
    let comb = Vec3::from_array(o.comb).normalize_or(Vec3::NEG_Y);
    let mut out = Mesh::new();
    let mut roots = Vec::new();
    for i in 0..o.count as u64 {
        // Pick a triangle by area, then a uniform point in it.
        let mut r = rand(o.seed, i, 0) * total;
        let tri = tris.iter().find(|t| {
            r -= t.2;
            r <= 0.0
        });
        let (p, vids, _) = tri.unwrap_or(tris.last().unwrap());
        let (mut u, mut v) = (rand(o.seed, i, 1), rand(o.seed, i, 2));
        if u + v > 1.0 {
            (u, v) = (1.0 - u, 1.0 - v);
        }
        let root = p[0] + (p[1] - p[0]) * u + (p[2] - p[0]) * v;
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or(Vec3::Y);
        let nearest = if u > 0.5 { vids[1] } else if v > 0.5 { vids[2] } else { vids[0] };
        // Combed direction along the surface, lifted off it, with variation.
        let along = (comb - n * comb.dot(n)).normalize_or(n.any_orthonormal_vector());
        let jitter = Vec3::new(rand(o.seed, i, 3) - 0.5, rand(o.seed, i, 4) - 0.5, rand(o.seed, i, 5) - 0.5) * o.random;
        let mut dir = (along * (1.0 - o.lift) + n * o.lift + jitter).normalize_or(n);
        let len = length * (1.0 - o.random * 0.4 * rand(o.seed, i, 6));
        let step = len / segs as f32;
        let column = (rand(o.seed, i, 7) * vars as f32) as u32 % vars;
        let (u0, u1) = (column as f32 / vars as f32, (column + 1) as f32 / vars as f32);
        // Spine points.
        let mut spine = vec![root + n * size * 0.002];
        for _ in 0..segs {
            dir = (dir + Vec3::NEG_Y * (o.gravity * 2.0 / segs as f32)).normalize_or(dir);
            spine.push(*spine.last().unwrap() + dir * step);
        }
        // Cross direction: perpendicular to the strand, tangent to the scalp.
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (k, sp) in spine.iter().enumerate() {
            let t = k as f32 / segs as f32;
            let d = if k + 1 < spine.len() { spine[k + 1] - *sp } else { *sp - spine[k - 1] };
            let side = d.cross(n).normalize_or(n.any_orthonormal_vector());
            let w = width * (1.0 - (1.0 - o.taper) * t) * 0.5;
            left.push(out.add_vert(*sp - side * w));
            right.push(out.add_vert(*sp + side * w));
        }
        for k in 0..segs as usize {
            let (t0, t1) = (k as f32 / segs as f32, (k + 1) as f32 / segs as f32);
            let c = |uu: f32, vv: f32| Corner { uv: Vec2::new(uu, vv), color: Vec4::ONE };
            // Counter-clockwise seen from outside (along n).
            let verts = vec![left[k], left[k + 1], right[k + 1], right[k]];
            let corners = vec![c(u0, t0), c(u0, t1), c(u1, t1), c(u1, t0)];
            let fid = out.add_face_with(verts, corners, 0, true)?;
            let fnorm = out.face_normal(&out.faces[&fid]);
            if fnorm.dot(n) < 0.0 {
                out.faces.get_mut(&fid).unwrap().reverse();
            }
        }
        for v in left.iter().chain(&right) {
            roots.push((*v, nearest));
        }
    }
    Ok((out, roots))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Primitive;

    #[test]
    fn grows_cards_with_uv_columns() {
        let head = Primitive::UvSphere { radius: 0.1, segments: 12, rings: 6 }.build().unwrap();
        let top: BTreeSet<FaceId> = head.faces.iter().filter(|(_, f)| head.face_center(f).y > 0.03).map(|(i, _)| *i).collect();
        let o = HairCards { count: 30, variations: 3, ..Default::default() };
        let (m, roots) = hair_cards(&head, &top, &o).unwrap();
        assert_eq!(m.faces.len(), 30 * 4);
        assert_eq!(roots.len(), m.verts.len());
        for f in m.faces.values() {
            let us: Vec<f32> = f.corners.iter().map(|c| c.uv.x).collect();
            let col = (us.iter().cloned().fold(1.0, f32::min) * 3.0).round();
            assert!((0.0..3.0).contains(&col));
            assert!(f.corners.iter().all(|c| (0.0..=1.0).contains(&c.uv.y)));
        }
        // Cards hang down: tips below roots on average.
        let ys: Vec<f32> = m.verts.values().map(|v| v.pos.y).collect();
        assert!(ys.iter().cloned().fold(f32::MAX, f32::min) < 0.03);
        assert!(hair_cards(&head, &BTreeSet::new(), &o).is_err());
    }
}
