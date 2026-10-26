use std::collections::BTreeSet;

use glam::{Vec2, Vec3};
use polly_core::geom::triangulate_polygon;
use polly_core::ids;
use polly_core::mesh::{FaceId, Mesh, short_list};
use serde::{Deserialize, Serialize};

use crate::uv_islands;

/// Stretch reported for faces with (near) zero uv area.
const MAX_STRETCH: f32 = 999.0;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IslandInfo {
    pub faces: Vec<String>,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// x, y, w, h in texture pixels (texels touched by the island's bbox).
    pub pixel_rect: [u32; 4],
}

/// UV layout quality summary, meant to be read by an agent.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct UvReport {
    /// Connected pieces of the current layout (see [`crate::uv_islands`]).
    pub islands: Vec<IslandInfo>,
    /// Fraction of the 0..1 square covered by uv triangles (overlaps counted once).
    pub coverage: f32,
    /// Per-face distortion relative to the mean texel density: 1.0 is perfect,
    /// 2.0 means some direction is stretched or squashed by 2x.
    pub max_stretch: f32,
    /// Area-weighted mean of the per-face distortion.
    pub mean_stretch: f32,
    /// Most distorted faces (stretch > 1.1), worst first.
    pub worst_faces: Vec<(String, f32)>,
    /// Faces whose uv triangles overlap another face's.
    pub overlapping_faces: Vec<String>,
    /// Faces with a uv outside 0..1.
    pub out_of_bounds_faces: Vec<String>,
    /// Faces whose uvs are mirrored (wound the wrong way on the texture).
    pub flipped_faces: Vec<String>,
    /// Mean texels per world unit at the given texture size.
    pub texel_density: f32,
    pub warnings: Vec<String>,
}

struct UvTri {
    face: FaceId,
    uv: [Vec2; 3],
    pos: [Vec3; 3],
}

impl UvTri {
    fn uv_area(&self) -> f32 {
        (self.uv[1] - self.uv[0]).perp_dot(self.uv[2] - self.uv[0]) * 0.5
    }
    fn area_3d(&self) -> f32 {
        (self.pos[1] - self.pos[0]).cross(self.pos[2] - self.pos[0]).length() * 0.5
    }
    /// Singular values of the 3-D -> uv (pixel) map.
    fn singular_values(&self, px: Vec2) -> (f32, f32) {
        let (ab, ac) = (self.pos[1] - self.pos[0], self.pos[2] - self.pos[0]);
        let n = ab.cross(ac);
        if n.length() < 1e-12 {
            return (0.0, 0.0);
        }
        let x = ab.normalize();
        let y = n.normalize().cross(x);
        let (p1, p2) = (Vec2::new(ab.length(), 0.0), Vec2::new(ac.dot(x), ac.dot(y)));
        let (q1, q2) = ((self.uv[1] - self.uv[0]) * px, (self.uv[2] - self.uv[0]) * px);
        let p = glam::Mat2::from_cols(p1, p2);
        let j = glam::Mat2::from_cols(q1, q2) * p.inverse();
        let e = j.x_axis.length_squared() + j.y_axis.length_squared();
        let det = j.determinant().abs();
        let root = (e * e - 4.0 * det * det).max(0.0).sqrt();
        (((e + root) * 0.5).sqrt(), ((e - root) * 0.5).max(0.0).sqrt())
    }
}

fn uv_tris(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Vec<UvTri> {
    let mut out = Vec::new();
    for &f in faces {
        let Some(face) = mesh.faces.get(&f) else { continue };
        let p = mesh.face_positions(face);
        for [a, b, c] in triangulate_polygon(&p) {
            let uv = [a, b, c].map(|i| face.corners[i].uv);
            out.push(UvTri { face: f, uv, pos: [p[a], p[b], p[c]] });
        }
    }
    out
}

/// Measure the current layout of `faces` against a texture of `texture_size`.
pub fn report(mesh: &Mesh, faces: &BTreeSet<FaceId>, texture_size: [u32; 2]) -> UvReport {
    let px = Vec2::new(texture_size[0].max(1) as f32, texture_size[1].max(1) as f32);
    let tris = uv_tris(mesh, faces);
    let mut warnings = Vec::new();

    let islands = uv_islands(mesh, faces)
        .into_iter()
        .map(|isl| {
            let (lo, hi) = crate::uv_bounds(mesh, &isl);
            let (a, b) = ((lo * px + 1e-3).floor().max(Vec2::ZERO), (hi * px - 1e-3).ceil().max(Vec2::ZERO));
            let b = b.max(a + 1.0);
            IslandInfo {
                faces: ids::faces(isl.iter().copied()),
                uv_min: lo.to_array(),
                uv_max: hi.to_array(),
                pixel_rect: [a.x as u32, a.y as u32, (b.x - a.x) as u32, (b.y - a.y) as u32],
            }
        })
        .collect();

    let total_uv: f32 = tris.iter().map(|t| t.uv_area().abs() * px.x * px.y).sum();
    let total_3d: f32 = tris.iter().map(UvTri::area_3d).sum();
    let density = if total_3d > 0.0 { (total_uv / total_3d).sqrt() } else { 0.0 };

    // Stretch: max(sigma_max, 1/sigma_min) after normalizing by the mean density.
    let mut face_stretch: std::collections::BTreeMap<FaceId, f32> = Default::default();
    let (mut weighted, mut weight) = (0.0, 0.0);
    for t in &tris {
        let a3 = t.area_3d();
        if a3 < 1e-12 {
            continue;
        }
        let (s1, s2) = t.singular_values(px);
        let s = if density <= 0.0 || s2 < 1e-6 * density {
            MAX_STRETCH
        } else {
            (s1 / density).max(density / s2).min(MAX_STRETCH)
        };
        let e = face_stretch.entry(t.face).or_insert(1.0);
        *e = e.max(s);
        weighted += s * a3;
        weight += a3;
    }
    let max_stretch = face_stretch.values().copied().fold(1.0, f32::max);
    let mean_stretch = if weight > 0.0 { weighted / weight } else { 1.0 };
    let mut worst: Vec<(FaceId, f32)> = face_stretch.iter().filter(|(_, s)| **s > 1.1).map(|(f, s)| (*f, *s)).collect();
    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    let degenerate = face_stretch.values().filter(|s| **s >= MAX_STRETCH).count();
    if degenerate > 0 {
        warnings.push(format!("{degenerate} face(s) have (near) zero uv area"));
    }

    let overlapping = overlaps(&tris);
    let out_of_bounds: Vec<FaceId> = faces
        .iter()
        .filter(|f| {
            mesh.faces.get(f).is_some_and(|face| {
                face.corners.iter().any(|c| c.uv.cmplt(Vec2::splat(-1e-4)).any() || c.uv.cmpgt(Vec2::splat(1.0 + 1e-4)).any())
            })
        })
        .copied()
        .collect();
    // In image space (v down) a face that is counter-clockwise from outside
    // has negative signed area.
    let flipped: Vec<FaceId> = faces
        .iter()
        .filter(|f| {
            mesh.faces.get(f).is_some_and(|face| {
                let n = face.corners.len();
                (0..n).map(|i| face.corners[i].uv.perp_dot(face.corners[(i + 1) % n].uv)).sum::<f32>() > 1e-9
            })
        })
        .copied()
        .collect();
    for (list, what) in [
        (&overlapping, "overlap other faces in uv"),
        (&out_of_bounds, "lie outside 0..1"),
        (&flipped, "are mirrored in uv"),
    ] {
        if !list.is_empty() {
            warnings.push(format!("{} face(s) {what}: {}", list.len(), short_list(&ids::faces(list.iter().copied()))));
        }
    }

    UvReport {
        islands,
        coverage: coverage(&tris, texture_size),
        max_stretch,
        mean_stretch,
        worst_faces: worst.into_iter().take(8).map(|(f, s)| (ids::face(f), s)).collect(),
        overlapping_faces: ids::faces(overlapping),
        out_of_bounds_faces: ids::faces(out_of_bounds),
        flipped_faces: ids::faces(flipped),
        texel_density: density,
        warnings,
    }
}

/// Rasterize the triangles at (up to 4x) texture resolution and count samples.
fn coverage(tris: &[UvTri], texture_size: [u32; 2]) -> f32 {
    let k = (1024 / texture_size[0].max(texture_size[1]).max(1)).clamp(1, 4);
    let (w, h) = ((texture_size[0].max(1) * k) as usize, (texture_size[1].max(1) * k) as usize);
    let mut hit = vec![false; w * h];
    let scale = Vec2::new(w as f32, h as f32);
    for t in tris {
        let p = t.uv.map(|q| q * scale);
        let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
        let (x0, y0) = (lo.x.floor().max(0.0) as usize, lo.y.floor().max(0.0) as usize);
        let (x1, y1) = ((hi.x.ceil().max(0.0) as usize).min(w), (hi.y.ceil().max(0.0) as usize).min(h));
        let sign = if t.uv_area() < 0.0 { -1.0 } else { 1.0 };
        for y in y0..y1 {
            for x in x0..x1 {
                let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                if (0..3).all(|i| sign * (p[(i + 1) % 3] - p[i]).perp_dot(c - p[i]) >= 0.0) {
                    hit[y * w + x] = true;
                }
            }
        }
    }
    hit.iter().filter(|h| **h).count() as f32 / (w * h) as f32
}

/// Faces with a uv triangle whose interior overlaps another face's.
fn overlaps(tris: &[UvTri]) -> Vec<FaceId> {
    let bbox = |t: &UvTri| (t.uv[0].min(t.uv[1]).min(t.uv[2]), t.uv[0].max(t.uv[1]).max(t.uv[2]));
    let mut order: Vec<usize> = (0..tris.len()).filter(|&i| tris[i].uv_area().abs() > 1e-12).collect();
    order.sort_by(|&a, &b| bbox(&tris[a]).0.x.total_cmp(&bbox(&tris[b]).0.x));
    let mut out = BTreeSet::new();
    for (k, &i) in order.iter().enumerate() {
        let (lo, hi) = bbox(&tris[i]);
        for &j in &order[k + 1..] {
            let (lo2, hi2) = bbox(&tris[j]);
            if lo2.x >= hi.x {
                break;
            }
            if tris[i].face == tris[j].face || lo2.y >= hi.y || hi2.y <= lo.y {
                continue;
            }
            if tris_overlap(&tris[i].uv, &tris[j].uv) {
                out.insert(tris[i].face);
                out.insert(tris[j].face);
            }
        }
    }
    out.into_iter().collect()
}

/// Separating-axis test that ignores touching edges and vertices.
fn tris_overlap(a: &[Vec2; 3], b: &[Vec2; 3]) -> bool {
    const EPS: f32 = 1e-6;
    for tri in [a, b] {
        for i in 0..3 {
            let axis = (tri[(i + 1) % 3] - tri[i]).perp();
            let len = axis.length();
            if len < 1e-12 {
                continue;
            }
            let proj = |t: &[Vec2; 3]| {
                let d = t.map(|p| p.dot(axis) / len);
                (d[0].min(d[1]).min(d[2]), d[0].max(d[1]).max(d[2]))
            };
            let ((amin, amax), (bmin, bmax)) = (proj(a), proj(b));
            if amax <= bmin + EPS || bmax <= amin + EPS {
                return false;
            }
        }
    }
    true
}
