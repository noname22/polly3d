//! Map baking for PBR texturing: tangent-space normal maps and ambient
//! occlusion from a detailed source surface (or the mesh itself), and
//! curvature / cavity masks. Every texel of the target's UV layout looks up
//! the source surface along its normal.

use std::collections::{BTreeSet, HashMap};

use glam::{Vec3, Vec4};
use polly_core::bvh::Bvh;
use polly_core::mesh::{Edge, FaceId, Mesh, VertId};
use polly_core::scene::Texture;
use serde::{Deserialize, Serialize};

use crate::canvas::{Paint, apply, hash};
use crate::color::Rgba;
use crate::surface::{Sample, SurfaceMap};

/// The detailed surface maps are baked from, in the same space as the target map.
pub struct Source<'a> {
    pub mesh: &'a Mesh,
    pub bvh: Bvh,
    normals: HashMap<(FaceId, usize), Vec3>,
    curvature: Option<HashMap<VertId, f32>>,
}

impl<'a> Source<'a> {
    pub fn new(mesh: &'a Mesh) -> Self {
        Source { mesh, bvh: Bvh::new(mesh, glam::Mat4::IDENTITY), normals: mesh.corner_normals(), curvature: None }
    }
}

/// Where a target texel lands on the source: face, corner weights, point.
#[derive(Debug, Clone, Copy)]
struct Hit {
    face: FaceId,
    corners: [(usize, f32); 3],
    pos: Vec3,
}

/// Find the source surface for a texel: along its normal, the outermost
/// source surface within `max_dist` (ray from outside inwards).
fn project(src: &Source, s: &Sample, max_dist: f32) -> Option<Hit> {
    let n = s.normal.normalize_or_zero();
    if n == Vec3::ZERO {
        return None;
    }
    let origin = s.pos + n * max_dist;
    let (k, t, b) = src.bvh.first_hit(origin, -n, 0.0, 2.0 * max_dist)?;
    let tri = &src.bvh.tris[k];
    Some(Hit { face: tri.face, corners: [(tri.corners[0], b.x), (tri.corners[1], b.y), (tri.corners[2], b.z)], pos: origin - n * t })
}

fn hit_normal(src: &Source, h: &Hit) -> Vec3 {
    h.corners.iter().map(|(c, w)| src.normals.get(&(h.face, *c)).copied().unwrap_or(Vec3::ZERO) * *w).sum::<Vec3>().normalize_or_zero()
}

fn self_hit(s: &Sample) -> Hit {
    Hit { face: s.face, corners: s.corners, pos: s.pos }
}

/// Bake settings shared by the maps.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BakeOptions {
    /// How far (world units) the source may be from the target surface, along its normal.
    pub max_distance: Option<f32>,
    /// AO rays per texel (default 32).
    pub ao_samples: Option<u32>,
    /// AO: how far occluders count (default 30% of the model size).
    pub ao_distance: Option<f32>,
    /// Curvature/cavity contrast (default 1).
    pub contrast: Option<f32>,
}

/// Texels with no source surface within reach.
pub struct BakeReport {
    pub changed: usize,
    pub missed: usize,
}

fn encode(v: Vec3) -> Rgba {
    let c = (v * 0.5 + Vec3::splat(0.5)).clamp(Vec3::ZERO, Vec3::ONE) * 255.0;
    [c.x.round() as u8, c.y.round() as u8, c.z.round() as u8, 255]
}

/// Tangent-space normal map of `src` on the target `map`. `tangents` are the
/// target's per-corner MikkTSpace tangents (xyz + handedness) as exported;
/// the bitangent is `w * cross(N, T)`, as glTF / Bevy shaders rebuild it.
/// Without a source (`None`), the target's own shading is baked (flat map).
pub fn bake_normal(
    tex: &mut Texture,
    map: &SurfaceMap,
    target: &Mesh,
    tangents: &HashMap<(FaceId, usize), [f32; 4]>,
    src: Option<&Source>,
    max_dist: f32,
    p: &Paint,
) -> BakeReport {
    let tnormals = target.corner_normals();
    let mut missed = 0;
    let changed = apply(tex, &map.mask(), p, |x, y, _| {
        let s = map.get(x, y)?;
        let mut n = Vec3::ZERO;
        let mut t = Vec4::ZERO;
        for (c, w) in s.corners {
            n += tnormals.get(&(s.face, c)).copied().unwrap_or(s.normal) * w;
            t += Vec4::from(tangents.get(&(s.face, c)).copied().unwrap_or([1.0, 0.0, 0.0, 1.0])) * w;
        }
        let n = n.normalize_or(s.normal);
        let sign = if t.w < 0.0 { -1.0 } else { 1.0 };
        let t = (t.truncate() - n * n.dot(t.truncate())).normalize_or(n.any_orthonormal_vector());
        let b = n.cross(t) * sign;
        let hn = match src {
            Some(src) => match project(src, s, max_dist) {
                Some(h) => hit_normal(src, &h),
                None => {
                    missed += 1;
                    n
                }
            },
            None => n,
        };
        Some(encode(Vec3::new(hn.dot(t), hn.dot(b), hn.dot(n)).normalize_or(Vec3::Z)))
    });
    BakeReport { changed, missed }
}

/// Ambient occlusion (white = open, darker = occluded) written as grey,
/// sampled on the source surface (or the target itself) against `occluders`.
pub fn bake_ao(tex: &mut Texture, map: &SurfaceMap, src: Option<&Source>, occluders: &Bvh, o: &BakeOptions, max_dist: f32, p: &Paint) -> BakeReport {
    let Some((lo, hi)) = map.bounds() else { return BakeReport { changed: 0, missed: 0 } };
    let size = (hi - lo).length().max(1e-4);
    let dist = o.ao_distance.unwrap_or(size * 0.3);
    let samples = o.ao_samples.unwrap_or(32).clamp(1, 512);
    let mut missed = 0;
    let changed = apply(tex, &map.mask(), p, |x, y, _| {
        let s = map.get(x, y)?;
        let (pos, n) = match src {
            Some(src) => match project(src, s, max_dist) {
                Some(h) => (h.pos, hit_normal(src, &h)),
                None => {
                    missed += 1;
                    (s.pos, s.normal)
                }
            },
            None => (s.pos, s.normal),
        };
        let n = n.normalize_or(Vec3::Y);
        let t1 = n.any_orthonormal_vector();
        let t2 = n.cross(t1);
        let origin = pos + n * size * 1e-3;
        let mut hits = 0;
        for i in 0..samples {
            let hsh = hash(0xB0 + i as u64, x, y);
            let (u1, u2) = (((hsh >> 40) as f32) / (1u64 << 24) as f32, ((hsh.rotate_left(24) >> 40) as f32) / (1u64 << 24) as f32);
            let r = u1.sqrt();
            let phi = std::f32::consts::TAU * u2;
            let d = (t1 * (r * phi.cos()) + t2 * (r * phi.sin()) + n * (1.0 - u1).max(0.0).sqrt()).normalize();
            if occluders.any_hit(origin, d, dist) {
                hits += 1;
            }
        }
        let v = ((1.0 - hits as f32 / samples as f32) * 255.0).round() as u8;
        Some([v, v, v, 255])
    });
    BakeReport { changed, missed }
}

/// Signed mean curvature per vertex (convex > 0), from neighbouring normals.
pub fn vertex_curvature(mesh: &Mesh) -> HashMap<VertId, f32> {
    let vf = mesh.vert_faces();
    let normals: HashMap<VertId, Vec3> = mesh.verts.keys().map(|v| (*v, mesh.vert_normal(*v, &vf))).collect();
    let mut sum: HashMap<VertId, (f32, u32)> = HashMap::new();
    let edges: BTreeSet<Edge> = mesh.edges();
    for e in edges {
        let (a, b) = (e.0, e.1);
        let (pa, pb) = (mesh.pos(a), mesh.pos(b));
        let d = pb - pa;
        let l2 = d.length_squared();
        if l2 < 1e-12 {
            continue;
        }
        let k = (normals[&b] - normals[&a]).dot(d) / l2;
        for v in [a, b] {
            let s = sum.entry(v).or_default();
            s.0 += k;
            s.1 += 1;
        }
    }
    sum.into_iter().map(|(v, (k, n))| (v, k / n.max(1) as f32)).collect()
}

/// Which curvature-derived mask to bake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CurvatureMask {
    /// 0.5 = flat, brighter on convex edges, darker in concave creases.
    Curvature,
    /// White on convex edges only (edge wear).
    Convexity,
    /// White in concave creases only (dirt, grime).
    Cavity,
}

/// Curvature-based mask from the source surface (or the target), as grey.
pub fn bake_curvature(tex: &mut Texture, map: &SurfaceMap, target: &Mesh, src: Option<&mut Source>, kind: CurvatureMask, o: &BakeOptions, max_dist: f32, p: &Paint) -> BakeReport {
    let contrast = o.contrast.unwrap_or(1.0).max(0.0);
    let (mesh, curv, src_ref): (&Mesh, HashMap<VertId, f32>, Option<&Source>) = match src {
        Some(src) => {
            if src.curvature.is_none() {
                src.curvature = Some(vertex_curvature(src.mesh));
            }
            let c = src.curvature.clone().unwrap();
            (src.mesh, c, Some(&*src))
        }
        None => (target, vertex_curvature(target), None),
    };
    // Normalise by a typical magnitude (90th percentile), so the mask uses the range.
    let mut mags: Vec<f32> = curv.values().map(|k| k.abs()).filter(|k| k.is_finite()).collect();
    mags.sort_by(f32::total_cmp);
    let scale = mags.get(mags.len() * 9 / 10).copied().unwrap_or(1.0).max(1e-6);
    let mut missed = 0;
    let changed = apply(tex, &map.mask(), p, |x, y, _| {
        let s = map.get(x, y)?;
        let h = match src_ref {
            Some(src) => match project(src, s, max_dist) {
                Some(h) => h,
                None => {
                    missed += 1;
                    self_hit(s)
                }
            },
            None => self_hit(s),
        };
        let f = mesh.faces.get(&h.face)?;
        let k: f32 = h.corners.iter().map(|(c, w)| curv.get(&f.verts[*c]).copied().unwrap_or(0.0) * w).sum::<f32>() / scale * contrast;
        let v = match kind {
            CurvatureMask::Curvature => 0.5 + 0.5 * k.tanh(),
            CurvatureMask::Convexity => k.max(0.0).tanh(),
            CurvatureMask::Cavity => (-k).max(0.0).tanh(),
        };
        let b = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        Some([b, b, b, 255])
    });
    BakeReport { changed, missed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    use polly_core::primitives::{Primitive, plane};

    fn uv_plane() -> Mesh {
        let mut m = plane(Vec2::splat(2.0), 1, 1);
        for f in m.faces.values_mut() {
            let vs = f.verts.clone();
            for (c, v) in f.corners.iter_mut().zip(vs) {
                let p = m.verts[&v].pos;
                c.uv = Vec2::new(p.x * 0.5 + 0.5, p.z * 0.5 + 0.5);
            }
        }
        m
    }

    #[test]
    fn flat_source_gives_flat_normal_map() {
        let low = uv_plane();
        let high = uv_plane();
        let faces = low.faces.keys().copied().collect();
        let map = SurfaceMap::new(&low, &faces, 16, 16, 0);
        let mut tex = Texture::new("n", 16, 16, [0; 4]);
        let tangents = HashMap::new();
        let src = Source::new(&high);
        let r = bake_normal(&mut tex, &map, &low, &tangents, Some(&src), 0.5, &Paint::default());
        assert_eq!(r.missed, 0);
        let px = tex.get(8, 8);
        assert!((px[0] as i32 - 128).abs() <= 1 && (px[1] as i32 - 128).abs() <= 1 && px[2] == 255, "{px:?}");
    }

    #[test]
    fn tilted_source_tilts_the_normals() {
        // Source tilted around Z: its normal leans towards +X, which is the
        // plane's +u (tangent) direction, so red goes up.
        let low = uv_plane();
        let mut high = uv_plane();
        for v in high.verts.values_mut() {
            v.pos.y += v.pos.x * -0.3;
        }
        let faces = low.faces.keys().copied().collect();
        let map = SurfaceMap::new(&low, &faces, 16, 16, 0);
        let mut tex = Texture::new("n", 16, 16, [0; 4]);
        let tangents: HashMap<(FaceId, usize), [f32; 4]> =
            low.faces.iter().flat_map(|(f, face)| (0..face.len()).map(move |i| ((*f, i), [1.0, 0.0, 0.0, 1.0]))).collect();
        let src = Source::new(&high);
        bake_normal(&mut tex, &map, &low, &tangents, Some(&src), 1.0, &Paint::default());
        let px = tex.get(8, 8);
        assert!(px[0] > 160 && (px[1] as i32 - 128).abs() <= 2, "{px:?}");
    }

    #[test]
    fn curvature_marks_convex_and_concave() {
        let sphere = Primitive::UvSphere { radius: 1.0, segments: 16, rings: 8 }.build().unwrap();
        let k = vertex_curvature(&sphere);
        assert!(k.values().all(|c| *c > 0.0), "a sphere is convex everywhere");
        let mut inside = sphere.clone();
        for f in inside.faces.values_mut() {
            f.reverse();
        }
        let k = vertex_curvature(&inside);
        assert!(k.values().all(|c| *c < 0.0), "seen from inside it is concave");
    }

    #[test]
    fn ao_darkens_under_a_cover() {
        let low = uv_plane();
        let mut cover = uv_plane();
        for v in cover.verts.values_mut() {
            v.pos.y += 0.2;
        }
        let mut occ = low.clone();
        occ.append(&cover);
        let faces = low.faces.keys().copied().collect();
        let map = SurfaceMap::new(&low, &faces, 8, 8, 0);
        let mut covered = Texture::new("ao", 8, 8, [0; 4]);
        bake_ao(&mut covered, &map, None, &Bvh::new(&occ, glam::Mat4::IDENTITY), &BakeOptions { max_distance: None, ao_samples: Some(64), ao_distance: Some(1.0), contrast: None }, 0.1, &Paint::default());
        let mut open = Texture::new("ao", 8, 8, [0; 4]);
        bake_ao(&mut open, &map, None, &Bvh::new(&low, glam::Mat4::IDENTITY), &BakeOptions { max_distance: None, ao_samples: Some(64), ao_distance: Some(1.0), contrast: None }, 0.1, &Paint::default());
        assert!(covered.get(4, 4)[0] < 100 && open.get(4, 4)[0] > 250, "{:?} {:?}", covered.get(4, 4), open.get(4, 4));
    }
}
