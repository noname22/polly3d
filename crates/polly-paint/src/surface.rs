//! Painting in 3-D surface space: each covered texel knows the point on the
//! model it shows. Effects computed from those points (noise, distance-based
//! stamps, ambient occlusion) are continuous across UV seams.

use std::collections::BTreeSet;

use glam::{Vec3, Vec4};
use polly_core::bvh::Bvh;
use polly_core::mesh::{FaceId, Mesh};
use polly_core::scene::Texture;
use serde::{Deserialize, Serialize};

use crate::canvas::{Dither, Paint, apply, hash};
use crate::mask::NEIGHBORS8;
use crate::color::Rgba;
use crate::faces::raster_poly;
use crate::mask::Mask;

/// The surface point (object space), shading normal and face behind a texel.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub pos: Vec3,
    pub normal: Vec3,
    pub face: FaceId,
    /// Face corners and their weights at this texel.
    pub corners: [(usize, f32); 3],
}

/// Surface samples for every texel covered by the faces' UVs.
pub struct SurfaceMap {
    pub width: u32,
    pub height: u32,
    cells: Vec<Option<Sample>>,
}

impl SurfaceMap {
    /// Rasterize `faces` into a `w` x `h` texture. `grow_px` extends coverage
    /// outward (copying neighbouring samples) to hide seams with nearest filtering.
    pub fn new(mesh: &Mesh, faces: &BTreeSet<FaceId>, w: u32, h: u32, grow_px: u32) -> Self {
        let normals = mesh.corner_normals();
        let mut cells: Vec<Option<Sample>> = vec![None; (w * h) as usize];
        for &fid in faces {
            let Some(f) = mesh.faces.get(&fid) else { continue };
            let poly: Vec<[f32; 2]> = f.corners.iter().map(|c| c.uv.to_array()).collect();
            let pts = mesh.face_positions(f);
            raster_poly(&poly, w, h, |x, y, wts| {
                let cell = &mut cells[(y as u32 * w + x as u32) as usize];
                if cell.is_none() {
                    let pos = wts.iter().map(|&(i, k)| pts[i] * k).sum();
                    let normal = wts.iter().map(|&(i, k)| normals[&(fid, i)] * k).sum::<Vec3>().normalize_or_zero();
                    *cell = Some(Sample { pos, normal, face: fid, corners: *wts });
                }
            });
        }
        for _ in 0..grow_px {
            let prev = cells.clone();
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let i = (y as u32 * w + x as u32) as usize;
                    if prev[i].is_some() {
                        continue;
                    }
                    cells[i] = NEIGHBORS8
                        .iter()
                        .map(|(dx, dy)| (x + dx, y + dy))
                        .filter(|&(nx, ny)| nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h)
                        .find_map(|(nx, ny)| prev[(ny as u32 * w + nx as u32) as usize]);
                }
            }
        }
        SurfaceMap { width: w, height: h, cells }
    }

    pub fn get(&self, x: i32, y: i32) -> Option<&Sample> {
        if x < 0 || y < 0 || x as u32 >= self.width || y as u32 >= self.height {
            return None;
        }
        self.cells[(y as u32 * self.width + x as u32) as usize].as_ref()
    }

    pub fn mask(&self) -> Mask {
        let mut m = Mask::empty(self.width, self.height);
        m.bits = self.cells.iter().map(Option::is_some).collect();
        m
    }

    pub fn samples(&self) -> impl Iterator<Item = (i32, i32, &Sample)> {
        let w = self.width;
        self.cells.iter().enumerate().filter_map(move |(i, c)| c.as_ref().map(|s| ((i as u32 % w) as i32, (i as u32 / w) as i32, s)))
    }

    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        self.samples().fold(None, |acc, (_, _, s)| {
            Some(match acc {
                None => (s.pos, s.pos),
                Some((lo, hi)) => (lo.min(s.pos), hi.max(s.pos)),
            })
        })
    }
}

fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

fn hash3(seed: u64, x: i32, y: i32, z: i32) -> f32 {
    unit(hash(hash(seed, x, y), z, 0x5bd1))
}

/// Smooth 3-D value noise in [0, 1].
pub fn value_noise3(seed: u64, p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let s = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let (x, y, z) = (i.x as i32, i.y as i32, i.z as i32);
    let c = |dx, dy, dz| hash3(seed, x + dx, y + dy, z + dz);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), s.x);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), s.x);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), s.x);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), s.x);
    lerp(lerp(x00, x10, s.y), lerp(x01, x11, s.y), s.z)
}

/// Fractal (octave-summed) value noise in [0, 1].
pub fn fbm3(seed: u64, p: Vec3, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut freq) = (0.0, 1.0, 0.0, 1.0);
    for o in 0..octaves.max(1) {
        sum += value_noise3(seed.wrapping_add(o as u64 * 7919), p * freq) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

/// Options for [`noise3d`].
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Noise3d {
    /// Colours from dark/low to light/high; the noise picks bands among them
    /// (or blends between them with `smooth`). Hard bands of clearly different
    /// colours read as camouflage; for subtle fabric or skin mottling use close
    /// shades, `smooth: true` and contrast below 1.
    pub colors: Vec<Rgba>,
    /// Feature size in world units (e.g. 0.1 = blotches about 10 cm across).
    pub scale: f32,
    /// Detail layers (1 = soft blobs, 3-4 = grainy); default 2.
    #[serde(default = "two")]
    pub octaves: u32,
    /// >1 pushes values towards the extremes (more of the first/last colour), <1 evens them out. Default 1.
    #[serde(default = "onef")]
    pub contrast: f32,
    #[serde(default)]
    pub seed: u64,
    /// Ordered dithering at band boundaries (retro), default none.
    #[serde(default)]
    pub dither: Dither,
    /// Blend smoothly between neighbouring colours instead of hard bands (default false).
    #[serde(default)]
    pub smooth: bool,
}

fn two() -> u32 {
    2
}
fn onef() -> f32 {
    1.0
}

/// Fill the mapped texels with noise sampled at their 3-D surface points, so the
/// pattern is continuous across UV seams.
pub fn noise3d(tex: &mut Texture, map: &SurfaceMap, n: &Noise3d, p: &Paint) -> usize {
    if n.colors.is_empty() {
        return 0;
    }
    let inv = 1.0 / n.scale.max(1e-4);
    let k = n.colors.len() as f32;
    apply(tex, &map.mask(), p, |x, y, _| {
        let s = map.get(x, y)?;
        let v = fbm3(n.seed, s.pos * inv, n.octaves);
        let v = (0.5 + (v - 0.5) * n.contrast).clamp(0.0, 0.9999);
        if n.smooth {
            let t = v * (k - 1.0);
            let (i, f) = (t.floor() as usize, t.fract());
            let (a, b) = (n.colors[i.min(n.colors.len() - 1)], n.colors[(i + 1).min(n.colors.len() - 1)]);
            return Some(std::array::from_fn(|c| (a[c] as f32 + (b[c] as f32 - a[c] as f32) * f).round() as u8));
        }
        let t = v * k;
        let (i, frac) = (t.floor(), t.fract());
        let mut idx = i as usize;
        // Dither towards the neighbouring band near boundaries.
        if n.dither != Dither::None {
            let d = (frac - 0.5).abs() * 2.0; // 0 at band centre .. 1 at boundary
            if d > 0.5 && n.dither.threshold(x, y) < (d - 0.5) {
                idx = if frac > 0.5 { (idx + 1).min(n.colors.len() - 1) } else { idx.saturating_sub(1) };
            }
        }
        Some(n.colors[idx.min(n.colors.len() - 1)])
    })
}

/// Ambient occlusion settings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Ao {
    /// Rays per texel (default 24).
    #[serde(default = "rays")]
    pub samples: u32,
    /// How far occluders count, in world units (default: 30% of the faces' size).
    pub distance: Option<f32>,
    /// 0..1, how dark fully occluded areas get (default 0.6).
    #[serde(default = "strength")]
    pub strength: f32,
}

/// Height-based darkening (e.g. darker towards the feet / underside).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HeightShade {
    /// Direction that gets lighter (default up, [0,1,0]).
    #[serde(default = "up")]
    pub axis: [f32; 3],
    /// 0..1, darkening at the low end (default 0.4).
    #[serde(default = "hstrength")]
    pub strength: f32,
}

fn rays() -> u32 {
    24
}
fn strength() -> f32 {
    0.6
}
fn hstrength() -> f32 {
    0.4
}
fn up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Shading {
    pub ao: Option<Ao>,
    pub height: Option<HeightShade>,
    /// With a palette-locked texture, dither between palette colours instead of snapping.
    #[serde(default)]
    pub dither: Dither,
}

/// Darken the existing texels by ambient occlusion and/or height. `bvh` holds
/// the occluding geometry in the same space as the map's positions.
pub fn bake_shading(tex: &mut Texture, map: &SurfaceMap, bvh: &Bvh, o: &Shading, p: &Paint) -> usize {
    let Some((lo, hi)) = map.bounds() else { return 0 };
    let size = (hi - lo).length().max(1e-4);
    let axis = o.height.map(|h| Vec3::from(h.axis).normalize_or(Vec3::Y));
    let (amin, amax) = match axis {
        Some(a) => map.samples().fold((f32::MAX, f32::MIN), |(mn, mx), (_, _, s)| (mn.min(s.pos.dot(a)), mx.max(s.pos.dot(a)))),
        None => (0.0, 1.0),
    };
    let palette = p.palette.clone();
    apply(tex, &map.mask(), p, |x, y, cur| {
        let s = map.get(x, y)?;
        let mut k = 1.0;
        if let (Some(h), Some(a)) = (o.height, axis) {
            let t = ((s.pos.dot(a) - amin) / (amax - amin).max(1e-6)).clamp(0.0, 1.0);
            k *= 1.0 - h.strength.clamp(0.0, 1.0) * (1.0 - t);
        }
        if let Some(ao) = o.ao {
            let dist = ao.distance.unwrap_or(size * 0.3);
            let n = s.normal;
            let helper = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
            let t1 = n.cross(helper).normalize();
            let t2 = n.cross(t1);
            let origin = s.pos + n * size * 1e-3;
            let samples = ao.samples.clamp(1, 256);
            let mut hits = 0;
            for i in 0..samples {
                // Cosine-weighted hemisphere, decorrelated per texel.
                let hsh = hash(0xA0 + i as u64, x, y);
                let (u1, u2) = (unit(hsh), unit(hsh.rotate_left(24)));
                let r = u1.sqrt();
                let phi = std::f32::consts::TAU * u2;
                let d = (t1 * (r * phi.cos()) + t2 * (r * phi.sin()) + n * (1.0 - u1).sqrt()).normalize();
                if bvh.any_hit(origin, d, dist) {
                    hits += 1;
                }
            }
            k *= 1.0 - ao.strength.clamp(0.0, 1.0) * hits as f32 / samples as f32;
        }
        let c = Vec4::new(cur[0] as f32, cur[1] as f32, cur[2] as f32, 255.0);
        let shaded: Rgba = [(c.x * k).round() as u8, (c.y * k).round() as u8, (c.z * k).round() as u8, cur[3]];
        Some(match (&palette, o.dither) {
            (Some(pal), d) if d != Dither::None => crate::quantize::dithered(pal, shaded, d, x as usize, y as usize),
            _ => shaded,
        })
    })
}

/// A disc or ring painted by 3-D distance from `center` (sized in world
/// units, continuous across seams). Only texels on surfaces facing roughly
/// along `normal` are painted, so thin parts don't get stamped on the back.
#[allow(clippy::too_many_arguments)]
pub fn stamp(
    tex: &mut Texture,
    map: &SurfaceMap,
    center: Vec3,
    normal: Vec3,
    radius: f32,
    inner_radius: f32,
    color: Rgba,
    outline: Option<(Rgba, f32)>,
    p: &Paint,
) -> usize {
    let n = normal.normalize_or_zero();
    apply(tex, &map.mask(), p, |x, y, _| {
        let s = map.get(x, y)?;
        if n != Vec3::ZERO && s.normal.dot(n) < 0.2 {
            return None;
        }
        let d = s.pos.distance(center);
        if d > radius || d < inner_radius {
            return None;
        }
        match outline {
            Some((oc, w)) if d > radius - w || (inner_radius > 0.0 && d < inner_radius + w) => Some(oc),
            _ => Some(color),
        }
    })
}

/// Dashes along a [`surface_line`]: `dash` painted, then `gap` skipped (world units).
#[derive(Debug, Clone, Copy)]
pub struct Dashes {
    pub dash: f32,
    pub gap: f32,
}

/// A line painted by 3-D distance from a polyline lying on the surface
/// (`points` with their surface `normals`), `width` world units wide and
/// continuous across seams. Only texels facing roughly like the nearest line
/// point are painted, so the line doesn't bleed through thin parts.
#[allow(clippy::too_many_arguments)]
pub fn surface_line(
    tex: &mut Texture,
    map: &SurfaceMap,
    points: &[Vec3],
    normals: &[Vec3],
    width: f32,
    color: Rgba,
    outline: Option<(Rgba, f32)>,
    dashes: Option<Dashes>,
    p: &Paint,
) -> usize {
    if points.len() < 2 || points.len() != normals.len() {
        return 0;
    }
    let r = width * 0.5;
    // Arc length at each point (for dashes).
    let mut arc = vec![0.0f32];
    for w in points.windows(2) {
        arc.push(arc.last().unwrap() + w[0].distance(w[1]));
    }
    let (lo, hi) = points.iter().fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let (lo, hi) = (lo - Vec3::splat(r), hi + Vec3::splat(r));
    apply(tex, &map.mask(), p, |x, y, _| {
        let s = map.get(x, y)?;
        if s.pos.cmplt(lo).any() || s.pos.cmpgt(hi).any() {
            return None;
        }
        // Nearest point on the polyline: (distance, arc length, segment).
        let mut best = (f32::MAX, 0.0, 0);
        for (i, w) in points.windows(2).enumerate() {
            let d = w[1] - w[0];
            let t = ((s.pos - w[0]).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0);
            let dist = s.pos.distance(w[0] + d * t);
            if dist < best.0 {
                best = (dist, arc[i] + (arc[i + 1] - arc[i]) * t, i);
            }
        }
        let (dist, along, seg) = best;
        if dist > r {
            return None;
        }
        let n = (normals[seg] + normals[seg + 1]).normalize_or_zero();
        if n != Vec3::ZERO && s.normal.dot(n) < 0.2 {
            return None;
        }
        if let Some(d) = dashes
            && along.rem_euclid((d.dash + d.gap).max(1e-6)) > d.dash
        {
            return None;
        }
        match outline {
            Some((oc, w)) if dist > r - w => Some(oc),
            _ => Some(color),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polly_core::primitives::cube;

    /// Cube with each face mapped to its own cell of a 3x2 atlas.
    fn atlas_cube() -> Mesh {
        let mut m = cube(Vec3::ONE);
        for (i, f) in m.faces.values_mut().enumerate() {
            let (cx, cy) = ((i % 3) as f32 / 3.0, (i / 3) as f32 / 2.0);
            let quad = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            for (c, q) in f.corners.iter_mut().zip(quad) {
                c.uv = glam::Vec2::new(cx + q[0] / 3.0, cy + q[1] / 2.0);
            }
        }
        m
    }

    #[test]
    fn lines_are_measured_in_world_units_with_dashes() {
        let m = atlas_cube();
        let all: BTreeSet<_> = m.faces.keys().copied().collect();
        let map = SurfaceMap::new(&m, &all, 96, 64, 0);
        // Across the +Z face (z = 0.5), 0.2 wide.
        let pts = [Vec3::new(-0.5, 0.0, 0.5), Vec3::new(0.5, 0.0, 0.5)];
        let nrm = [Vec3::Z, Vec3::Z];
        let mut t = Texture::new("t", 96, 64, [0, 0, 0, 255]);
        let solid = surface_line(&mut t, &map, &pts, &nrm, 0.2, [255, 0, 0, 255], None, None, &Paint::default());
        // The face is 32x32 texels per unit: about 32 x 6.4 texels.
        assert!((150..260).contains(&solid), "{solid}");
        // Nothing on the back (-Z) face even though it is within reach in y.
        let back_texels = (0..t.height).flat_map(|y| (0..t.width).map(move |x| (x, y))).filter(|(x, y)| {
            map.get(*x as i32, *y as i32).is_some_and(|s| s.normal.z < -0.5) && t.get(*x, *y)[0] == 255
        });
        assert_eq!(back_texels.count(), 0);
        let mut t = Texture::new("t", 96, 64, [0, 0, 0, 255]);
        let dashed = surface_line(&mut t, &map, &pts, &nrm, 0.2, [255, 0, 0, 255], None, Some(Dashes { dash: 0.1, gap: 0.1 }), &Paint::default());
        assert!(dashed > solid / 3 && dashed < solid * 2 / 3, "{dashed} of {solid}");
    }

    #[test]
    fn smooth_noise_blends_between_colours() {
        let m = atlas_cube();
        let all: BTreeSet<_> = m.faces.keys().copied().collect();
        let map = SurfaceMap::new(&m, &all, 48, 32, 0);
        let colours = |smooth: bool| {
            let mut t = Texture::new("t", 48, 32, [0, 0, 0, 255]);
            let n = Noise3d { colors: vec![[0, 0, 0, 255], [200, 200, 200, 255]], scale: 0.3, octaves: 2, contrast: 1.0, seed: 1, dither: Dither::None, smooth };
            noise3d(&mut t, &map, &n, &Paint::default());
            t.pixels.chunks(4).map(|c| c[0]).collect::<BTreeSet<u8>>()
        };
        assert_eq!(colours(false), BTreeSet::from([0, 200]), "bands: only the given colours");
        assert!(colours(true).len() > 10, "smooth: shades in between");
    }

    #[test]
    fn samples_lie_on_the_surface() {
        let m = atlas_cube();
        let all: BTreeSet<_> = m.faces.keys().copied().collect();
        let map = SurfaceMap::new(&m, &all, 48, 32, 0);
        let n = map.samples().count();
        assert!(n > 48 * 32 * 9 / 10, "{n}");
        for (_, _, s) in map.samples() {
            assert!((s.pos.abs().max_element() - 0.5).abs() < 1e-4, "{:?} not on a cube face", s.pos);
            assert!(s.normal.dot(s.pos) > 0.0);
        }
    }

    #[test]
    fn stamp_is_sized_in_world_units_and_crosses_seams() {
        let m = atlas_cube();
        let all: BTreeSet<_> = m.faces.keys().copied().collect();
        let map = SurfaceMap::new(&m, &all, 96, 64, 0);
        let mut t = Texture::new("t", 96, 64, [0, 0, 0, 255]);
        // At a cube edge midpoint: the disc must spill onto both faces.
        let c = Vec3::new(0.5, 0.0, 0.5);
        stamp(&mut t, &map, c, Vec3::ZERO, 0.2, 0.0, [255, 0, 0, 255], None, &Paint::new([0; 4]));
        let painted: BTreeSet<FaceId> = map.samples().filter(|(x, y, _)| t.get(*x as u32, *y as u32)[0] == 255).map(|(_, _, s)| s.face).collect();
        assert_eq!(painted.len(), 2, "the stamp should cover both faces at the edge");
        for (x, y, s) in map.samples() {
            let red = t.get(x as u32, y as u32)[0] == 255;
            assert_eq!(red, s.pos.distance(c) <= 0.2, "texel at {:?}", s.pos);
        }
    }

    #[test]
    fn noise_is_continuous_across_seams() {
        // Two texels on different faces that touch in 3-D get nearly the same value.
        let a = fbm3(7, Vec3::new(0.5, 0.1, 0.4999) * 8.0, 2);
        let b = fbm3(7, Vec3::new(0.4999, 0.1, 0.5) * 8.0, 2);
        assert!((a - b).abs() < 0.01);
    }

    #[test]
    fn ao_darkens_inside_corners_only() {
        // An L of two plates: texels near the inner corner get occluded.
        let mut m = polly_core::primitives::plane(glam::Vec2::ONE, 1, 1);
        let mut wall = polly_core::primitives::plane(glam::Vec2::ONE, 1, 1);
        wall.transform(glam::Mat4::from_translation(Vec3::new(0.0, 0.5, -0.5)) * glam::Mat4::from_rotation_x(90f32.to_radians()));
        m.append(&wall);
        for f in m.faces.values_mut().take(1) {
            for (c, q) in f.corners.iter_mut().zip([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]) {
                c.uv = glam::Vec2::from(q);
            }
        }
        let floor: BTreeSet<_> = m.faces.keys().copied().take(1).collect();
        let map = SurfaceMap::new(&m, &floor, 16, 16, 0);
        let bvh = Bvh::new(&m, glam::Mat4::IDENTITY);
        let mut t = Texture::new("t", 16, 16, [200, 200, 200, 255]);
        let o = Shading { ao: Some(Ao { samples: 64, distance: Some(0.6), strength: 1.0 }), height: None, dither: Dither::None };
        bake_shading(&mut t, &map, &bvh, &o, &Paint::new([0; 4]));
        let (mut near, mut far) = (vec![], vec![]);
        for (x, y, s) in map.samples() {
            let v = t.get(x as u32, y as u32)[0];
            if s.pos.z < -0.4 { near.push(v) } else if s.pos.z > 0.4 { far.push(v) }
        }
        let avg = |v: &[u8]| v.iter().map(|x| *x as f32).sum::<f32>() / v.len() as f32;
        assert!(avg(&near) < avg(&far) - 30.0, "near wall {} vs open {}", avg(&near), avg(&far));
    }
}
