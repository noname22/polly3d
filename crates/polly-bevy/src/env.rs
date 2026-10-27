//! Image-based lighting: procedural studio / sky environments or an
//! equirectangular .hdr file, prefiltered on the CPU into the diffuse
//! (irradiance) and specular (GGX, one mip per roughness step) cubemaps that
//! Bevy's `EnvironmentMapLight` expects. Deterministic: no GPU filtering pass,
//! so the first render after a change is already lit.

use std::f32::consts::PI;

use anyhow::{Context, Result, bail};
use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::math::Vec3;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension};

/// Specular cubemap face size (mip 0); one mip per halving down to 1 px.
const SPECULAR_SIZE: u32 = 128;
const DIFFUSE_SIZE: u32 = 16;
const SKYBOX_SIZE: u32 = 256;
const SAMPLES: u32 = 96;

/// An environment's radiance as a function of world direction (linear RGB,
/// 1.0 = an output of 1 at polly's unit exposure).
pub enum Source {
    Studio,
    Sky,
    /// Equirectangular image pyramid (finest first), each (w, h, rgb).
    Hdr(Vec<(u32, u32, Vec<[f32; 3]>)>),
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A soft round light of angular radius `r` (radians) around `c`.
fn softbox(d: Vec3, c: Vec3, r: f32) -> f32 {
    let ang = d.dot(c.normalize()).clamp(-1.0, 1.0).acos();
    1.0 - smoothstep(r * 0.6, r, ang)
}

impl Source {
    pub fn load(spec: &str) -> Result<Source> {
        Ok(match spec {
            "studio" => Source::Studio,
            "sky" => Source::Sky,
            path => {
                let img = image::open(path).with_context(|| format!("cannot read environment '{path}'"))?.to_rgb32f();
                let (w, h) = img.dimensions();
                if w < 4 || h < 2 {
                    bail!("environment image '{path}' is too small");
                }
                let mut level: Vec<[f32; 3]> = img.pixels().map(|p| p.0).collect();
                let (mut w, mut h) = (w, h);
                // Shrink very large images first (sampling only needs a few hundred px).
                while w > 1024 {
                    (level, w, h) = halve(&level, w, h);
                }
                let mut pyramid = vec![(w, h, level.clone())];
                while w > 8 && h > 4 {
                    (level, w, h) = halve(&level, w, h);
                    pyramid.push((w, h, level.clone()));
                }
                Source::Hdr(pyramid)
            }
        })
    }

    /// Radiance from world direction `d` (unit), optionally from a blurrier
    /// pyramid level (0 = sharpest) for image sources.
    fn radiance(&self, d: Vec3, level: usize) -> Vec3 {
        match self {
            Source::Studio => {
                // Grey floor, darker walls, brighter ceiling; a large key softbox
                // front-left-above, a rim softbox behind-right, a strip overhead.
                let y = d.y;
                let base = if y < 0.0 {
                    Vec3::splat(0.16).lerp(Vec3::splat(0.30), smoothstep(-0.25, 0.0, y))
                } else {
                    Vec3::splat(0.30).lerp(Vec3::splat(0.55), smoothstep(0.0, 0.8, y))
                };
                let key = softbox(d, Vec3::new(-0.55, 0.65, 0.55), 0.45) * 3.2;
                let rim = softbox(d, Vec3::new(0.7, 0.45, -0.6), 0.35) * 1.6;
                let top = softbox(d, Vec3::new(0.1, 1.0, 0.0), 0.3) * 1.2;
                base + Vec3::splat(key + rim + top)
            }
            Source::Sky => {
                let y = d.y;
                // Sky light only: pair it with a `sun` for direct light and shadows.
                let zenith = Vec3::new(0.22, 0.42, 0.95) * 0.6;
                let horizon = Vec3::new(0.75, 0.84, 1.0) * 0.7;
                let ground = Vec3::new(0.28, 0.25, 0.22) * 0.4;
                let sky = if y >= 0.0 { horizon.lerp(zenith, smoothstep(0.0, 0.6, y).sqrt()) } else { horizon.lerp(ground, smoothstep(-0.12, 0.0, -y)) };
                let sun_dir = Vec3::new(0.4, 0.6, 0.3).normalize();
                // A small disk for reflections, too small to light much.
                let sun = softbox(d, sun_dir, 0.04) * 20.0 + softbox(d, sun_dir, 0.3) * 0.2;
                sky + Vec3::new(1.0, 0.95, 0.85) * sun
            }
            Source::Hdr(pyr) => {
                let (w, h, px) = &pyr[level.min(pyr.len() - 1)];
                // Equirectangular: u from the angle around Y (0 at -Z, increasing towards +X), v from the pole.
                let u = 0.5 + d.x.atan2(-d.z) / (2.0 * PI);
                let v = d.y.clamp(-1.0, 1.0).acos() / PI;
                let fx = u * *w as f32 - 0.5;
                let fy = (v * *h as f32 - 0.5).clamp(0.0, *h as f32 - 1.0);
                let (x0, y0) = (fx.floor(), fy.floor());
                let (tx, ty) = (fx - x0, fy - y0);
                let at = |x: f32, y: f32| {
                    let xi = (x as i64).rem_euclid(*w as i64) as u32;
                    let yi = (y.max(0.0) as u32).min(h - 1);
                    Vec3::from_array(px[(yi * w + xi) as usize])
                };
                let a = at(x0, y0).lerp(at(x0 + 1.0, y0), tx);
                let b = at(x0, y0 + 1.0).lerp(at(x0 + 1.0, y0 + 1.0), tx);
                a.lerp(b, ty)
            }
        }
    }

    fn levels(&self) -> usize {
        match self {
            Source::Hdr(p) => p.len(),
            _ => 1,
        }
    }
}

fn halve(px: &[[f32; 3]], w: u32, h: u32) -> (Vec<[f32; 3]>, u32, u32) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = Vec::with_capacity((nw * nh) as usize);
    for y in 0..nh {
        for x in 0..nw {
            let mut s = [0.0; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = px[(((y * 2 + dy).min(h - 1)) * w + (x * 2 + dx).min(w - 1)) as usize];
                for i in 0..3 {
                    s[i] += p[i] * 0.25;
                }
            }
            out.push(s);
        }
    }
    (out, nw, nh)
}

/// World direction of cube face `face` at face coordinates (u, v) in -1..1
/// (u right, v down). Bevy samples cubemaps with z negated (left-handed).
fn cube_dir(face: u32, u: f32, v: f32) -> Vec3 {
    let d = match face {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        _ => Vec3::new(-u, -v, -1.0),
    }
    .normalize();
    Vec3::new(d.x, d.y, -d.z)
}

fn hammersley(i: u32, n: u32) -> (f32, f32) {
    (i as f32 / n as f32, i.reverse_bits() as f32 * 2.328_306_4e-10)
}

/// GGX-prefiltered radiance around `n` for roughness `alpha` (= perceptual²).
fn prefilter(src: &Source, n: Vec3, alpha: f32, level: usize) -> Vec3 {
    let up = if n.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
    let tx = up.cross(n).normalize();
    let ty = n.cross(tx);
    let mut sum = Vec3::ZERO;
    let mut weight = 0.0;
    let a2 = alpha * alpha;
    for i in 0..SAMPLES {
        let (e1, e2) = hammersley(i, SAMPLES);
        let phi = 2.0 * PI * e1;
        let cos_t = ((1.0 - e2) / (1.0 + (a2 - 1.0) * e2)).sqrt();
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        let h = tx * (sin_t * phi.cos()) + ty * (sin_t * phi.sin()) + n * cos_t;
        let l = (2.0 * n.dot(h)) * h - n;
        let nl = n.dot(l);
        if nl > 0.0 {
            sum += src.radiance(l, level) * nl;
            weight += nl;
        }
    }
    if weight > 0.0 { sum / weight } else { src.radiance(n, level) }
}

fn f16_bits(x: f32) -> u16 {
    let x = if x.is_finite() { x.clamp(0.0, 65000.0) } else { 0.0 };
    let b = x.to_bits();
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if x == 0.0 || exp <= 0 {
        // Flush tiny values (and subnormals) to zero.
        return 0;
    }
    if exp >= 31 {
        return 0x7bff;
    }
    // Round to nearest.
    let m = (man + 0x1000) >> 13;
    if m == 0x400 {
        return ((exp as u16 + 1) << 10).min(0x7bff);
    }
    ((exp as u16) << 10) | m as u16
}

fn push_texel(out: &mut Vec<u8>, c: Vec3) {
    for v in [c.x, c.y, c.z, 1.0] {
        out.extend(f16_bits(v).to_le_bytes());
    }
}

fn cube_image(size: u32, mips: u32, data: Vec<u8>) -> Image {
    let mut img = Image::new_uninit(
        Extent3d { width: size, height: size, depth_or_array_layers: 6 },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    // Layer-major (each face with all its mips), Bevy's default data order.
    img.data = Some(data);
    img.texture_descriptor.mip_level_count = mips;
    img.texture_view_descriptor = Some(TextureViewDescriptor { dimension: Some(TextureViewDimension::Cube), ..Default::default() });
    img
}

/// Texel centre in -1..1 face coordinates.
fn coord(i: u32, size: u32) -> f32 {
    (i as f32 + 0.5) / size as f32 * 2.0 - 1.0
}

/// Diffuse irradiance cubemap (radiance-weighted cosine average, i.e. E/π),
/// via 9 spherical-harmonic coefficients.
pub fn diffuse(src: &Source) -> Image {
    let n = 32u32;
    let mut sh = [Vec3::ZERO; 9];
    let mut total = 0.0;
    let level = src.levels().saturating_sub(3);
    for face in 0..6 {
        for y in 0..n {
            for x in 0..n {
                let (u, v) = (coord(x, n), coord(y, n));
                let d = cube_dir(face, u, v);
                // Solid angle of the texel.
                let w = 1.0 / (1.0 + u * u + v * v).powf(1.5);
                let l = src.radiance(d, level) * w;
                let b = sh_basis(d);
                for i in 0..9 {
                    sh[i] += l * b[i];
                }
                total += w;
            }
        }
    }
    let norm = 4.0 * PI / total;
    for c in &mut sh {
        *c *= norm;
    }
    // Cosine-lobe convolution, divided by π.
    let a = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
    let mut data = Vec::new();
    let s = DIFFUSE_SIZE;
    for face in 0..6 {
        for y in 0..s {
            for x in 0..s {
                let d = cube_dir(face, coord(x, s), coord(y, s));
                let b = sh_basis(d);
                let mut e = Vec3::ZERO;
                for i in 0..9 {
                    e += sh[i] * (a[i] * b[i]);
                }
                push_texel(&mut data, e.max(Vec3::ZERO));
            }
        }
    }
    cube_image(s, 1, data)
}

fn sh_basis(d: Vec3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

/// Specular cubemap with one GGX-prefiltered mip per roughness step
/// (Bevy picks mip = perceptual roughness x (mips - 1)).
pub fn specular(src: &Source) -> Image {
    let mips = SPECULAR_SIZE.trailing_zeros() + 1;
    let mut data = Vec::new();
    // wgpu layout: for each layer, all mips (layer-major).
    for face in 0..6 {
        for mip in 0..mips {
            let s = SPECULAR_SIZE >> mip;
            let rough = mip as f32 / (mips - 1) as f32;
            let level = ((rough * src.levels() as f32 * 1.2) as usize).min(src.levels() - 1);
            for y in 0..s {
                for x in 0..s {
                    let d = cube_dir(face, coord(x, s), coord(y, s));
                    let c = if mip == 0 { src.radiance(d, 0) } else { prefilter(src, d, (rough * rough).max(1e-3), level) };
                    push_texel(&mut data, c);
                }
            }
        }
    }
    cube_image(SPECULAR_SIZE, mips, data)
}

/// A sharp cubemap for showing the environment as the background.
pub fn skybox(src: &Source) -> Image {
    let s = SKYBOX_SIZE;
    let mut data = Vec::new();
    for face in 0..6 {
        for y in 0..s {
            for x in 0..s {
                push_texel(&mut data, src.radiance(cube_dir(face, coord(x, s), coord(y, s)), 0));
            }
        }
    }
    cube_image(s, 1, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats() {
        for (x, bits) in [(1.0f32, 0x3c00u16), (0.5, 0x3800), (2.0, 0x4000), (0.0, 0)] {
            assert_eq!(f16_bits(x), bits, "{x}");
        }
        assert!(f16_bits(1e9) >= 0x7be0 && f16_bits(1e9) <= 0x7bff);
    }

    #[test]
    fn cube_faces_cover_the_axes() {
        // Face centres point along +X, -X, +Y, -Y and (z negated for Bevy) -Z, +Z.
        let want = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, -Vec3::Z, Vec3::Z];
        for (f, w) in want.iter().enumerate() {
            assert!(cube_dir(f as u32, 0.0, 0.0).dot(*w) > 0.999, "face {f}");
        }
    }

    #[test]
    fn loads_equirectangular_hdr() {
        // Top half blue sky (2.0), bottom half dark ground: up-facing irradiance is bluer than down-facing.
        let (w, h) = (16u32, 8u32);
        let px: Vec<image::Rgb<f32>> =
            (0..w * h).map(|i| if i / w < h / 2 { image::Rgb([0.5, 0.8, 2.0]) } else { image::Rgb([0.1, 0.1, 0.1]) }).collect();
        let dir = std::env::temp_dir().join(format!("polly-env-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.hdr");
        let f = std::fs::File::create(&path).unwrap();
        image::codecs::hdr::HdrEncoder::new(f).encode(&px, w as usize, h as usize).unwrap();
        let src = Source::load(path.to_str().unwrap()).unwrap();
        let up = src.radiance(Vec3::Y, 0);
        let down = src.radiance(-Vec3::Y, 0);
        assert!(up.z > 1.5 && down.z < 0.2, "{up:?} {down:?}");
        assert!(Source::load("/nope.hdr").is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn constant_environment_gives_unit_irradiance() {
        // A uniform environment of radiance 1: irradiance / π = 1 everywhere.
        let src = Source::Hdr(vec![(8, 4, vec![[1.0; 3]; 32])]);
        let img = diffuse(&src);
        let data = img.data.unwrap();
        let first = u16::from_le_bytes([data[0], data[1]]);
        assert!((first as i32 - 0x3c00).abs() < 40, "{first:#x}");
    }
}
