//! Pixel-space drawing: crisp primitives, fills, gradients, dithering and noise.
//!
//! Coordinates are integer pixels, (0,0) top-left. Everything clips silently to
//! the texture. Every shape is first rasterized to a coverage [`Mask`] and then
//! written once, so `Alpha`/`Multiply` blending never double-applies on overlap.
//! Drawing functions return the number of pixels whose value actually changed.

use std::collections::BTreeSet;

use polly_core::scene::Texture;
use serde::{Deserialize, Serialize};

use crate::color::{Rgba, from_hsv, lerp, to_hsv};
use crate::mask::{Mask, NEIGHBORS4, Region};
use crate::palette::nearest;

/// How a source colour combines with the existing pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Blend {
    /// Overwrite (including alpha).
    #[default]
    Replace,
    /// Source-over alpha compositing.
    Alpha,
    /// Multiply RGB (alpha of the source controls strength).
    Multiply,
}

/// Ordered-dither pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Dither {
    #[default]
    None,
    Bayer2,
    Bayer4,
    Bayer8,
}

impl Dither {
    /// Threshold in (0, 1) for pixel `(x, y)`; `None` is a flat 0.5.
    pub fn threshold(self, x: i32, y: i32) -> f32 {
        let k = match self {
            Dither::None => return 0.5,
            Dither::Bayer2 => 1,
            Dither::Bayer4 => 2,
            Dither::Bayer8 => 3,
        };
        let (x, y) = (x.rem_euclid(1 << k) as u32, y.rem_euclid(1 << k) as u32);
        // M_2n = 4 * M_n + M_2: the lowest coordinate bits carry the most weight.
        let v = (0..k).fold(0u32, |v, i| {
            let (xb, yb) = ((x >> i) & 1, (y >> i) & 1);
            v * 4 + 2 * (xb ^ yb) + yb
        });
        (v as f32 + 0.5) / (1u32 << (2 * k)) as f32
    }
}

/// Brush settings shared by all drawing operations.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Paint {
    pub color: Rgba,
    #[serde(default)]
    pub blend: Blend,
    /// If set, every written pixel is snapped to the nearest palette colour.
    #[serde(default)]
    pub palette: Option<Vec<Rgba>>,
    /// If set, writes outside the mask are discarded (e.g. keep strokes inside faces).
    #[serde(skip)]
    #[schemars(skip)]
    pub mask: Option<Mask>,
}

impl Default for Paint {
    fn default() -> Self {
        Paint::new([255, 255, 255, 255])
    }
}

impl Paint {
    /// Replace-blend paint with no palette lock or mask.
    pub fn new(color: Rgba) -> Self {
        Paint { color, blend: Blend::Replace, palette: None, mask: None }
    }
    pub fn with_blend(mut self, blend: Blend) -> Self {
        self.blend = blend;
        self
    }
    pub fn with_palette(mut self, palette: Option<Vec<Rgba>>) -> Self {
        self.palette = palette;
        self
    }
    pub fn with_mask(mut self, mask: Option<Mask>) -> Self {
        self.mask = mask;
        self
    }
}

/// Combine `src` onto `dst`.
pub fn blend(dst: Rgba, src: Rgba, mode: Blend) -> Rgba {
    match mode {
        Blend::Replace => src,
        Blend::Alpha => {
            let sa = src[3] as f32 / 255.0;
            let da = dst[3] as f32 / 255.0;
            let oa = sa + da * (1.0 - sa);
            if oa <= 0.0 {
                return [0, 0, 0, 0];
            }
            let ch = |i: usize| ((src[i] as f32 * sa + dst[i] as f32 * da * (1.0 - sa)) / oa).round() as u8;
            [ch(0), ch(1), ch(2), (oa * 255.0).round() as u8]
        }
        Blend::Multiply => {
            let mul: Rgba = std::array::from_fn(|i| if i == 3 { dst[3] } else { ((src[i] as u32 * dst[i] as u32 + 127) / 255) as u8 });
            lerp(dst, mul, src[3] as f32 / 255.0)
        }
    }
}

pub(crate) fn pixel(tex: &Texture, x: i32, y: i32) -> Option<Rgba> {
    if x >= 0 && y >= 0 && (x as u32) < tex.width && (y as u32) < tex.height {
        Some(tex.get(x as u32, y as u32))
    } else {
        None
    }
}

/// Write `f(x, y, current)` to every texel of `cov` (and `p.mask`), with
/// blending and palette snapping. Returns changed-pixel count.
pub(crate) fn apply(tex: &mut Texture, cov: &Mask, p: &Paint, mut f: impl FnMut(i32, i32, Rgba) -> Option<Rgba>) -> usize {
    let w = tex.width;
    let coords: Vec<(i32, i32)> = cov
        .iter()
        .filter(|&(x, y)| (x as u32) < w && (y as u32) < tex.height && p.mask.as_ref().is_none_or(|m| m.get(x, y)))
        .collect();
    let px = tex.pixels_mut();
    let mut changed = 0;
    for (x, y) in coords {
        let i = ((y as u32 * w + x as u32) * 4) as usize;
        let dst: Rgba = px[i..i + 4].try_into().unwrap();
        let Some(src) = f(x, y, dst) else { continue };
        let mut out = blend(dst, src, p.blend);
        if let Some(pal) = &p.palette {
            out = nearest(pal, out).1;
        }
        if out != dst {
            px[i..i + 4].copy_from_slice(&out);
            changed += 1;
        }
    }
    changed
}

fn solid(tex: &mut Texture, cov: &Mask, p: &Paint) -> usize {
    apply(tex, cov, p, |_, _, _| Some(p.color))
}

/// Write individual pixels (out-of-bounds ones are skipped). Returns changed count.
pub fn set_pixels(tex: &mut Texture, px: &[(i32, i32, Rgba)], mode: Blend) -> usize {
    let mut changed = 0;
    let w = tex.width;
    let h = tex.height;
    let buf = tex.pixels_mut();
    for &(x, y, c) in px {
        if x < 0 || y < 0 || x as u32 >= w || y as u32 >= h {
            continue;
        }
        let i = ((y as u32 * w + x as u32) * 4) as usize;
        let dst: Rgba = buf[i..i + 4].try_into().unwrap();
        let out = blend(dst, c, mode);
        if out != dst {
            buf[i..i + 4].copy_from_slice(&out);
            changed += 1;
        }
    }
    changed
}

/// Bresenham cells from `a` to `b` inclusive.
pub(crate) fn bresenham(a: [i32; 2], b: [i32; 2], mut plot: impl FnMut(i32, i32)) {
    let (mut x, mut y) = (a[0], a[1]);
    let (dx, dy) = ((b[0] - x).abs(), -(b[1] - y).abs());
    let (sx, sy) = ((b[0] - x).signum(), (b[1] - y).signum());
    let mut err = dx + dy;
    loop {
        plot(x, y);
        if x == b[0] && y == b[1] {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn line_mask(m: &mut Mask, a: [i32; 2], b: [i32; 2], width: u32) {
    let w = width.max(1) as i32;
    let (lo, hi) = (-(w - 1) / 2, w / 2);
    bresenham(a, b, |x, y| {
        for oy in lo..=hi {
            for ox in lo..=hi {
                m.set(x + ox, y + oy, true);
            }
        }
    });
}

/// Crisp (non-antialiased) line from `a` to `b`, both endpoints included, `width` px square pen.
pub fn line(tex: &mut Texture, a: [i32; 2], b: [i32; 2], width: u32, p: &Paint) -> usize {
    let mut m = Mask::empty(tex.width, tex.height);
    line_mask(&mut m, a, b, width);
    solid(tex, &m, p)
}

/// Connected crisp polyline through `pts`.
pub fn polyline(tex: &mut Texture, pts: &[[i32; 2]], width: u32, p: &Paint) -> usize {
    let mut m = Mask::empty(tex.width, tex.height);
    for s in pts.windows(2) {
        line_mask(&mut m, s[0], s[1], width);
    }
    if let [only] = pts {
        line_mask(&mut m, *only, *only, width);
    }
    solid(tex, &m, p)
}

/// Axis-aligned rectangle, `min`/`max` inclusive; outline is 1 px.
pub fn rect(tex: &mut Texture, min: [i32; 2], max: [i32; 2], filled: bool, p: &Paint) -> usize {
    let mut m = Region::Rect { min, max }.mask(tex.width, tex.height);
    if !filled {
        let (x0, x1, y0, y1) = (min[0].min(max[0]), min[0].max(max[0]), min[1].min(max[1]), min[1].max(max[1]));
        for y in (y0 + 1)..y1 {
            for x in (x0 + 1)..x1 {
                m.set(x, y, false);
            }
        }
    }
    solid(tex, &m, p)
}

/// Axis-aligned ellipse centred on pixel `center` with pixel radii (0 = 1 px thick).
/// The outline is the 1 px, 4-connected inner boundary of the filled shape.
pub fn ellipse(tex: &mut Texture, center: [i32; 2], radii: [i32; 2], filled: bool, p: &Paint) -> usize {
    let (rx, ry) = (radii[0].abs() as f32 + 0.5, radii[1].abs() as f32 + 0.5);
    let inside = |x: i32, y: i32| {
        let (dx, dy) = ((x - center[0]) as f32 / rx, (y - center[1]) as f32 / ry);
        dx * dx + dy * dy <= 1.0
    };
    let mut m = Mask::empty(tex.width, tex.height);
    for y in (center[1] - radii[1].abs())..=(center[1] + radii[1].abs()) {
        for x in (center[0] - radii[0].abs())..=(center[0] + radii[0].abs()) {
            if inside(x, y) && (filled || NEIGHBORS4.iter().any(|(dx, dy)| !inside(x + dx, y + dy))) {
                m.set(x, y, true);
            }
        }
    }
    solid(tex, &m, p)
}

/// Closed polygon through pixel centres `pts`; filled uses the even-odd rule plus the edges.
pub fn polygon(tex: &mut Texture, pts: &[[i32; 2]], filled: bool, p: &Paint) -> usize {
    let mut m = Mask::empty(tex.width, tex.height);
    if pts.is_empty() {
        return 0;
    }
    for (a, b) in pts.iter().zip(pts.iter().cycle().skip(1)) {
        line_mask(&mut m, *a, *b, 1);
    }
    if filled && pts.len() >= 3 {
        let y0 = pts.iter().map(|q| q[1]).min().unwrap().max(0);
        let y1 = pts.iter().map(|q| q[1]).max().unwrap().min(tex.height as i32 - 1);
        for y in y0..=y1 {
            let yc = y as f32;
            let mut xs: Vec<f32> = (0..pts.len())
                .filter_map(|i| {
                    let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
                    let (ay, by) = (a[1] as f32, b[1] as f32);
                    ((ay <= yc) != (by <= yc)).then(|| a[0] as f32 + (yc - ay) / (by - ay) * (b[0] - a[0]) as f32)
                })
                .collect();
            xs.sort_by(f32::total_cmp);
            for s in xs.chunks_exact(2) {
                for x in s[0].ceil() as i32..=s[1].floor() as i32 {
                    m.set(x, y, true);
                }
            }
        }
    }
    solid(tex, &m, p)
}

/// 4-connected flood fill from `at`, spreading over pixels whose every channel is
/// within `tolerance` of the seed pixel. Returns pixels changed.
pub fn flood_fill(tex: &mut Texture, at: [i32; 2], tolerance: u8, p: &Paint) -> usize {
    let Some(seed) = pixel(tex, at[0], at[1]) else { return 0 };
    let close = |c: Rgba| (0..4).all(|i| c[i].abs_diff(seed[i]) <= tolerance);
    let mut m = Mask::empty(tex.width, tex.height);
    let mut stack = vec![(at[0], at[1])];
    m.set(at[0], at[1], true);
    while let Some((x, y)) = stack.pop() {
        for (dx, dy) in NEIGHBORS4 {
            let (nx, ny) = (x + dx, y + dy);
            if !m.get(nx, ny) && pixel(tex, nx, ny).is_some_and(close) && p.mask.as_ref().is_none_or(|k| k.get(nx, ny)) {
                m.set(nx, ny, true);
                stack.push((nx, ny));
            }
        }
    }
    solid(tex, &m, p)
}

/// Gradient direction.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GradientDir {
    /// Linear across the region's bounding box; 0° = left→right, 90° = top→bottom.
    Linear { angle_deg: f32 },
    /// Linear from pixel point `from` (t=0) to `to` (t=1).
    Points { from: [f32; 2], to: [f32; 2] },
    /// Radial; defaults to the region's bbox centre and half its larger extent.
    Radial {
        #[serde(default)]
        center: Option<[f32; 2]>,
        #[serde(default)]
        radius: Option<f32>,
    },
}

/// Pick a colour along `from..to` for parameter `t`, optionally posterized into
/// `steps` bands with ordered dithering at band boundaries. With dithering but no
/// `steps`, 4 bands are used.
pub(crate) fn gradient_color(from: Rgba, to: Rgba, t: f32, x: i32, y: i32, dither: Dither, steps: Option<u32>) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    let n = steps.or((dither != Dither::None).then_some(4));
    match n {
        None | Some(0) => lerp(from, to, t),
        Some(1) => lerp(from, to, 0.5),
        Some(n) => {
            let f = t * (n - 1) as f32;
            let lo = f.floor();
            let band = if f - lo > dither.threshold(x, y) { lo + 1.0 } else { lo };
            lerp(from, to, band / (n - 1) as f32)
        }
    }
}

pub(crate) fn gradient_t(dir: &GradientDir, bbox: ([i32; 2], [i32; 2])) -> impl Fn(i32, i32) -> f32 + use<> {
    let (lo, hi) = (bbox.0.map(|v| v as f32), bbox.1.map(|v| v as f32 + 1.0));
    let dir = dir.clone();
    move |x, y| {
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        match &dir {
            GradientDir::Linear { angle_deg } => {
                let (s, c) = angle_deg.to_radians().sin_cos();
                let proj = |x: f32, y: f32| x * c + y * s;
                let corners = [proj(lo[0], lo[1]), proj(hi[0], lo[1]), proj(lo[0], hi[1]), proj(hi[0], hi[1])];
                let (mn, mx) = corners.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                // Sample range spans pixel centres so the end pixels hit the end colours.
                let (mn, mx) = (mn + 0.5 * (c.abs() + s.abs()), mx - 0.5 * (c.abs() + s.abs()));
                if mx - mn <= 1e-6 { 0.0 } else { (proj(px, py) - mn) / (mx - mn) }
            }
            GradientDir::Points { from, to } => {
                let d = [to[0] - from[0], to[1] - from[1]];
                let len2 = d[0] * d[0] + d[1] * d[1];
                if len2 <= 1e-9 { 0.0 } else { ((px - from[0]) * d[0] + (py - from[1]) * d[1]) / len2 }
            }
            GradientDir::Radial { center, radius } => {
                let c = center.unwrap_or([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5]);
                let r = radius.unwrap_or(((hi[0] - lo[0]).max(hi[1] - lo[1])) * 0.5).max(1e-3);
                ((px - c[0]).powi(2) + (py - c[1]).powi(2)).sqrt() / r
            }
        }
    }
}

/// Gradient over `region` (see [`GradientDir`]); `steps` posterizes into N bands
/// and `dither` blends band edges with an ordered pattern.
#[allow(clippy::too_many_arguments)]
pub fn gradient(
    tex: &mut Texture,
    region: &Region,
    from: Rgba,
    to: Rgba,
    dir: &GradientDir,
    dither: Dither,
    steps: Option<u32>,
    p: &Paint,
) -> usize {
    let m = region.mask(tex.width, tex.height);
    gradient_mask(tex, &m, from, to, dir, dither, steps, p)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_mask(
    tex: &mut Texture,
    m: &Mask,
    from: Rgba,
    to: Rgba,
    dir: &GradientDir,
    dither: Dither,
    steps: Option<u32>,
    p: &Paint,
) -> usize {
    let Some(bb) = m.bbox() else { return 0 };
    let t = gradient_t(dir, bb);
    apply(tex, m, p, |x, y, _| Some(gradient_color(from, to, t(x, y), x, y, dither, steps)))
}

/// Ordered dither between `a` and `b`: `density` 0 = all `a`, 1 = all `b`.
pub fn dither_fill(tex: &mut Texture, region: &Region, a: Rgba, b: Rgba, density: f32, pattern: Dither, p: &Paint) -> usize {
    let m = region.mask(tex.width, tex.height);
    apply(tex, &m, p, |x, y, _| Some(dither_pick(a, b, density, pattern, x, y)))
}

pub(crate) fn dither_pick(a: Rgba, b: Rgba, density: f32, pattern: Dither, x: i32, y: i32) -> Rgba {
    if density > pattern.threshold(x, y) { b } else { a }
}

/// Small deterministic integer hash (splitmix64 finaliser).
pub fn hash(seed: u64, x: i32, y: i32) -> u64 {
    let mut z = seed ^ (((x as u32 as u64) << 32) | y as u32 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Smooth value noise in 0..1 with lattice spacing `scale` pixels (2 octaves).
pub fn value_noise(seed: u64, x: i32, y: i32, scale: f32) -> f32 {
    let octave = |seed: u64, s: f32| {
        let (fx, fy) = ((x as f32 + 0.5) / s, (y as f32 + 0.5) / s);
        let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
        let sm = |t: f32| t * t * (3.0 - 2.0 * t);
        let (tx, ty) = (sm(fx - ix as f32), sm(fy - iy as f32));
        let v = |dx, dy| unit(hash(seed, ix + dx, iy + dy));
        let top = v(0, 0) + (v(1, 0) - v(0, 0)) * tx;
        let bot = v(0, 1) + (v(1, 1) - v(0, 1)) * tx;
        top + (bot - top) * ty
    };
    let s = scale.max(1.0);
    if s <= 1.0 {
        return unit(hash(seed, x, y));
    }
    (octave(seed, s) * 2.0 + octave(seed.wrapping_add(0x51ed), (s * 0.5).max(1.0))) / 3.0
}

pub(crate) fn noise_pick(colors: &[Rgba], seed: u64, scale: f32, x: i32, y: i32) -> Option<Rgba> {
    if colors.is_empty() {
        return None;
    }
    // Stretch the (centre-heavy) noise distribution so every colour shows up.
    let v = ((value_noise(seed, x, y, scale) - 0.5) * 1.6 + 0.5).clamp(0.0, 0.9999);
    Some(colors[(v * colors.len() as f32) as usize])
}

/// Deterministic value noise picking among `colors`; `scale` is the blob size in pixels
/// (1 = per-pixel speckle).
pub fn noise(tex: &mut Texture, region: &Region, colors: &[Rgba], seed: u64, scale: f32, p: &Paint) -> usize {
    let m = region.mask(tex.width, tex.height);
    apply(tex, &m, p, |x, y, _| noise_pick(colors, seed, scale, x, y))
}

/// Pixel-art outline: every transparent pixel in `region` that 4-touches a
/// non-transparent pixel gets `color`.
pub fn outline(tex: &mut Texture, region: &Region, color: Rgba) -> usize {
    let region_mask = region.mask(tex.width, tex.height);
    let mut m = Mask::empty(tex.width, tex.height);
    for (x, y) in region_mask.iter() {
        let opaque = |x, y| pixel(tex, x, y).is_some_and(|c| c[3] > 0);
        if !opaque(x, y) && NEIGHBORS4.iter().any(|(dx, dy)| opaque(x + dx, y + dy)) {
            m.set(x, y, true);
        }
    }
    solid(tex, &m, &Paint::new(color))
}

/// Draw the inner `width`-px border of `region` (e.g. panel lines around faces):
/// covered texels within `width` 4-connected steps of an uncovered or off-texture texel.
pub fn border(tex: &mut Texture, region: &Region, width: u32, p: &Paint) -> usize {
    let inside = region.mask(tex.width, tex.height);
    let mut reached = Mask::empty(tex.width, tex.height);
    let outside = |m: &Mask, x: i32, y: i32| !inside.get(x, y) || m.get(x, y);
    for _ in 0..width.max(1) {
        let prev = reached.clone();
        for (x, y) in inside.iter() {
            if NEIGHBORS4.iter().any(|(dx, dy)| outside(&prev, x + dx, y + dy)) {
                reached.set(x, y, true);
            }
        }
    }
    solid(tex, &reached, p)
}

/// Shift hue by `hue_deg` and multiply saturation / value; alpha is kept.
/// Uses `p.mask` and `p.palette` but ignores `p.color` / `p.blend`.
pub fn adjust(tex: &mut Texture, region: &Region, hue_deg: f32, sat: f32, value: f32, p: &Paint) -> usize {
    let m = region.mask(tex.width, tex.height);
    let p = Paint { blend: Blend::Replace, ..p.clone() };
    apply(tex, &m, &p, |_, _, c| {
        let [h, s, v] = to_hsv(c);
        Some(from_hsv([h + hue_deg, (s * sat).clamp(0.0, 1.0), (v * value).clamp(0.0, 1.0)], c[3]))
    })
}

/// Replace every pixel equal to a key of `map` (exact RGBA match) with its value.
pub fn replace_colors(tex: &mut Texture, region: &Region, map: &[(Rgba, Rgba)], p: &Paint) -> usize {
    let m = region.mask(tex.width, tex.height);
    let p = Paint { blend: Blend::Replace, ..p.clone() };
    apply(tex, &m, &p, |_, _, c| map.iter().find(|(k, _)| *k == c).map(|(_, v)| *v))
}

/// Distinct colours in `region`.
pub fn colors_in(tex: &Texture, region: &Region) -> BTreeSet<Rgba> {
    region.mask(tex.width, tex.height).iter().filter_map(|(x, y)| pixel(tex, x, y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLACK: Rgba = [0, 0, 0, 255];
    const WHITE: Rgba = [255, 255, 255, 255];
    const RED: Rgba = [255, 0, 0, 255];

    fn count(t: &Texture, c: Rgba) -> usize {
        t.pixels.chunks_exact(4).filter(|p| *p == c).count()
    }

    #[test]
    fn bayer_is_permutation() {
        for (d, n) in [(Dither::Bayer2, 2), (Dither::Bayer4, 4), (Dither::Bayer8, 8)] {
            let mut v: Vec<u32> =
                (0..n).flat_map(|y| (0..n).map(move |x| (d.threshold(x, y) * (n * n) as f32) as u32)).collect();
            v.sort();
            assert_eq!(v, (0..(n * n) as u32).collect::<Vec<_>>());
        }
        assert_eq!(Dither::Bayer2.threshold(1, 0), 2.5 / 4.0);
    }

    #[test]
    fn line_endpoints() {
        let mut t = Texture::new("t", 16, 16, BLACK);
        let n = line(&mut t, [1, 2], [13, 9], 1, &Paint::new(WHITE));
        assert_eq!(t.get(1, 2), WHITE);
        assert_eq!(t.get(13, 9), WHITE);
        assert_eq!(n, 13); // x-major: one pixel per column
        // Clipped line partially off-texture.
        let n = line(&mut t, [-5, 0], [3, 0], 1, &Paint::new(RED));
        assert_eq!(n, 4);
        let mut t = Texture::new("t", 16, 16, BLACK);
        assert_eq!(line(&mut t, [2, 5], [10, 5], 3, &Paint::new(WHITE)), 33); // square pen overhangs the ends
    }

    #[test]
    fn shapes() {
        let mut t = Texture::new("t", 16, 16, BLACK);
        assert_eq!(rect(&mut t, [2, 2], [5, 4], true, &Paint::new(WHITE)), 12);
        let mut t = Texture::new("t", 16, 16, BLACK);
        assert_eq!(rect(&mut t, [5, 4], [2, 2], false, &Paint::new(WHITE)), 10);
        let mut t = Texture::new("t", 16, 16, BLACK);
        ellipse(&mut t, [8, 8], [3, 3], true, &Paint::new(WHITE));
        assert_eq!(t.get(8, 8), WHITE);
        assert_eq!(t.get(11, 8), WHITE);
        assert_eq!(t.get(11, 11), BLACK);
        let filled = count(&t, WHITE);
        let mut t = Texture::new("t", 16, 16, BLACK);
        ellipse(&mut t, [8, 8], [3, 3], false, &Paint::new(WHITE));
        assert_eq!(t.get(8, 8), BLACK);
        assert!(count(&t, WHITE) < filled);
        let mut t = Texture::new("t", 16, 16, BLACK);
        polygon(&mut t, &[[0, 0], [7, 0], [7, 7], [0, 7]], true, &Paint::new(WHITE));
        assert_eq!(count(&t, WHITE), 64);
    }

    #[test]
    fn flood_counts() {
        let mut t = Texture::new("t", 10, 10, BLACK);
        line(&mut t, [5, 0], [5, 9], 1, &Paint::new(WHITE));
        assert_eq!(flood_fill(&mut t, [0, 0], 0, &Paint::new(RED)), 50);
        assert_eq!(flood_fill(&mut t, [9, 9], 0, &Paint::new(RED)), 40);
        assert_eq!(flood_fill(&mut t, [9, 9], 0, &Paint::new(RED)), 0);
        assert_eq!(flood_fill(&mut t, [20, 9], 0, &Paint::new(RED)), 0);
    }

    #[test]
    fn dither_half() {
        for d in [Dither::Bayer2, Dither::Bayer4, Dither::Bayer8] {
            let mut t = Texture::new("t", 32, 32, [0; 4]);
            dither_fill(&mut t, &Region::All, BLACK, WHITE, 0.5, d, &Paint::default());
            assert_eq!(count(&t, BLACK), 512);
            assert_eq!(count(&t, WHITE), 512);
        }
    }

    #[test]
    fn gradient_bands() {
        let mut t = Texture::new("t", 16, 1, [0; 4]);
        let dir = GradientDir::Linear { angle_deg: 0.0 };
        gradient(&mut t, &Region::All, BLACK, WHITE, &dir, Dither::None, None, &Paint::default());
        assert_eq!(t.get(0, 0), BLACK);
        assert_eq!(t.get(15, 0), WHITE);
        gradient(&mut t, &Region::All, BLACK, WHITE, &dir, Dither::None, Some(4), &Paint::default());
        let uniq: BTreeSet<Rgba> = t.pixels.chunks_exact(4).map(|p| p.try_into().unwrap()).collect();
        assert_eq!(uniq.len(), 4);
    }

    #[test]
    fn blend_and_palette_lock() {
        assert_eq!(blend(WHITE, [0, 0, 0, 128], Blend::Alpha), [127, 127, 127, 255]);
        assert_eq!(blend(WHITE, RED, Blend::Multiply), RED);
        let mut t = Texture::new("t", 4, 4, BLACK);
        let p = Paint::new([250, 10, 10, 255]).with_palette(Some(vec![BLACK, RED]));
        rect(&mut t, [0, 0], [3, 3], true, &p);
        assert_eq!(count(&t, RED), 16);
    }

    #[test]
    fn mask_restricts() {
        let mut t = Texture::new("t", 8, 8, BLACK);
        let mask = Region::Rect { min: [0, 0], max: [3, 7] }.mask(8, 8);
        let p = Paint::new(WHITE).with_mask(Some(mask));
        assert_eq!(rect(&mut t, [0, 0], [7, 7], true, &p), 32);
    }

    #[test]
    fn outline_and_noise() {
        let mut t = Texture::new("t", 8, 8, [0; 4]);
        rect(&mut t, [3, 3], [4, 4], true, &Paint::new(WHITE));
        assert_eq!(outline(&mut t, &Region::All, BLACK), 8);
        let mut a = Texture::new("a", 16, 16, [0; 4]);
        let mut b = Texture::new("b", 16, 16, [0; 4]);
        let cols = [BLACK, WHITE, RED];
        noise(&mut a, &Region::All, &cols, 7, 4.0, &Paint::default());
        noise(&mut b, &Region::All, &cols, 7, 4.0, &Paint::default());
        assert_eq!(a.pixels, b.pixels);
        assert!(cols.iter().all(|c| count(&a, *c) > 0));
        let mut t = Texture::new("t", 8, 8, BLACK);
        assert_eq!(border(&mut t, &Region::All, 1, &Paint::new(WHITE)), 28);
    }
}
