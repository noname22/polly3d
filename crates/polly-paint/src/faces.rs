//! Surface-space painting: rasterize mesh faces' UV polygons into the texture.
//!
//! Coverage uses the texel-centre rule: texel (x, y) belongs to a face when its
//! centre ((x+0.5)/w, (y+0.5)/h) lies inside (or on the edge of) the face's UV
//! polygon. Faces smaller than a texel still claim the texel under their centroid.

use std::collections::BTreeSet;

use glam::{Vec2, Vec4};
use polly_core::geom::triangulate_2d;
use polly_core::mesh::{FaceId, Mesh};
use polly_core::scene::Texture;
use serde::{Deserialize, Serialize};

use crate::canvas::{Dither, GradientDir, Paint, apply, dither_pick, gradient_mask, hash, noise_pick};
use crate::color::{Rgba, from_f32};
use crate::mask::{FaceUv, Mask, NEIGHBORS8};

/// Corner UV polygons of `faces` (missing ids are skipped), in face-id order.
pub fn face_uv_polys(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Vec<FaceUv> {
    faces.iter().filter_map(|f| mesh.faces.get(f)).map(|f| f.corners.iter().map(|c| c.uv.to_array()).collect()).collect()
}

fn to_px(uv: [f32; 2], w: u32, h: u32) -> Vec2 {
    Vec2::new(uv[0] * w as f32, uv[1] * h as f32)
}

/// Visit every texel centre inside triangle `t` (pixel space, either winding)
/// with its barycentric weights.
fn raster_tri(t: [Vec2; 3], w: u32, h: u32, mut f: impl FnMut(i32, i32, [f32; 3])) {
    let area = (t[1] - t[0]).perp_dot(t[2] - t[0]);
    if area.abs() < 1e-12 {
        return;
    }
    let lo = t[0].min(t[1]).min(t[2]);
    let hi = t[0].max(t[1]).max(t[2]);
    let x0 = (lo.x - 0.5).floor().max(0.0) as i32;
    let y0 = (lo.y - 0.5).floor().max(0.0) as i32;
    let x1 = ((hi.x - 0.5).ceil() as i32).min(w as i32 - 1);
    let y1 = ((hi.y - 0.5).ceil() as i32).min(h as i32 - 1);
    let eps = 1e-5 * area.abs().max(1.0);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let e = [(t[2] - t[1]).perp_dot(p - t[1]), (t[0] - t[2]).perp_dot(p - t[2]), (t[1] - t[0]).perp_dot(p - t[0])];
            let inside = e.iter().all(|&v| v * area.signum() >= -eps);
            if inside {
                f(x, y, e.map(|v| (v / area).clamp(0.0, 1.0)));
            }
        }
    }
}

/// Rasterize one UV polygon; calls `f(x, y, corner_weights)` per covered texel.
fn raster_poly(poly: &[[f32; 2]], w: u32, h: u32, mut f: impl FnMut(i32, i32, &[(usize, f32); 3])) {
    if poly.len() < 3 || w == 0 || h == 0 {
        return;
    }
    let pts: Vec<Vec2> = poly.iter().map(|&uv| to_px(uv, w, h)).collect();
    let mut any = false;
    for [a, b, c] in triangulate_2d(&pts) {
        raster_tri([pts[a], pts[b], pts[c]], w, h, |x, y, bc| {
            any = true;
            f(x, y, &[(a, bc[0]), (b, bc[1]), (c, bc[2])]);
        });
    }
    if !any {
        // Sub-texel face: claim the texel under its centroid so it still gets painted.
        let c = pts.iter().copied().sum::<Vec2>() / pts.len() as f32;
        let (x, y) = (c.x.floor() as i32, c.y.floor() as i32);
        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
            let k = 1.0 / 3.0;
            f(x, y, &[(0, k), (1, k), (2, k)]);
        }
    }
}

/// Texel-centre coverage of UV polygons for a `tex_w` x `tex_h` texture, dilated
/// by `grow_px` texels (use 1-2 to avoid seam bleed with nearest filtering).
pub fn face_mask(tex_w: u32, tex_h: u32, polys: &[FaceUv], grow_px: u32) -> Mask {
    let mut m = Mask::empty(tex_w, tex_h);
    for poly in polys {
        raster_poly(poly, tex_w, tex_h, |x, y, _| m.set(x, y, true));
    }
    m.dilate(grow_px);
    m
}

fn one() -> u32 {
    1
}

/// What to paint into faces. Procedural patterns are aligned to the texel grid in
/// texture space, so they tile consistently across neighbouring faces.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Fill {
    /// One flat colour.
    Solid { color: Rgba },
    /// Interpolate the faces' per-corner vertex colours (bakes vertex colours).
    VertexColors,
    /// Gradient over the bounding box of all covered texels.
    Gradient {
        from: Rgba,
        to: Rgba,
        dir: GradientDir,
        #[serde(default)]
        dither: Dither,
        #[serde(default)]
        steps: Option<u32>,
    },
    /// Ordered dither between `a` and `b`; `density` 0 = all `a`, 1 = all `b`.
    Dither { a: Rgba, b: Rgba, density: f32, pattern: Dither },
    /// Value noise choosing among `colors`; `scale` = blob size in px.
    Noise {
        colors: Vec<Rgba>,
        #[serde(default)]
        seed: u64,
        scale: f32,
    },
    /// Checkerboard of `size_px` squares.
    Checker { a: Rgba, b: Rgba, size_px: u32 },
    /// Repeating stripes, each `width_px` wide, cycling through `colors`.
    Stripes {
        colors: Vec<Rgba>,
        width_px: u32,
        #[serde(default)]
        vertical: bool,
    },
    /// Running-bond bricks of `size_px` = [w, h] (including mortar); each brick
    /// picks a colour from `brick` by hash; mortar lines are `mortar_px` thick.
    Bricks {
        mortar: Rgba,
        brick: Vec<Rgba>,
        size_px: [u32; 2],
        #[serde(default = "one")]
        mortar_px: u32,
        #[serde(default)]
        seed: u64,
    },
    /// Wood planks `width_px` wide (1 px `gap` line between planks), with staggered
    /// end joints every `length_px` (0 = none) and optional sparse `grain` streaks.
    /// Planks run horizontally unless `vertical`.
    Planks {
        colors: Vec<Rgba>,
        gap: Rgba,
        width_px: u32,
        #[serde(default)]
        length_px: u32,
        #[serde(default)]
        vertical: bool,
        #[serde(default)]
        grain: Option<Rgba>,
        #[serde(default)]
        seed: u64,
    },
}

fn pick(colors: &[Rgba], h: u64) -> Option<Rgba> {
    (!colors.is_empty()).then(|| colors[(h % colors.len() as u64) as usize])
}

impl Fill {
    /// Colour of a per-pixel (non-vertex-colour, non-gradient) fill at texel `(x, y)`.
    pub fn sample(&self, x: i32, y: i32) -> Option<Rgba> {
        match self {
            Fill::Solid { color } => Some(*color),
            Fill::VertexColors | Fill::Gradient { .. } => None,
            Fill::Dither { a, b, density, pattern } => Some(dither_pick(*a, *b, *density, *pattern, x, y)),
            Fill::Noise { colors, seed, scale } => noise_pick(colors, *seed, *scale, x, y),
            Fill::Checker { a, b, size_px } => {
                let s = (*size_px).max(1) as i32;
                Some(if (x.div_euclid(s) + y.div_euclid(s)) % 2 == 0 { *a } else { *b })
            }
            Fill::Stripes { colors, width_px, vertical } => {
                let s = (*width_px).max(1) as i32;
                let k = if *vertical { x } else { y }.div_euclid(s);
                pick(colors, k.rem_euclid(colors.len().max(1) as i32) as u64)
            }
            Fill::Bricks { mortar, brick, size_px, mortar_px, seed } => {
                let (bw, bh) = (size_px[0].max(2) as i32, size_px[1].max(2) as i32);
                let row = y.div_euclid(bh);
                let xs = x + if row.rem_euclid(2) == 1 { bw / 2 } else { 0 };
                let (col, lx, ly) = (xs.div_euclid(bw), xs.rem_euclid(bw), y.rem_euclid(bh));
                let m = *mortar_px as i32;
                if lx < m || ly < m { Some(*mortar) } else { pick(brick, hash(*seed, col, row)).or(Some(*mortar)) }
            }
            Fill::Planks { colors, gap, width_px, length_px, vertical, grain, seed } => {
                let (a, b) = if *vertical { (y, x) } else { (x, y) };
                let pw = (*width_px).max(2) as i32;
                let (strip, across) = (b.div_euclid(pw), b.rem_euclid(pw));
                if across == 0 {
                    return Some(*gap);
                }
                let mut seg = 0;
                if *length_px > 0 {
                    let len = *length_px as i32;
                    let along = a + (hash(*seed, strip, -1) % len as u64) as i32;
                    if along.rem_euclid(len) == 0 {
                        return Some(*gap);
                    }
                    seg = along.div_euclid(len);
                }
                if let Some(g) = grain {
                    let streak = hash(seed ^ 0x6a09_e667, b, 0).is_multiple_of(3);
                    if streak && hash(seed ^ 0xbb67_ae85, a.div_euclid(3), b).is_multiple_of(3) {
                        return Some(*g);
                    }
                }
                pick(colors, hash(*seed, strip, seg)).or(Some(*gap))
            }
        }
    }
}

/// Paint `fill` into the texels covered by `faces` (grown by `grow_px`). Returns
/// pixels changed. Respects `p.blend`, `p.palette` and `p.mask` (`p.color` unused).
pub fn fill_faces(tex: &mut Texture, mesh: &Mesh, faces: &BTreeSet<FaceId>, fill: &Fill, grow_px: u32, p: &Paint) -> usize {
    let (w, h) = (tex.width, tex.height);
    match fill {
        Fill::VertexColors => {
            let mut buf: Vec<Option<Vec4>> = vec![None; (w * h) as usize];
            for f in faces.iter().filter_map(|f| mesh.faces.get(f)) {
                let poly: FaceUv = f.corners.iter().map(|c| c.uv.to_array()).collect();
                raster_poly(&poly, w, h, |x, y, wts| {
                    let c = wts.iter().map(|&(i, k)| f.corners[i].color * k).sum::<Vec4>();
                    buf[(y as u32 * w + x as u32) as usize] = Some(c);
                });
            }
            // Edge padding: grown texels copy the average of covered neighbours.
            for _ in 0..grow_px {
                let prev = buf.clone();
                for y in 0..h as i32 {
                    for x in 0..w as i32 {
                        if prev[(y as u32 * w + x as u32) as usize].is_some() {
                            continue;
                        }
                        let ns: Vec<Vec4> = NEIGHBORS8
                            .iter()
                            .map(|(dx, dy)| (x + dx, y + dy))
                            .filter(|&(nx, ny)| nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h)
                            .filter_map(|(nx, ny)| prev[(ny as u32 * w + nx as u32) as usize])
                            .collect();
                        if !ns.is_empty() {
                            buf[(y as u32 * w + x as u32) as usize] = Some(ns.iter().sum::<Vec4>() / ns.len() as f32);
                        }
                    }
                }
            }
            let mut m = Mask::empty(w, h);
            m.bits = buf.iter().map(Option::is_some).collect();
            apply(tex, &m, p, |x, y, _| buf[(y as u32 * w + x as u32) as usize].map(|c| from_f32(c.to_array())))
        }
        _ => {
            let m = face_mask(w, h, &face_uv_polys(mesh, faces), grow_px);
            match fill {
                Fill::Gradient { from, to, dir, dither, steps } => gradient_mask(tex, &m, *from, *to, dir, *dither, *steps, p),
                _ => apply(tex, &m, p, |x, y, _| fill.sample(x, y)),
            }
        }
    }
}

/// Bake per-corner vertex colours of `faces` into the texture (1 px edge padding).
pub fn bake_vertex_colors(tex: &mut Texture, mesh: &Mesh, faces: &BTreeSet<FaceId>) -> usize {
    fill_faces(tex, mesh, faces, &Fill::VertexColors, 1, &Paint::default())
}

fn stamp(m: &mut Mask, c: Vec2, r: f32) {
    m.set(c.x.floor() as i32, c.y.floor() as i32, true);
    let ri = r.ceil() as i32 + 1;
    let (cx, cy) = (c.x.floor() as i32, c.y.floor() as i32);
    for y in (cy - ri)..=(cy + ri) {
        for x in (cx - ri)..=(cx + ri) {
            if Vec2::new(x as f32 + 0.5, y as f32 + 0.5).distance(c) <= r {
                m.set(x, y, true);
            }
        }
    }
}

/// Round pixel-brush stamp at `uv` with radius `radius_px` texels (always covers
/// at least the texel under `uv`). Returns pixels changed.
pub fn paint_at_uv(tex: &mut Texture, uv: [f32; 2], radius_px: f32, p: &Paint) -> usize {
    stroke_uv(tex, &[uv], radius_px, p)
}

/// Stamp the round brush densely along the UV polyline `pts` (each texel is
/// written at most once, so alpha strokes don't build up). Returns pixels changed.
pub fn stroke_uv(tex: &mut Texture, pts: &[[f32; 2]], radius_px: f32, p: &Paint) -> usize {
    let (w, h) = (tex.width, tex.height);
    let mut m = Mask::empty(w, h);
    let px: Vec<Vec2> = pts.iter().map(|&uv| to_px(uv, w, h)).collect();
    if let [only] = px.as_slice() {
        stamp(&mut m, *only, radius_px);
    }
    for s in px.windows(2) {
        let n = (s[0].distance(s[1]) * 2.0).ceil().max(1.0) as usize;
        for i in 0..=n {
            stamp(&mut m, s[0].lerp(s[1], i as f32 / n as f32), radius_px);
        }
    }
    apply(tex, &m, p, |_, _, _| Some(p.color))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polly_core::primitives::cube;

    const BLACK: Rgba = [0, 0, 0, 255];
    const WHITE: Rgba = [255, 255, 255, 255];

    fn count(t: &Texture, c: Rgba) -> usize {
        t.pixels.chunks_exact(4).filter(|p| *p == c).count()
    }

    fn cube_with_full_uvs() -> (Mesh, FaceId) {
        let mut m = cube(glam::Vec3::ONE);
        let f = *m.faces.keys().next().unwrap();
        let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for (c, uv) in m.faces.get_mut(&f).unwrap().corners.iter_mut().zip(uvs) {
            c.uv = Vec2::from_array(uv);
        }
        (m, f)
    }

    #[test]
    fn full_square_face_fills_texture() {
        let (m, f) = cube_with_full_uvs();
        let mut t = Texture::new("t", 16, 16, BLACK);
        let n = fill_faces(&mut t, &m, &BTreeSet::from([f]), &Fill::Solid { color: WHITE }, 0, &Paint::default());
        assert_eq!(n, 256);
    }

    #[test]
    fn half_square_triangle() {
        // Texel centres on the diagonal (x + y = 7) lie exactly on the edge and are included.
        let tri = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let m = face_mask(8, 8, std::slice::from_ref(&tri), 0);
        assert_eq!(m.count(), 36);
        assert!(m.get(0, 0) && m.get(7, 0) && m.get(3, 4) && !m.get(4, 4) && !m.get(7, 7));
        let m = face_mask(9, 9, &[tri], 0);
        assert_eq!(m.count(), 45); // x + y <= 8
    }

    #[test]
    fn mask_grow() {
        let quad = vec![[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]];
        let m0 = face_mask(8, 8, std::slice::from_ref(&quad), 0);
        assert_eq!(m0.count(), 16);
        let m1 = face_mask(8, 8, std::slice::from_ref(&quad), 1);
        assert_eq!(m1.count(), 36);
        assert!(m1.get(1, 1) && !m1.get(0, 0));
        assert_eq!(face_mask(8, 8, &[quad], 3).count(), 64);
        // Sub-texel face still claims one texel.
        let tiny = vec![[0.51, 0.51], [0.52, 0.51], [0.51, 0.52]];
        assert_eq!(face_mask(8, 8, &[tiny], 0).count(), 1);
    }

    #[test]
    fn vertex_colors_and_patterns() {
        let (mut m, f) = cube_with_full_uvs();
        for c in m.faces.get_mut(&f).unwrap().corners.iter_mut() {
            c.color = Vec4::new(1.0, 0.0, 0.0, 1.0);
        }
        let mut t = Texture::new("t", 8, 8, BLACK);
        assert_eq!(bake_vertex_colors(&mut t, &m, &BTreeSet::from([f])), 64);
        assert_eq!(count(&t, [255, 0, 0, 255]), 64);

        let fill = Fill::Checker { a: BLACK, b: WHITE, size_px: 2 };
        let mut t = Texture::new("t", 8, 8, [0; 4]);
        fill_faces(&mut t, &m, &BTreeSet::from([f]), &fill, 0, &Paint::default());
        assert_eq!(count(&t, WHITE), 32);
        assert_eq!(t.get(2, 0), WHITE);

        let fill = Fill::Bricks { mortar: BLACK, brick: vec![WHITE], size_px: [4, 4], mortar_px: 1, seed: 0 };
        let mut t = Texture::new("t", 8, 8, [0; 4]);
        fill_faces(&mut t, &m, &BTreeSet::from([f]), &fill, 0, &Paint::default());
        assert_eq!(count(&t, WHITE), 36); // 3x3 brick interior per 4x4 cell
        let fill = Fill::Planks {
            colors: vec![WHITE],
            gap: BLACK,
            width_px: 4,
            length_px: 0,
            vertical: false,
            grain: None,
            seed: 0,
        };
        fill_faces(&mut t, &m, &BTreeSet::from([f]), &fill, 0, &Paint::default());
        assert_eq!(count(&t, BLACK), 16);
    }

    #[test]
    fn strokes() {
        let mut t = Texture::new("t", 16, 16, BLACK);
        assert_eq!(paint_at_uv(&mut t, [0.5, 0.5], 0.0, &Paint::new(WHITE)), 1);
        let mut t = Texture::new("t", 16, 16, BLACK);
        let n = stroke_uv(&mut t, &[[0.0, 0.5 / 16.0], [1.0, 0.5 / 16.0]], 0.5, &Paint::new(WHITE));
        assert_eq!(n, 16);
        assert_eq!(t.get(0, 0), WHITE);
        assert_eq!(t.get(15, 0), WHITE);
    }
}
