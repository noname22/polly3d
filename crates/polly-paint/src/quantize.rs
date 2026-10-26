//! Palette quantization and resampling.

use polly_core::scene::Texture;

use crate::canvas::Dither;
use crate::color::Rgba;
use crate::palette::{distance, nearest};

/// Snap every non-transparent pixel to `palette`. With an ordered `dither`, each
/// pixel chooses between its two nearest palette colours according to where it
/// lies between them, giving the classic crosshatch blend. Alpha is preserved.
/// Returns pixels changed.
pub fn quantize(tex: &mut Texture, palette: &[Rgba], dither: Dither) -> usize {
    if palette.is_empty() {
        return 0;
    }
    let w = tex.width as usize;
    let mut changed = 0;
    for (i, px) in tex.pixels_mut().chunks_exact_mut(4).enumerate() {
        let c: Rgba = (&*px).try_into().unwrap();
        if c[3] == 0 {
            continue;
        }
        let out = if dither == Dither::None { nearest(palette, c).1 } else { dithered(palette, c, dither, i % w, i / w) };
        if out != c {
            px.copy_from_slice(&out);
            changed += 1;
        }
    }
    changed
}

fn dithered(palette: &[Rgba], c: Rgba, dither: Dither, x: usize, y: usize) -> Rgba {
    let (i1, c1) = nearest(palette, c);
    let second = palette
        .iter()
        .enumerate()
        .filter(|(i, p)| *i != i1 && p[3] > 0 && p[..3] != c1[..3])
        .min_by(|(_, a), (_, b)| distance(**a, c).total_cmp(&distance(**b, c)));
    let Some((_, c2)) = second else { return c1 };
    // Fraction of the way from c1 to c2 (projected in RGB).
    let d: [f32; 3] = std::array::from_fn(|k| c2[k] as f32 - c1[k] as f32);
    let v: [f32; 3] = std::array::from_fn(|k| c[k] as f32 - c1[k] as f32);
    let len2: f32 = d.iter().map(|x| x * x).sum();
    let t = (d.iter().zip(v).map(|(a, b)| a * b).sum::<f32>() / len2).clamp(0.0, 1.0);
    if t > dither.threshold(x as i32, y as i32) { [c2[0], c2[1], c2[2], c[3]] } else { c1 }
}

/// Nearest-neighbour resample to `w` x `h` (texel-centre mapping). Other fields are copied.
pub fn resize_nearest(tex: &Texture, w: u32, h: u32) -> Texture {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let sy = (((y as f32 + 0.5) * tex.height as f32 / h as f32) as u32).min(tex.height.saturating_sub(1));
        for x in 0..w {
            let sx = (((x as f32 + 0.5) * tex.width as f32 / w as f32) as u32).min(tex.width.saturating_sub(1));
            pixels.extend_from_slice(&tex.get(sx, sy));
        }
    }
    Texture { width: w, height: h, pixels: std::sync::Arc::new(pixels), ..tex.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::builtin;

    #[test]
    fn quantize_only_palette() {
        let pal = builtin("pico8").unwrap();
        for d in [Dither::None, Dither::Bayer4] {
            let mut t = Texture::new("t", 16, 16, [0; 4]);
            for (i, p) in t.pixels_mut().chunks_exact_mut(4).enumerate() {
                p.copy_from_slice(&[(i * 7) as u8, (i * 13) as u8, (i * 3) as u8, if i == 5 { 0 } else { 255 }]);
            }
            quantize(&mut t, &pal, d);
            for (i, p) in t.pixels.chunks_exact(4).enumerate() {
                if i == 5 {
                    assert_eq!(p[3], 0);
                } else {
                    assert!(pal.contains(&p.try_into().unwrap()), "{p:?}");
                }
            }
        }
    }

    #[test]
    fn dither_mixes_midtone() {
        let pal = [[0, 0, 0, 255], [255, 255, 255, 255]];
        let mut t = Texture::new("t", 8, 8, [128, 128, 128, 255]);
        quantize(&mut t, &pal, Dither::Bayer4);
        let white = t.pixels.chunks_exact(4).filter(|p| p[0] == 255).count();
        assert!((28..=36).contains(&white), "{white}");
    }

    #[test]
    fn resize() {
        let mut t = Texture::new("t", 2, 2, [0, 0, 0, 255]);
        t.pixels_mut()[0..4].copy_from_slice(&[255, 0, 0, 255]);
        let r = resize_nearest(&t, 4, 4);
        assert_eq!(r.get(1, 1), [255, 0, 0, 255]);
        assert_eq!(r.get(2, 1), [0, 0, 0, 255]);
        assert_eq!(resize_nearest(&r, 2, 2).pixels, t.pixels);
    }
}
