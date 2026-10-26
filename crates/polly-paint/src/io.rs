//! PNG encode/decode and texture statistics.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use polly_core::Result;
use polly_core::error::invalid;
use polly_core::scene::Texture;
use serde::Serialize;

use crate::color::{Rgba, to_hex};

/// Encode as an RGBA8 PNG.
pub fn to_png(tex: &Texture) -> Vec<u8> {
    let img = image::RgbaImage::from_raw(tex.width, tex.height, tex.pixels.to_vec()).expect("pixel buffer matches size");
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).expect("PNG encoding to memory");
    out.into_inner()
}

/// Decode a PNG (or any format the `image` crate was built with) into a nearest-filtered texture.
pub fn from_png(name: &str, bytes: &[u8]) -> Result<Texture> {
    let img = match image::load_from_memory(bytes) {
        Ok(i) => i.to_rgba8(),
        Err(e) => return invalid(format!("cannot decode image: {e}")),
    };
    let (width, height) = img.dimensions();
    let mut t = Texture::new(name, 0, 0, [0; 4]);
    t.width = width;
    t.height = height;
    t.pixels = Arc::new(img.into_raw());
    Ok(t)
}

/// Summary of a texture's contents.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct TextureStats {
    pub width: u32,
    pub height: u32,
    pub unique_colors: usize,
    /// Pixels (alpha > 0) whose colour is not in the given palette; `None` without a palette.
    pub palette_violations: Option<usize>,
    pub transparent_pixels: usize,
    /// Most frequent colours as (hex, count), up to 8.
    pub top_colors: Vec<(String, usize)>,
}

/// Compute [`TextureStats`]; pass the texture's locked palette to count violations.
pub fn stats(tex: &Texture, palette: Option<&[Rgba]>) -> TextureStats {
    let mut hist: HashMap<Rgba, usize> = HashMap::new();
    for p in tex.pixels.chunks_exact(4) {
        *hist.entry(p.try_into().unwrap()).or_default() += 1;
    }
    let transparent_pixels = hist.iter().filter(|(c, _)| c[3] == 0).map(|(_, n)| n).sum();
    let palette_violations = palette.map(|pal| {
        hist.iter().filter(|(c, _)| c[3] > 0 && !pal.iter().any(|p| p[..3] == c[..3])).map(|(_, n)| n).sum()
    });
    let mut top: Vec<(Rgba, usize)> = hist.iter().map(|(c, n)| (*c, *n)).collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    TextureStats {
        width: tex.width,
        height: tex.height,
        unique_colors: hist.len(),
        palette_violations,
        transparent_pixels,
        top_colors: top.into_iter().take(8).map(|(c, n)| (to_hex(c), n)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip() {
        let mut t = Texture::new("t", 5, 3, [10, 20, 30, 255]);
        t.pixels_mut()[4..8].copy_from_slice(&[1, 2, 3, 0]);
        t.pixels_mut()[20..24].copy_from_slice(&[200, 100, 50, 128]);
        let png = to_png(&t);
        assert_eq!(&png[1..4], b"PNG");
        let back = from_png("back", &png).unwrap();
        assert_eq!((back.width, back.height), (5, 3));
        assert_eq!(back.pixels, t.pixels);
        assert!(from_png("x", b"not a png").is_err());
    }

    #[test]
    fn texture_stats() {
        let mut t = Texture::new("t", 4, 4, [10, 20, 30, 255]);
        t.pixels_mut()[0..4].copy_from_slice(&[0, 0, 0, 0]);
        t.pixels_mut()[4..8].copy_from_slice(&[255, 0, 0, 255]);
        let s = stats(&t, Some(&[[10, 20, 30, 255]]));
        assert_eq!(s.unique_colors, 3);
        assert_eq!(s.transparent_pixels, 1);
        assert_eq!(s.palette_violations, Some(1));
        assert_eq!(s.top_colors[0], ("#0a141e".to_string(), 14));
        assert_eq!(stats(&t, None).palette_violations, None);
    }
}
