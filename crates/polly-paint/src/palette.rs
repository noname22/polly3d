//! Built-in retro palettes, nearest-colour matching, ramps and palette extraction.

use std::collections::HashMap;

use polly_core::scene::Texture;

use crate::color::{Rgba, lerp, rgb};

const PICO8: &[u32] = &[
    0x000000, 0x1d2b53, 0x7e2553, 0x008751, 0xab5236, 0x5f574f, 0xc2c3c7, 0xfff1e8, 0xff004d, 0xffa300, 0xffec27,
    0x00e436, 0x29adff, 0x83769c, 0xff77a8, 0xffccaa,
];

const DB16: &[u32] = &[
    0x140c1c, 0x442434, 0x30346d, 0x4e4a4e, 0x854c30, 0x346524, 0xd04648, 0x757161, 0x597dce, 0xd27d2c, 0x8595a1,
    0x6daa2c, 0xd2aa99, 0x6dc2ca, 0xdad45e, 0xdeeed6,
];

const DB32: &[u32] = &[
    0x000000, 0x222034, 0x45283c, 0x663931, 0x8f563b, 0xdf7126, 0xd9a066, 0xeec39a, 0xfbf236, 0x99e550, 0x6abe30,
    0x37946e, 0x4b692f, 0x524b24, 0x323c39, 0x3f3f74, 0x306082, 0x5b6ee1, 0x639bff, 0x5fcde4, 0xcbdbfc, 0xffffff,
    0x9badb7, 0x847e87, 0x696a6a, 0x595652, 0x76428a, 0xac3232, 0xd95763, 0xd77bba, 0x8f974a, 0x8a6f30,
];

const ENDESGA32: &[u32] = &[
    0xbe4a2f, 0xd77643, 0xead4aa, 0xe4a672, 0xb86f50, 0x733e39, 0x3e2731, 0xa22633, 0xe43b44, 0xf77622, 0xfeae34,
    0xfee761, 0x63c74d, 0x3e8948, 0x265c42, 0x193c3e, 0x124e89, 0x0099db, 0x2ce8f5, 0xffffff, 0xc0cbdc, 0x8b9bb4,
    0x5a6988, 0x3a4466, 0x262b44, 0x181425, 0xff0044, 0x68386c, 0xb55088, 0xf6757a, 0xe8b796, 0xc28569,
];

/// NES 2C02 approximation (the common "FCEUX-ish" table), duplicates and
/// redundant blacks removed.
const NES: &[u32] = &[
    0x7c7c7c, 0x0000fc, 0x0000bc, 0x4428bc, 0x940084, 0xa80020, 0xa81000, 0x881400, 0x503000, 0x007800, 0x006800,
    0x005800, 0x004058, 0x000000, 0xbcbcbc, 0x0078f8, 0x0058f8, 0x6844fc, 0xd800cc, 0xe40058, 0xf83800, 0xe45c10,
    0xac7c00, 0x00b800, 0x00a800, 0x00a844, 0x008888, 0xf8f8f8, 0x3cbcfc, 0x6888fc, 0x9878f8, 0xf878f8, 0xf85898,
    0xf87858, 0xfca044, 0xf8b800, 0xb8f818, 0x58d854, 0x58f898, 0x00e8d8, 0x787878, 0xfcfcfc, 0xa4e4fc, 0xb8b8f8,
    0xd8b8f8, 0xf8b8f8, 0xf8a4c0, 0xf0d0b0, 0xfce0a8, 0xf8d878, 0xd8f878, 0xb8f8b8, 0xb8f8d8, 0x00fcfc, 0xf8d8f8,
];

const GAMEBOY: &[u32] = &[0x0f380f, 0x306230, 0x8bac0f, 0x9bbc0f];

const SWEETIE16: &[u32] = &[
    0x1a1c2c, 0x5d275d, 0xb13e53, 0xef7d57, 0xffcd75, 0xa7f070, 0x38b764, 0x257179, 0x29366f, 0x3b5dc9, 0x41a6f6,
    0x73eff7, 0xf4f4f4, 0x94b0c2, 0x566c86, 0x333c57,
];

const NAMES: &[&str] = &["pico8", "db16", "db32", "endesga32", "nes", "gameboy", "grayscale8", "sweetie16"];

/// Names accepted by [`builtin`].
pub fn builtin_names() -> &'static [&'static str] {
    NAMES
}

/// A built-in palette by (case-insensitive) name; `-`/`_` are ignored, so
/// `PICO-8` and `endesga_32` work too.
pub fn builtin(name: &str) -> Option<Vec<Rgba>> {
    let key: String = name.chars().filter(|c| !matches!(c, '-' | '_' | ' ')).collect::<String>().to_ascii_lowercase();
    let hex = match key.as_str() {
        "pico8" => PICO8,
        "db16" | "dawnbringer16" => DB16,
        "db32" | "dawnbringer32" => DB32,
        "endesga32" => ENDESGA32,
        "nes" => NES,
        "gameboy" | "gb" | "dmg" => GAMEBOY,
        "sweetie16" => SWEETIE16,
        "grayscale8" | "greyscale8" | "gray8" | "grey8" => {
            return Some((0..8).map(|i| [(i * 255 / 7) as u8; 3]).map(|[r, g, b]| [r, g, b, 255]).collect());
        }
        _ => return None,
    };
    Some(hex.iter().map(|&h| rgb(h)).collect())
}

/// Perceptually weighted ("redmean") squared RGB distance.
pub fn distance(a: Rgba, b: Rgba) -> f32 {
    let rm = (a[0] as f32 + b[0] as f32) * 0.5;
    let [dr, dg, db] = [0, 1, 2].map(|i| a[i] as f32 - b[i] as f32);
    (2.0 + rm / 256.0) * dr * dr + 4.0 * dg * dg + (2.0 + (255.0 - rm) / 256.0) * db * db
}

/// Index of the nearest palette entry (by RGB, ignoring fully transparent entries
/// unless the palette has nothing else) and the resulting colour. The result keeps
/// the input's alpha, so transparent pixels stay transparent. An empty palette
/// returns `(0, c)`.
pub fn nearest(palette: &[Rgba], c: Rgba) -> (usize, Rgba) {
    let opaque_only = palette.iter().any(|p| p[3] > 0);
    let best = palette
        .iter()
        .enumerate()
        .filter(|(_, p)| !opaque_only || p[3] > 0)
        .min_by(|(_, a), (_, b)| distance(**a, c).total_cmp(&distance(**b, c)));
    match best {
        Some((i, p)) => (i, [p[0], p[1], p[2], c[3]]),
        None => (0, c),
    }
}

/// `steps` colours evenly interpolated from `from` to `to` (inclusive).
pub fn ramp(from: Rgba, to: Rgba, steps: usize) -> Vec<Rgba> {
    match steps {
        0 => vec![],
        1 => vec![from],
        n => (0..n).map(|i| lerp(from, to, i as f32 / (n - 1) as f32)).collect(),
    }
}

/// Extract up to `max_colors` representative opaque colours (alpha > 0) using
/// median cut. If the texture already has few enough colours they are returned
/// exactly, most frequent first.
pub fn extract(tex: &Texture, max_colors: usize) -> Vec<Rgba> {
    let mut hist: HashMap<Rgba, usize> = HashMap::new();
    for p in tex.pixels.chunks_exact(4) {
        if p[3] > 0 {
            *hist.entry([p[0], p[1], p[2], 255]).or_default() += 1;
        }
    }
    let mut colors: Vec<(Rgba, usize)> = hist.into_iter().collect();
    colors.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if max_colors == 0 {
        return vec![];
    }
    if colors.len() <= max_colors {
        return colors.into_iter().map(|(c, _)| c).collect();
    }
    // Median cut over the weighted histogram.
    let mut boxes = vec![colors];
    while boxes.len() < max_colors {
        let range = |b: &Vec<(Rgba, usize)>, ch: usize| {
            let (lo, hi) = b.iter().fold((255u8, 0u8), |(lo, hi), (c, _)| (lo.min(c[ch]), hi.max(c[ch])));
            hi.saturating_sub(lo)
        };
        let Some((bi, ch, _)) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .flat_map(|(i, b)| (0..3).map(move |ch| (i, ch, range(b, ch))))
            .max_by_key(|&(_, _, r)| r)
        else {
            break;
        };
        let mut b = boxes.swap_remove(bi);
        b.sort_by_key(|(c, _)| c[ch]);
        let total: usize = b.iter().map(|(_, n)| n).sum();
        let mut acc = 0;
        let mut cut = b.len() / 2;
        for (i, (_, n)) in b.iter().enumerate() {
            acc += n;
            if acc * 2 >= total {
                cut = (i + 1).clamp(1, b.len() - 1);
                break;
            }
        }
        let rest = b.split_off(cut);
        boxes.push(b);
        boxes.push(rest);
    }
    let mut out: Vec<(Rgba, usize)> = boxes
        .iter()
        .map(|b| {
            let n: usize = b.iter().map(|(_, n)| n).sum();
            let avg = |ch: usize| (b.iter().map(|(c, k)| c[ch] as usize * k).sum::<usize>() as f32 / n as f32).round() as u8;
            ([avg(0), avg(1), avg(2), 255], n)
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out.into_iter().map(|(c, _)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_counts() {
        let expect = [
            ("pico8", 16),
            ("db16", 16),
            ("db32", 32),
            ("endesga32", 32),
            ("gameboy", 4),
            ("grayscale8", 8),
            ("sweetie16", 16),
        ];
        for (n, k) in expect {
            assert_eq!(builtin(n).unwrap().len(), k, "{n}");
        }
        let nes = builtin("NES").unwrap();
        assert!((50..=56).contains(&nes.len()));
        for n in builtin_names() {
            let p = builtin(n).unwrap();
            let mut u = p.clone();
            u.sort();
            u.dedup();
            assert_eq!(u.len(), p.len(), "{n} has duplicates");
        }
        assert_eq!(builtin("PICO-8").unwrap()[8], [0xff, 0x00, 0x4d, 255]);
        assert!(builtin("nope").is_none());
    }

    #[test]
    fn nearest_and_ramp() {
        let p = builtin("gameboy").unwrap();
        assert_eq!(nearest(&p, [10, 50, 10, 255]).0, 0);
        assert_eq!(nearest(&p, [160, 190, 20, 255]).0, 3);
        assert_eq!(nearest(&p, [160, 190, 20, 0]).1[3], 0);
        let r = ramp([0, 0, 0, 255], [255, 255, 255, 255], 3);
        assert_eq!(r, vec![[0, 0, 0, 255], [128, 128, 128, 255], [255, 255, 255, 255]]);
    }

    #[test]
    fn extract_colors() {
        let mut t = Texture::new("t", 4, 4, [255, 0, 0, 255]);
        t.pixels_mut()[0..4].copy_from_slice(&[0, 0, 255, 255]);
        assert_eq!(extract(&t, 8), vec![[255, 0, 0, 255], [0, 0, 255, 255]]);
        let mut g = Texture::new("g", 16, 16, [0; 4]);
        for (i, p) in g.pixels_mut().chunks_exact_mut(4).enumerate() {
            p.copy_from_slice(&[i as u8, i as u8, i as u8, 255]);
        }
        assert_eq!(extract(&g, 4).len(), 4);
    }
}
