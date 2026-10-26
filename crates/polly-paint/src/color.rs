//! Colour values and parsing.

use polly_core::Result;
use polly_core::error::invalid;

/// Straight (non-premultiplied) RGBA8 colour.
pub type Rgba = [u8; 4];

pub const TRANSPARENT: Rgba = [0, 0, 0, 0];

const NAMED: &[(&str, Rgba)] = &[
    ("black", [0, 0, 0, 255]),
    ("white", [255, 255, 255, 255]),
    ("red", [255, 0, 0, 255]),
    ("green", [0, 128, 0, 255]),
    ("lime", [0, 255, 0, 255]),
    ("blue", [0, 0, 255, 255]),
    ("yellow", [255, 255, 0, 255]),
    ("cyan", [0, 255, 255, 255]),
    ("magenta", [255, 0, 255, 255]),
    ("gray", [128, 128, 128, 255]),
    ("grey", [128, 128, 128, 255]),
    ("darkgray", [64, 64, 64, 255]),
    ("lightgray", [192, 192, 192, 255]),
    ("orange", [255, 165, 0, 255]),
    ("purple", [128, 0, 128, 255]),
    ("brown", [139, 69, 19, 255]),
    ("pink", [255, 192, 203, 255]),
    ("navy", [0, 0, 128, 255]),
    ("teal", [0, 128, 128, 255]),
    ("olive", [128, 128, 0, 255]),
    ("maroon", [128, 0, 0, 255]),
    ("transparent", TRANSPARENT),
];

/// Parse a colour: `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa` (the `#` is optional),
/// a CSS-ish name (`black`, `white`, `red`, ..., `transparent`), or a palette index
/// `p3` / `palette:3` (requires `palette`).
pub fn parse_color(s: &str, palette: Option<&[Rgba]>) -> Result<Rgba> {
    let t = s.trim().to_ascii_lowercase();
    if let Some((_, c)) = NAMED.iter().find(|(n, _)| *n == t) {
        return Ok(*c);
    }
    let index = t.strip_prefix("palette:").or_else(|| t.strip_prefix('p')).and_then(|n| n.trim().parse::<usize>().ok());
    if let Some(i) = index {
        let Some(pal) = palette else {
            return invalid(format!("'{s}' is a palette index but no palette is set"));
        };
        return pal
            .get(i)
            .copied()
            .ok_or_else(|| polly_core::Error::Invalid(format!("palette index {i} out of range (palette has {})", pal.len())));
    }
    let hex = t.strip_prefix('#').unwrap_or(&t);
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return invalid(format!("cannot parse colour '{s}' (use #rrggbb, #rrggbbaa, a name, or p<index>)"));
    }
    let nib = |i: usize| u8::from_str_radix(&hex[i..i + 1], 16).unwrap() * 17;
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
    Ok(match hex.len() {
        3 => [nib(0), nib(1), nib(2), 255],
        4 => [nib(0), nib(1), nib(2), nib(3)],
        6 => [byte(0), byte(2), byte(4), 255],
        8 => [byte(0), byte(2), byte(4), byte(6)],
        _ => return invalid(format!("cannot parse colour '{s}' (use #rrggbb, #rrggbbaa, a name, or p<index>)")),
    })
}

/// `#rrggbb`, or `#rrggbbaa` when not fully opaque.
pub fn to_hex(c: Rgba) -> String {
    if c[3] == 255 {
        format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
    } else {
        format!("#{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3])
    }
}

/// Convert a 24-bit `0xRRGGBB` value to an opaque colour.
pub const fn rgb(hex: u32) -> Rgba {
    [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8, 255]
}

/// Linear interpolation per channel, `t` in 0..=1.
pub fn lerp(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    std::array::from_fn(|i| (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).round() as u8)
}

/// Convert a float colour (0..1 per channel, e.g. a vertex colour) to RGBA8.
pub fn from_f32(c: [f32; 4]) -> Rgba {
    c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// RGB -> (hue degrees 0..360, saturation 0..1, value 0..1).
pub fn to_hsv(c: Rgba) -> [f32; 3] {
    let [r, g, b] = [c[0], c[1], c[2]].map(|v| v as f32 / 255.0);
    let max = r.max(g).max(b);
    let d = max - r.min(g).min(b);
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    [h, if max == 0.0 { 0.0 } else { d / max }, max]
}

/// Inverse of [`to_hsv`]; alpha is taken from `alpha`.
pub fn from_hsv([h, s, v]: [f32; 3], alpha: u8) -> Rgba {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let f = |u: f32| ((u + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    [f(r), f(g), f(b), alpha]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        assert_eq!(parse_color("#fff", None).unwrap(), [255, 255, 255, 255]);
        assert_eq!(parse_color("#102030", None).unwrap(), [16, 32, 48, 255]);
        assert_eq!(parse_color("10203040", None).unwrap(), [16, 32, 48, 64]);
        assert_eq!(parse_color(" Red ", None).unwrap(), [255, 0, 0, 255]);
        assert_eq!(parse_color("transparent", None).unwrap(), [0, 0, 0, 0]);
        let pal = [[1, 2, 3, 255], [4, 5, 6, 255]];
        assert_eq!(parse_color("p1", Some(&pal)).unwrap(), [4, 5, 6, 255]);
        assert_eq!(parse_color("palette:0", Some(&pal)).unwrap(), [1, 2, 3, 255]);
        assert!(parse_color("p2", Some(&pal)).is_err());
        assert!(parse_color("p0", None).is_err());
        assert!(parse_color("#12345", None).is_err());
        assert!(parse_color("nope", None).is_err());
        assert_eq!(to_hex([16, 32, 48, 255]), "#102030");
        assert_eq!(to_hex([16, 32, 48, 64]), "#10203040");
    }

    #[test]
    fn hsv_roundtrip() {
        for c in [[255, 0, 0, 255], [12, 200, 99, 255], [7, 7, 7, 255], [30, 60, 250, 9]] {
            assert_eq!(from_hsv(to_hsv(c), c[3]), c);
        }
    }
}
