//! Tiny CPU 2-D drawing helpers (labels, legend, UV layouts) on `RgbaImage`.

use image::RgbaImage;

pub type Rgba = [u8; 4];

pub const WHITE: Rgba = [255, 255, 255, 255];
pub const OUTLINE: Rgba = [10, 10, 14, 255];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub fn overlaps(&self, o: &Rect) -> bool {
        self.x < o.x + o.w && o.x < self.x + self.w && self.y < o.y + o.h && o.y < self.y + self.h
    }
    pub fn inside(&self, w: u32, h: u32) -> bool {
        self.x >= 0 && self.y >= 0 && self.x + self.w <= w as i32 && self.y + self.h <= h as i32
    }
}

#[inline]
pub fn blend(img: &mut RgbaImage, x: i32, y: i32, c: Rgba, cov: f32) {
    if x < 0 || y < 0 || x >= img.width() as i32 || y >= img.height() as i32 {
        return;
    }
    let a = (c[3] as f32 / 255.0) * cov.clamp(0.0, 1.0);
    if a <= 0.0 {
        return;
    }
    let p = img.get_pixel_mut(x as u32, y as u32);
    for (d, s) in p.0.iter_mut().zip(c).take(3) {
        *d = (*d as f32 * (1.0 - a) + s as f32 * a).round() as u8;
    }
    p.0[3] = (p.0[3] as f32 + (255.0 - p.0[3] as f32) * a).round() as u8;
}

pub fn fill_rect(img: &mut RgbaImage, r: Rect, c: Rgba) {
    for y in r.y..r.y + r.h {
        for x in r.x..r.x + r.w {
            blend(img, x, y, c, 1.0);
        }
    }
}

pub fn stroke_rect(img: &mut RgbaImage, r: Rect, c: Rgba) {
    for x in r.x..r.x + r.w {
        blend(img, x, r.y, c, 1.0);
        blend(img, x, r.y + r.h - 1, c, 1.0);
    }
    for y in r.y + 1..r.y + r.h - 1 {
        blend(img, r.x, y, c, 1.0);
        blend(img, r.x + r.w - 1, y, c, 1.0);
    }
}

/// Anti-aliased line of the given width (pixels). Coordinates are continuous
/// (pixel centres at +0.5).
pub fn line(img: &mut RgbaImage, a: [f32; 2], b: [f32; 2], c: Rgba, width: f32) {
    let (ax, ay, bx, by) = (a[0], a[1], b[0], b[1]);
    if !(ax.is_finite() && ay.is_finite() && bx.is_finite() && by.is_finite()) {
        return;
    }
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let hw = width * 0.5;
    let reach = (hw + 1.0).ceil() as i32;
    let cover = |px: i32, py: i32| -> f32 {
        let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
        let t = if len2 > 0.0 { (((cx - ax) * dx + (cy - ay) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let (qx, qy) = (ax + dx * t - cx, ay + dy * t - cy);
        (hw + 0.5 - (qx * qx + qy * qy).sqrt()).clamp(0.0, 1.0)
    };
    // Clip the walk to the image (with margin) to keep huge off-screen lines cheap.
    let (w, h) = (img.width() as f32, img.height() as f32);
    let seen = |img: &mut RgbaImage, px: i32, py: i32| {
        let cv = cover(px, py);
        if cv > 0.0 {
            blend(img, px, py, c, cv);
        }
    };
    if dx.abs() >= dy.abs() {
        let (x0, x1) = (ax.min(bx).max(-2.0), ax.max(bx).min(w + 2.0));
        let mut x = x0.floor() as i32 - reach;
        while x <= x1.ceil() as i32 + reach {
            let t = if dx.abs() > 1e-9 { ((x as f32 + 0.5 - ax) / dx).clamp(0.0, 1.0) } else { 0.0 };
            let yc = (ay + dy * t).floor() as i32;
            for y in yc - reach..=yc + reach {
                seen(img, x, y);
            }
            x += 1;
        }
    } else {
        let (y0, y1) = (ay.min(by).max(-2.0), ay.max(by).min(h + 2.0));
        let mut y = y0.floor() as i32 - reach;
        while y <= y1.ceil() as i32 + reach {
            let t = if dy.abs() > 1e-9 { ((y as f32 + 0.5 - ay) / dy).clamp(0.0, 1.0) } else { 0.0 };
            let xc = (ax + dx * t).floor() as i32;
            for x in xc - reach..=xc + reach {
                seen(img, x, y);
            }
            y += 1;
        }
    }
}

pub fn disc(img: &mut RgbaImage, c: [f32; 2], r: f32, col: Rgba) {
    let reach = r.ceil() as i32 + 1;
    let (cx, cy) = (c[0].floor() as i32, c[1].floor() as i32);
    for y in cy - reach..=cy + reach {
        for x in cx - reach..=cx + reach {
            let d = ((x as f32 + 0.5 - c[0]).powi(2) + (y as f32 + 0.5 - c[1]).powi(2)).sqrt();
            blend(img, x, y, col, r + 0.5 - d);
        }
    }
}

fn glyph(ch: char) -> [u8; 8] {
    let i = ch as usize;
    if i < 128 { font8x8::legacy::BASIC_LEGACY[i] } else { font8x8::legacy::BASIC_LEGACY[b'?' as usize] }
}

/// Glyph advance in font pixels. The 8x8 font leaves the rightmost column
/// mostly blank, so 7 keeps text compact while still readable.
const ADVANCE: i32 = 7;

pub fn text_size(s: &str, scale: u32) -> (i32, i32) {
    let n = s.chars().count() as i32;
    ((n * ADVANCE + 1) * scale as i32, 8 * scale as i32)
}

/// Text with a 1px (scaled) dark outline, top-left at (x, y).
pub fn text(img: &mut RgbaImage, x: i32, y: i32, s: &str, scale: u32, color: Rgba, outline: Option<Rgba>) {
    let sc = scale as i32;
    let on = |gx: i32, gy: i32, g: &[u8; 8]| -> bool { (0..8).contains(&gx) && (0..8).contains(&gy) && (g[gy as usize] >> gx) & 1 == 1 };
    for (i, ch) in s.chars().enumerate() {
        let g = glyph(ch);
        let ox = x + i as i32 * ADVANCE * sc;
        if let Some(oc) = outline {
            for gy in -1..9 {
                for gx in -1..9 {
                    if on(gx, gy, &g) {
                        continue;
                    }
                    let near = (-1..=1).any(|dy| (-1..=1).any(|dx| on(gx + dx, gy + dy, &g)));
                    if near {
                        fill_rect(img, Rect { x: ox + gx * sc, y: y + gy * sc, w: sc, h: sc }, oc);
                    }
                }
            }
        }
        for gy in 0..8 {
            for gx in 0..8 {
                if on(gx, gy, &g) {
                    fill_rect(img, Rect { x: ox + gx * sc, y: y + gy * sc, w: sc, h: sc }, color);
                }
            }
        }
    }
}

/// Rectangle a boxed label centred at (cx, cy) would occupy.
pub fn label_rect(s: &str, scale: u32, cx: f32, cy: f32) -> Rect {
    let (tw, th) = text_size(s, scale);
    let pad = scale as i32 + 1;
    let w = tw + pad * 2;
    let h = th + pad * 2;
    Rect { x: (cx - w as f32 * 0.5).round() as i32, y: (cy - h as f32 * 0.5).round() as i32, w, h }
}

/// Draw a boxed label into `r` (from `label_rect`).
pub fn boxed_label(img: &mut RgbaImage, r: Rect, s: &str, scale: u32, bg: Rgba, fg: Rgba) {
    fill_rect(img, r, bg);
    stroke_rect(img, r, [0, 0, 0, 160]);
    let pad = scale as i32 + 1;
    text(img, r.x + pad, r.y + pad, s, scale, fg, Some(OUTLINE));
}
