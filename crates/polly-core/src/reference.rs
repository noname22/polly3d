//! Reference images: pictures shown behind or over the model in one of the
//! axis views (front, back, left, right, top, bottom), placed in world units
//! so they zoom with the model.

use std::sync::Arc;

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::error::{Result, invalid};
use crate::scene::Scene;

/// The orthographic view a reference is shown in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RefView {
    #[default]
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
}

impl RefView {
    pub const ALL: [RefView; 6] = [RefView::Front, RefView::Back, RefView::Left, RefView::Right, RefView::Top, RefView::Bottom];

    /// World directions of the view's screen right (u) and screen up (v),
    /// as in the render views (a back image reads as drawn, not mirrored).
    pub fn axes(self) -> (Vec3, Vec3) {
        match self {
            RefView::Front => (Vec3::X, Vec3::Y),
            RefView::Back => (Vec3::NEG_X, Vec3::Y),
            RefView::Left => (Vec3::Z, Vec3::Y),
            RefView::Right => (Vec3::NEG_Z, Vec3::Y),
            RefView::Top => (Vec3::X, Vec3::NEG_Z),
            RefView::Bottom => (Vec3::X, Vec3::Z),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    /// Under the model (and grid/wireframe).
    #[default]
    Behind,
    /// Over everything, blended by its opacity.
    Front,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub name: String,
    pub view: RefView,
    /// Source image size in pixels.
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, top row first. Never changed after import.
    #[serde(with = "crate::scene::b64")]
    pub pixels: Arc<Vec<u8>>,
    /// Part of the image to show: [x, y, w, h] in source pixels; `None` = all.
    #[serde(default)]
    pub crop: Option<[u32; 4]>,
    /// Centre of the shown image in the view's screen axes (world units).
    pub center: [f32; 2],
    /// Height of the shown image in world units; the width follows its aspect ratio.
    pub size: f32,
    #[serde(default)]
    pub layer: Layer,
    pub opacity: f32,
    pub visible: bool,
    /// Mirror the image left/right (e.g. a side sheet reused for the other side).
    #[serde(default)]
    pub flip: bool,
}

impl Reference {
    /// The crop rect, or the whole image.
    pub fn rect(&self) -> [u32; 4] {
        self.crop.unwrap_or([0, 0, self.width, self.height])
    }

    /// World units per source pixel.
    pub fn scale(&self) -> f32 {
        self.size / self.rect()[3].max(1) as f32
    }

    /// Width and height in world units.
    pub fn extent(&self) -> Vec2 {
        let r = self.rect();
        Vec2::new(r[2] as f32, r[3] as f32) * self.scale()
    }

    /// World corners (top-left, top-right, bottom-right, bottom-left) and
    /// their UVs (0,0 = top-left of the source image).
    pub fn quad(&self) -> [(Vec3, [f32; 2]); 4] {
        let (u, v) = self.view.axes();
        let c = u * self.center[0] + v * self.center[1];
        let h = self.extent() * 0.5;
        let r = self.rect();
        let (w, ht) = (self.width.max(1) as f32, self.height.max(1) as f32);
        let (mut u0, v0) = (r[0] as f32 / w, r[1] as f32 / ht);
        let (mut u1, v1) = ((r[0] + r[2]) as f32 / w, (r[1] + r[3]) as f32 / ht);
        if self.flip {
            std::mem::swap(&mut u0, &mut u1);
        }
        [
            (c - u * h.x + v * h.y, [u0, v0]),
            (c + u * h.x + v * h.y, [u1, v0]),
            (c + u * h.x - v * h.y, [u1, v1]),
            (c - u * h.x - v * h.y, [u0, v1]),
        ]
    }

    /// Source pixel (x, y; image coordinates, top-left origin) -> view
    /// coordinates (u, v) in world units.
    pub fn pixel_to_view(&self, px: f32, py: f32) -> Vec2 {
        let r = self.rect();
        let (cx, cy) = (r[0] as f32 + r[2] as f32 / 2.0, r[1] as f32 + r[3] as f32 / 2.0);
        let sx = if self.flip { -1.0 } else { 1.0 };
        let k = self.scale();
        Vec2::new(self.center[0] + sx * (px - cx) * k, self.center[1] - (py - cy) * k)
    }

    /// View coordinates (u, v) -> source pixel (x, y).
    pub fn view_to_pixel(&self, uv: Vec2) -> Vec2 {
        let r = self.rect();
        let (cx, cy) = (r[0] as f32 + r[2] as f32 / 2.0, r[1] as f32 + r[3] as f32 / 2.0);
        let sx = if self.flip { -1.0 } else { 1.0 };
        let k = self.scale().max(1e-12);
        Vec2::new(cx + sx * (uv.x - self.center[0]) / k, cy - (uv.y - self.center[1]) / k)
    }

    /// Place the image so two source pixels land on two view points (u, v):
    /// sets `size` (uniform scale from their distance) and `center`.
    pub fn calibrate(&mut self, a: (Vec2, Vec2), b: (Vec2, Vec2)) -> Result<()> {
        let (dp, dw) = (a.0.distance(b.0), a.1.distance(b.1));
        if dp < 1e-3 || dw < 1e-6 {
            return invalid("calibration points must be apart, both in the image and in the world");
        }
        let k = dw / dp;
        self.size = self.rect()[3].max(1) as f32 * k;
        // Solve the centre from point a.
        self.center = [0.0, 0.0];
        let at_origin = self.pixel_to_view(a.0.x, a.0.y);
        self.center = (a.1 - at_origin).to_array();
        Ok(())
    }

    fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        self.pixels[i..i + 4].try_into().unwrap()
    }

    /// Background colour: the median of the crop's border pixels.
    pub fn background(&self) -> [u8; 4] {
        let [x0, y0, w, h] = self.rect();
        let mut border = Vec::new();
        for x in x0..x0 + w {
            border.push(self.pixel(x, y0));
            border.push(self.pixel(x, y0 + h - 1));
        }
        for y in y0..y0 + h {
            border.push(self.pixel(x0, y));
            border.push(self.pixel(x0 + w - 1, y));
        }
        let mut out = [0u8; 4];
        for (c, o) in out.iter_mut().enumerate() {
            let mut ch: Vec<u8> = border.iter().map(|p| p[c]).collect();
            ch.sort_unstable();
            *o = ch[ch.len() / 2];
        }
        out
    }

    /// Is this pixel part of the drawing: opaque and clearly unlike the background?
    pub fn is_ink(&self, x: u32, y: u32, bg: [u8; 4], threshold: f32) -> bool {
        let p = self.pixel(x, y);
        if p[3] < 128 {
            return false;
        }
        if bg[3] < 128 {
            return true; // cut-out image: everything opaque is the subject
        }
        let d = (0..3).map(|c| (p[c] as f32 - bg[c] as f32).powi(2)).sum::<f32>().sqrt();
        d > threshold
    }

    /// Outermost ink on a source row (within the crop): (first x, last x).
    pub fn ink_span(&self, y: u32, bg: [u8; 4], threshold: f32) -> Option<(u32, u32)> {
        let [x0, y0, w, h] = self.rect();
        if y < y0 || y >= y0 + h {
            return None;
        }
        let first = (x0..x0 + w).find(|x| self.is_ink(*x, y, bg, threshold))?;
        let last = (x0..x0 + w).rev().find(|x| self.is_ink(*x, y, bg, threshold))?;
        Some((first, last))
    }

    /// The drawing's outer edges at view height `v`, in view units (u_min, u_max).
    pub fn edges_at(&self, v: f32, bg: [u8; 4], threshold: f32) -> Option<(f32, f32)> {
        let py = self.view_to_pixel(Vec2::new(self.center[0], v)).y.floor();
        if py < 0.0 {
            return None;
        }
        let (a, b) = self.ink_span(py as u32, bg, threshold)?;
        // Pixel edges: the outer sides of the outermost ink pixels.
        let (ua, ub) = (self.pixel_to_view(a as f32, py).x, self.pixel_to_view(b as f32 + 1.0, py).x);
        Some((ua.min(ub), ua.max(ub)))
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return invalid("reference name must not be empty");
        }
        if let Some([x, y, w, h]) = self.crop
            && (w == 0 || h == 0 || x + w > self.width || y + h > self.height)
        {
            return invalid(format!("crop [{x},{y},{w},{h}] must be non-empty and inside the {}x{} image", self.width, self.height));
        }
        if !(self.size.is_finite() && self.size > 0.0) {
            return invalid("size must be > 0");
        }
        if !(0.0..=1.0).contains(&self.opacity) {
            return invalid("opacity must be in 0..1");
        }
        Ok(())
    }
}

impl Scene {
    pub fn reference(&self, name: &str) -> Result<&Reference> {
        self.references.iter().find(|r| r.name == name).ok_or_else(|| self.no_reference(name))
    }

    pub fn reference_mut(&mut self, name: &str) -> Result<&mut Reference> {
        let e = self.no_reference(name);
        self.references.iter_mut().find(|r| r.name == name).ok_or(e)
    }

    fn no_reference(&self, name: &str) -> crate::Error {
        let names: Vec<&str> = self.references.iter().map(|r| r.name.as_str()).collect();
        crate::Error::Invalid(format!(
            "no reference '{name}' (references: {})",
            if names.is_empty() { "none".into() } else { names.join(", ") }
        ))
    }

    /// Add a reference, replacing one with the same name.
    pub fn set_reference(&mut self, r: Reference) -> Result<()> {
        r.validate()?;
        match self.references.iter_mut().find(|x| x.name == r.name) {
            Some(x) => *x = r,
            None => self.references.push(r),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(view: RefView) -> Reference {
        Reference {
            name: "sheet".into(),
            view,
            width: 200,
            height: 100,
            pixels: Arc::new(vec![0; 200 * 100 * 4]),
            crop: Some([100, 0, 100, 100]),
            center: [1.0, 2.0],
            size: 4.0,
            layer: Layer::Behind,
            opacity: 0.5,
            visible: true,
            flip: false,
        }
    }

    #[test]
    fn quad_is_placed_on_the_view_plane() {
        // front: u = +X, v = +Y; a square crop 4 units high.
        let q = reference(RefView::Front).quad();
        assert_eq!(q[0].0, Vec3::new(-1.0, 4.0, 0.0));
        assert_eq!(q[2].0, Vec3::new(3.0, 0.0, 0.0));
        assert_eq!((q[0].1, q[2].1), ([0.5, 0.0], [1.0, 1.0]));
        // right: u = -Z; top: v = -Z; back: u = -X; left: u = +Z; bottom: v = +Z.
        let q = reference(RefView::Right).quad();
        assert_eq!(q[0].0, Vec3::new(0.0, 4.0, 1.0));
        let q = reference(RefView::Top).quad();
        assert_eq!(q[0].0, Vec3::new(-1.0, 0.0, -4.0));
        assert_eq!(reference(RefView::Back).quad()[0].0, Vec3::new(1.0, 4.0, 0.0));
        assert_eq!(reference(RefView::Left).quad()[0].0, Vec3::new(0.0, 4.0, -1.0));
        assert_eq!(reference(RefView::Bottom).quad()[0].0, Vec3::new(-1.0, 0.0, 4.0));
        // whole image: the width follows the aspect ratio.
        let mut r = reference(RefView::Front);
        r.crop = None;
        assert_eq!(r.extent(), Vec2::new(8.0, 4.0));
    }

    #[test]
    fn pixel_mapping_calibration_and_ink() {
        // 100x200 white image with a black bar in columns 40..60, rows 20..180.
        let mut px = vec![255u8; 100 * 200 * 4];
        for y in 20..180 {
            for x in 40..60 {
                let i = (y * 100 + x) * 4;
                px[i..i + 3].copy_from_slice(&[0, 0, 0]);
            }
        }
        let mut r = Reference {
            name: "bar".into(),
            view: RefView::Front,
            width: 100,
            height: 200,
            pixels: Arc::new(px),
            crop: None,
            center: [0.0, 1.0],
            size: 2.0,
            layer: Layer::Behind,
            opacity: 0.5,
            visible: true,
            flip: false,
        };
        // Row 180 (bottom of the bar) is the ground, row 20 is 1.6 m up; column 50 is x = 0.
        r.calibrate((Vec2::new(50.0, 180.0), Vec2::new(0.0, 0.0)), (Vec2::new(50.0, 20.0), Vec2::new(0.0, 1.6))).unwrap();
        assert!((r.scale() - 0.01).abs() < 1e-6);
        assert!(r.pixel_to_view(50.0, 180.0).abs_diff_eq(Vec2::ZERO, 1e-5));
        assert!(r.view_to_pixel(Vec2::new(0.0, 1.6)).abs_diff_eq(Vec2::new(50.0, 20.0), 1e-3));
        // The bar is 20 px = 0.2 m wide, centred on x = 0.
        let bg = r.background();
        assert_eq!(bg, [255, 255, 255, 255]);
        let (a, b) = r.edges_at(0.8, bg, 60.0).unwrap();
        assert!((a + 0.1).abs() < 1e-4 && (b - 0.1).abs() < 1e-4, "{a} {b}");
        assert!(r.edges_at(1.7, bg, 60.0).is_none(), "above the bar");
        // Flipped, the mapping mirrors about the centre.
        r.flip = true;
        r.calibrate((Vec2::new(40.0, 180.0), Vec2::new(0.0, 0.0)), (Vec2::new(40.0, 20.0), Vec2::new(0.0, 1.6))).unwrap();
        assert!(r.pixel_to_view(60.0, 100.0).x < 0.0, "flipped: right in the image is left in the view");
    }

    #[test]
    fn validation_and_replacement() {
        let mut s = Scene::default();
        let mut r = reference(RefView::Front);
        r.crop = Some([150, 0, 100, 100]);
        assert!(s.set_reference(r.clone()).is_err());
        r.crop = None;
        s.set_reference(r.clone()).unwrap();
        r.opacity = 1.0;
        s.set_reference(r).unwrap();
        assert_eq!(s.references.len(), 1);
        assert_eq!(s.reference("sheet").unwrap().opacity, 1.0);
        assert!(s.reference("x").unwrap_err().to_string().contains("sheet"));
        // survives a serde round trip (project files, undo is a clone)
        let back: Scene = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back.references[0].pixels.len(), 200 * 100 * 4);
    }
}
