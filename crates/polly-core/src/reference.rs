//! Reference images: pictures shown behind or over the model in one of the
//! orthographic views, placed in world units so they zoom with the model.

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
    Right,
    Top,
}

impl RefView {
    /// World directions of the view's screen right (u) and screen up (v).
    pub fn axes(self) -> (Vec3, Vec3) {
        match self {
            RefView::Front => (Vec3::X, Vec3::Y),
            RefView::Right => (Vec3::NEG_Z, Vec3::Y),
            RefView::Top => (Vec3::X, Vec3::NEG_Z),
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
        let (u0, v0) = (r[0] as f32 / w, r[1] as f32 / ht);
        let (u1, v1) = ((r[0] + r[2]) as f32 / w, (r[1] + r[3]) as f32 / ht);
        [
            (c - u * h.x + v * h.y, [u0, v0]),
            (c + u * h.x + v * h.y, [u1, v0]),
            (c + u * h.x - v * h.y, [u1, v1]),
            (c - u * h.x - v * h.y, [u0, v1]),
        ]
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
        }
    }

    #[test]
    fn quad_is_placed_on_the_view_plane() {
        // front: u = +X, v = +Y; a square crop 4 units high.
        let q = reference(RefView::Front).quad();
        assert_eq!(q[0].0, Vec3::new(-1.0, 4.0, 0.0));
        assert_eq!(q[2].0, Vec3::new(3.0, 0.0, 0.0));
        assert_eq!((q[0].1, q[2].1), ([0.5, 0.0], [1.0, 1.0]));
        // right: u = -Z; top: v = -Z.
        let q = reference(RefView::Right).quad();
        assert_eq!(q[0].0, Vec3::new(0.0, 4.0, 1.0));
        let q = reference(RefView::Top).quad();
        assert_eq!(q[0].0, Vec3::new(-1.0, 0.0, -4.0));
        // whole image: the width follows the aspect ratio.
        let mut r = reference(RefView::Front);
        r.crop = None;
        assert_eq!(r.extent(), Vec2::new(8.0, 4.0));
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
