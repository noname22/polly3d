//! Texel masks and paint regions.

use serde::{Deserialize, Serialize};

use crate::faces::face_mask;

/// A UV-space polygon (one face's corner UVs, in loop order). glTF convention:
/// (0,0) is the top-left of the image, v points down.
pub type FaceUv = Vec<[f32; 2]>;

/// Boolean per-texel coverage of a `width` x `height` texture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub bits: Vec<bool>,
}

impl Mask {
    /// All-false mask.
    pub fn empty(width: u32, height: u32) -> Self {
        Mask { width, height, bits: vec![false; (width * height) as usize] }
    }

    /// All-true mask.
    pub fn full(width: u32, height: u32) -> Self {
        Mask { width, height, bits: vec![true; (width * height) as usize] }
    }

    fn index(&self, x: i32, y: i32) -> Option<usize> {
        if x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height {
            Some((y as u32 * self.width + x as u32) as usize)
        } else {
            None
        }
    }

    /// Coverage at `(x, y)`; out of bounds is `false`.
    pub fn get(&self, x: i32, y: i32) -> bool {
        self.index(x, y).is_some_and(|i| self.bits[i])
    }

    /// Set coverage at `(x, y)`; out of bounds is ignored.
    pub fn set(&mut self, x: i32, y: i32, on: bool) {
        if let Some(i) = self.index(x, y) {
            self.bits[i] = on;
        }
    }

    /// Number of covered texels.
    pub fn count(&self) -> usize {
        self.bits.iter().filter(|b| **b).count()
    }

    /// Inclusive bounding box `(min, max)` of covered texels.
    pub fn bbox(&self) -> Option<([i32; 2], [i32; 2])> {
        let mut out: Option<([i32; 2], [i32; 2])> = None;
        for (x, y) in self.iter() {
            out = Some(match out {
                None => ([x, y], [x, y]),
                Some((lo, hi)) => ([lo[0].min(x), lo[1].min(y)], [hi[0].max(x), hi[1].max(y)]),
            });
        }
        out
    }

    /// Covered texel coordinates, row-major.
    pub fn iter(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        let w = self.width as usize;
        self.bits.iter().enumerate().filter(|(_, b)| **b).map(move |(i, _)| ((i % w) as i32, (i / w) as i32))
    }

    /// Grow by `px` texels (8-connected / square dilation).
    pub fn dilate(&mut self, px: u32) {
        for _ in 0..px {
            let src = self.clone();
            for y in 0..self.height as i32 {
                for x in 0..self.width as i32 {
                    if !src.get(x, y) && NEIGHBORS8.iter().any(|(dx, dy)| src.get(x + dx, y + dy)) {
                        self.set(x, y, true);
                    }
                }
            }
        }
    }

    /// Keep only texels also covered by `other`.
    pub fn intersect(&mut self, other: &Mask) {
        let w = self.width as i32;
        for (i, b) in self.bits.iter_mut().enumerate() {
            *b = *b && other.get(i as i32 % w, i as i32 / w);
        }
    }

    /// Add every texel covered by `other`.
    pub fn union(&mut self, other: &Mask) {
        let w = self.width as i32;
        for (i, b) in self.bits.iter_mut().enumerate() {
            *b = *b || other.get(i as i32 % w, i as i32 / w);
        }
    }
}

pub(crate) const NEIGHBORS4: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub(crate) const NEIGHBORS8: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Which texels an operation touches.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Region {
    /// The whole texture.
    #[default]
    All,
    /// Pixel rectangle, `min` and `max` inclusive (corners may be given in any order).
    Rect { min: [i32; 2], max: [i32; 2] },
    /// Texel-centre coverage of UV polygons, optionally grown by `grow` texels.
    Faces {
        polys: Vec<FaceUv>,
        #[serde(default)]
        grow: u32,
    },
}

impl Region {
    /// Rasterize the region for a `w` x `h` texture.
    pub fn mask(&self, w: u32, h: u32) -> Mask {
        match self {
            Region::All => Mask::full(w, h),
            Region::Rect { min, max } => {
                let mut m = Mask::empty(w, h);
                let (x0, x1) = (min[0].min(max[0]).max(0), min[0].max(max[0]).min(w as i32 - 1));
                let (y0, y1) = (min[1].min(max[1]).max(0), min[1].max(max[1]).min(h as i32 - 1));
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        m.set(x, y, true);
                    }
                }
                m
            }
            Region::Faces { polys, grow } => face_mask(w, h, polys, *grow),
        }
    }
}
