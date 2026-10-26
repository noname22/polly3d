//! UV tools for low-poly meshes: projections, seams, LSCM unwrapping,
//! pixel-aware packing and an agent-readable quality report.
//!
//! Convention (glTF): uv (0,0) is the top-left of the image and v grows
//! downward; texel (x, y) is centered at ((x+0.5)/w, (y+0.5)/h). Internally,
//! layouts are computed in "math" space (y up) and flipped when written.

mod chart;
mod pack;
mod project;
mod report;
mod seams;
mod unwrap;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use glam::Vec2;
use polly_core::Result;
use polly_core::error::invalid;
use polly_core::mesh::{FaceId, Mesh};
use serde::{Deserialize, Serialize};

pub use pack::{PackOptions, pack};
pub use project::{Projection, project};
pub use report::{IslandInfo, UvReport, report};
pub use seams::{auto_seams, islands, mark_seams, uv_islands};
pub use unwrap::{UnwrapMethod, UnwrapOptions, unwrap};

/// Affine edit of existing UVs: scale and rotate about `pivot`, then translate.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct TransformUv {
    /// Offset in uv units (1.0 = full texture width/height).
    pub translate: [f32; 2],
    pub scale: [f32; 2],
    /// Counter-clockwise as seen on the texture image.
    pub rotate_deg: f32,
    /// Defaults to the center of the faces' uv bounding box.
    pub pivot: Option<[f32; 2]>,
}

impl Default for TransformUv {
    fn default() -> Self {
        TransformUv { translate: [0.0; 2], scale: [1.0; 2], rotate_deg: 0.0, pivot: None }
    }
}

/// Apply `t` to every corner uv of `faces`.
pub fn transform_uvs(mesh: &mut Mesh, faces: &BTreeSet<FaceId>, t: &TransformUv) -> Result<()> {
    check_scope(mesh, faces)?;
    let pivot = match t.pivot {
        Some(p) => Vec2::from(p),
        None => {
            let (lo, hi) = uv_bounds(mesh, faces);
            (lo + hi) * 0.5
        }
    };
    let (s, c) = t.rotate_deg.to_radians().sin_cos();
    let (scale, translate) = (Vec2::from(t.scale), Vec2::from(t.translate));
    for f in faces {
        for corner in &mut mesh.faces.get_mut(f).unwrap().corners {
            let d = (corner.uv - pivot) * scale;
            // v points down, so a visually counter-clockwise turn flips the usual signs.
            corner.uv = pivot + Vec2::new(c * d.x + s * d.y, -s * d.x + c * d.y) + translate;
        }
    }
    Ok(())
}

pub(crate) fn check_scope(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Result<()> {
    if faces.is_empty() {
        return invalid("no faces given");
    }
    for &f in faces {
        mesh.face(f)?;
    }
    Ok(())
}

pub(crate) fn uv_bounds(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> (Vec2, Vec2) {
    let mut it = faces.iter().filter_map(|f| mesh.faces.get(f)).flat_map(|f| f.corners.iter().map(|c| c.uv));
    let first = it.next().unwrap_or(Vec2::ZERO);
    it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)))
}
