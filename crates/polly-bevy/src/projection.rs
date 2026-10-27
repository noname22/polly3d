//! Bevy camera projection built from polly's `CameraInfo`, so pixels line up
//! exactly with `prep.rs` picking and `annotate.rs` labels.

use bevy::camera::{CameraProjection, SubCameraView};
use bevy::math::{Mat4, Vec3A, Vec4};
use polly_render::CameraInfo;

/// polly's projection with reverse-Z depth (Bevy's convention: near = 1, far = 0).
/// The x/y rows are identical to [`CameraInfo::proj_matrix`].
#[derive(Debug, Clone)]
pub struct PollyProjection {
    pub cam: CameraInfo,
}

impl PollyProjection {
    pub fn matrix(&self) -> Mat4 {
        let c = &self.cam;
        if c.ortho {
            let hh = c.ortho_height * 0.5;
            let hw = hh * c.aspect();
            // near and far swapped: z maps far -> 0, near -> 1.
            crate::conv::m4(polly_core::glam::camera::rh::proj::directx::orthographic(-hw, hw, -hh, hh, c.far, c.near))
        } else {
            crate::conv::m4(polly_core::glam::camera::rh::proj::directx::perspective(c.fov_deg.to_radians(), c.aspect(), c.far, c.near))
        }
    }
}

impl CameraProjection for PollyProjection {
    fn get_clip_from_view(&self) -> Mat4 {
        self.matrix()
    }
    fn get_clip_from_view_for_sub(&self, _sub_view: &SubCameraView) -> Mat4 {
        self.matrix()
    }
    fn update(&mut self, _width: f32, _height: f32) {}
    fn far(&self) -> f32 {
        self.cam.far
    }
    fn get_frustum_corners(&self, z_near: f32, z_far: f32) -> [Vec3A; 8] {
        // View-space corners at the given (positive) distances, in the order
        // Bevy expects: near (bottom right, top right, top left, bottom left), then far.
        let inv = self.matrix().inverse();
        let corner = |x: f32, y: f32, d: f32| -> Vec3A {
            // A point on the ray through NDC (x, y), scaled to view depth d.
            let p = inv * Vec4::new(x, y, 0.5, 1.0);
            let p = p.truncate() / p.w;
            let p = if self.cam.ortho { bevy::math::Vec3::new(p.x, p.y, -d) } else { p * (d / -p.z) };
            Vec3A::from(p)
        };
        let mut out = [Vec3A::ZERO; 8];
        for (i, d) in [z_near, z_far].into_iter().enumerate() {
            out[i * 4] = corner(1.0, -1.0, d);
            out[i * 4 + 1] = corner(1.0, 1.0, d);
            out[i * 4 + 2] = corner(-1.0, 1.0, d);
            out[i * 4 + 3] = corner(-1.0, -1.0, d);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(ortho: bool) -> CameraInfo {
        CameraInfo {
            view: "t".into(),
            eye: [1.0, 2.0, 5.0],
            target: [0.0, 0.5, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_deg: 40.0,
            ortho,
            ortho_height: 3.0,
            near: 0.1,
            far: 20.0,
            width: 320,
            height: 200,
        }
    }

    /// Same pixel for every point as polly's own projection; depth reversed.
    #[test]
    fn xy_matches_polly_and_depth_is_reversed() {
        for ortho in [false, true] {
            let c = cam(ortho);
            let p = PollyProjection { cam: c.clone() };
            let view = crate::conv::m4(c.view_matrix());
            for q in [bevy::math::Vec3::new(0.3, 0.2, 0.5), bevy::math::Vec3::new(-0.5, 1.0, -0.4)] {
                let a = crate::conv::m4(c.proj_matrix()) * view * q.extend(1.0);
                let b = p.matrix() * view * q.extend(1.0);
                let (a, b) = (a / a.w, b / b.w);
                assert!((a.x - b.x).abs() < 1e-5 && (a.y - b.y).abs() < 1e-5, "{ortho}: {a:?} {b:?}");
                assert!((a.z + b.z - 1.0).abs() < 1e-3, "{ortho}: depth {a:?} {b:?}");
            }
        }
    }
}
