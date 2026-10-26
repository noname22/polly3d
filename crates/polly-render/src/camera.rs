//! Camera resolution (views, auto-framing) and projection helpers.

use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::options::View;

const DEFAULT_FOV: f32 = 40.0;

/// A fully resolved camera. Returned with every render so `pick` / `project`
/// can reuse exactly the same projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CameraInfo {
    /// View name (`front`, `persp`, ...).
    pub view: String,
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    /// Vertical field of view (perspective only).
    pub fov_deg: f32,
    pub ortho: bool,
    /// Visible world height (orthographic only).
    pub ortho_height: f32,
    pub near: f32,
    pub far: f32,
    pub width: u32,
    pub height: u32,
}

impl CameraInfo {
    pub fn eye_v(&self) -> Vec3 {
        Vec3::from(self.eye)
    }
    pub fn forward(&self) -> Vec3 {
        (Vec3::from(self.target) - self.eye_v()).normalize_or(Vec3::NEG_Z)
    }
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::from(self.up)).normalize_or(Vec3::X)
    }
    pub fn screen_up(&self) -> Vec3 {
        self.right().cross(self.forward())
    }
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }
    pub fn view_matrix(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye_v(), Vec3::from(self.target), self.screen_up())
    }
    pub fn proj_matrix(&self) -> Mat4 {
        if self.ortho {
            let hh = self.ortho_height * 0.5;
            let hw = hh * self.aspect();
            glam::camera::rh::proj::directx::orthographic(-hw, hw, -hh, hh, self.near, self.far)
        } else {
            glam::camera::rh::proj::directx::perspective(self.fov_deg.to_radians(), self.aspect(), self.near, self.far)
        }
    }
    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// World point -> pixel coordinates (top-left origin). `None` when behind
    /// the camera. Points outside the image return coordinates outside 0..size.
    pub fn project(&self, p: Vec3) -> Option<[f32; 2]> {
        let c = self.view_proj() * p.extend(1.0);
        if c.w <= 1e-6 {
            return None;
        }
        let n = c.truncate() / c.w;
        Some([(n.x + 1.0) * 0.5 * self.width as f32, (1.0 - n.y) * 0.5 * self.height as f32])
    }

    /// Like `project` but also returns NDC depth.
    pub(crate) fn project_depth(&self, p: Vec3) -> Option<([f32; 2], f32)> {
        let c = self.view_proj() * p.extend(1.0);
        if c.w <= 1e-6 {
            return None;
        }
        let n = c.truncate() / c.w;
        Some(([(n.x + 1.0) * 0.5 * self.width as f32, (1.0 - n.y) * 0.5 * self.height as f32], n.z))
    }

    /// Ray through pixel coordinates: (origin, unit direction). The origin is
    /// the eye for perspective cameras and the near plane for orthographic ones.
    pub fn ray(&self, px: f32, py: f32) -> (Vec3, Vec3) {
        let nx = px / self.width as f32 * 2.0 - 1.0;
        let ny = 1.0 - py / self.height as f32 * 2.0;
        let inv = self.view_proj().inverse();
        let un = |z: f32| {
            let v: Vec4 = inv * Vec4::new(nx, ny, z, 1.0);
            v.truncate() / v.w
        };
        if self.ortho {
            (un(0.0), self.forward())
        } else {
            let eye = self.eye_v();
            (eye, (un(1.0) - eye).normalize())
        }
    }

    /// Ray from the camera towards a world point.
    pub(crate) fn ray_to(&self, p: Vec3) -> (Vec3, Vec3) {
        if self.ortho {
            let f = self.forward();
            let d = (p - self.eye_v()).dot(f);
            (p - f * d, f)
        } else {
            let eye = self.eye_v();
            (eye, (p - eye).normalize_or(self.forward()))
        }
    }

    /// Direction from point `p` towards the camera.
    pub(crate) fn to_eye(&self, p: Vec3) -> Vec3 {
        if self.ortho { -self.forward() } else { (self.eye_v() - p).normalize_or(-self.forward()) }
    }

    /// e.g. `+X right, +Y up` when the screen axes line up with world axes.
    pub fn axis_hint(&self) -> Option<String> {
        let name = |v: Vec3| -> Option<String> {
            let axes = [(Vec3::X, "X"), (Vec3::Y, "Y"), (Vec3::Z, "Z")];
            for (a, n) in axes {
                let d = v.dot(a);
                if d > 0.999 {
                    return Some(format!("+{n}"));
                }
                if d < -0.999 {
                    return Some(format!("-{n}"));
                }
            }
            None
        };
        Some(format!("{} right, {} up", name(self.right())?, name(self.screen_up())?))
    }
}

/// Fixed direction (camera forward, up) for axis views, or orbit angles.
fn view_dirs(view: &View) -> Option<(Vec3, Vec3, bool)> {
    Some(match view {
        View::Front => (Vec3::NEG_Z, Vec3::Y, true),
        View::Back => (Vec3::Z, Vec3::Y, true),
        View::Left => (Vec3::X, Vec3::Y, true),
        View::Right => (Vec3::NEG_X, Vec3::Y, true),
        View::Top => (Vec3::NEG_Y, Vec3::NEG_Z, true),
        View::Bottom => (Vec3::Y, Vec3::Z, true),
        View::Persp => (-orbit_dir(35.0, 25.0), Vec3::Y, false),
        View::Iso => (-orbit_dir(45.0, 35.264), Vec3::Y, true),
        View::Orbit { yaw_deg, pitch_deg, .. } => (-orbit_dir(*yaw_deg, pitch_deg.clamp(-89.5, 89.5)), Vec3::Y, false),
        View::Custom { .. } => return None,
    })
}

/// Unit vector from target to eye for orbit angles.
pub fn orbit_dir(yaw_deg: f32, pitch_deg: f32) -> Vec3 {
    let (y, p) = (yaw_deg.to_radians(), pitch_deg.to_radians());
    Vec3::new(y.sin() * p.cos(), p.sin(), y.cos() * p.cos())
}

fn corners((lo, hi): (Vec3, Vec3)) -> [Vec3; 8] {
    std::array::from_fn(|i| {
        Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        )
    })
}

fn basis(f: Vec3, up: Vec3) -> (Vec3, Vec3, Vec3) {
    let f = f.normalize_or(Vec3::NEG_Z);
    let mut r = f.cross(up);
    if r.length_squared() < 1e-8 {
        r = f.cross(if f.y.abs() < 0.9 { Vec3::Y } else { Vec3::NEG_Z });
    }
    let r = r.normalize();
    (f, r, r.cross(f))
}

/// Resolve a camera. `frame` = bounds to fit, `depth` = bounds of everything
/// drawn (used for near/far planes).
pub(crate) fn resolve(view: &View, frame: (Vec3, Vec3), depth: (Vec3, Vec3), zoom: f32, size: [u32; 2]) -> CameraInfo {
    let zoom = if zoom.is_finite() && zoom > 0.0 { zoom.clamp(0.01, 1000.0) } else { 1.0 };
    let aspect = size[0] as f32 / size[1] as f32;
    let center = (frame.0 + frame.1) * 0.5;
    let fc = corners(frame);
    let radius = ((frame.1 - frame.0).length() * 0.5).max(0.05);

    let (eye, target, up, fov, ortho, ortho_height);
    match view {
        View::Custom { eye: e, target: t, up: u, fov_deg, ortho: o } => {
            let e = Vec3::from(*e);
            let t = Vec3::from(*t);
            let (f, r, uu) = basis(t - e, u.map(Vec3::from).unwrap_or(Vec3::Y));
            eye = if (t - e).length_squared() < 1e-12 { t - f } else { e };
            target = t;
            up = uu;
            ortho = *o;
            let f0 = fov_deg.unwrap_or(DEFAULT_FOV).clamp(1.0, 170.0);
            fov = (2.0 * ((f0.to_radians() * 0.5).tan() / zoom).atan()).to_degrees();
            let half = fc
                .iter()
                .map(|p| {
                    let d = *p - t;
                    d.dot(uu).abs().max(d.dot(r).abs() / aspect)
                })
                .fold(0.0f32, f32::max);
            ortho_height = (half * 2.0 * 1.3).max(0.1) / zoom;
        }
        _ => {
            let (f, up0, is_ortho) = view_dirs(view).unwrap();
            let (f, r, u) = basis(f, up0);
            target = center;
            up = u;
            ortho = is_ortho;
            if is_ortho {
                let half = fc
                    .iter()
                    .map(|p| {
                        let d = *p - center;
                        d.dot(u).abs().max(d.dot(r).abs() / aspect)
                    })
                    .fold(0.0f32, f32::max);
                ortho_height = (half * 2.0 * 1.3).max(0.1) / zoom;
                fov = DEFAULT_FOV;
                eye = center - f * (radius * 4.0 + 1.0);
            } else {
                let tv = (DEFAULT_FOV.to_radians() * 0.5).tan();
                let th = tv * aspect;
                let fit = fc
                    .iter()
                    .map(|p| {
                        let d = *p - center;
                        let z = -d.dot(f);
                        (z + d.dot(r).abs() / th).max(z + d.dot(u).abs() / tv)
                    })
                    .fold(0.0f32, f32::max)
                    * 1.2;
                let dist = match view {
                    View::Orbit { distance: Some(d), .. } if *d > 0.0 => *d,
                    // Orbits frame the bounding sphere, which is the same from every
                    // angle, so dragging around the model keeps a constant distance.
                    View::Orbit { .. } => radius * 1.05 / tv.min(th).atan().sin(),
                    _ => fit.max(radius * 1.2),
                };
                fov = (2.0 * (tv / zoom).atan()).to_degrees();
                ortho_height = 2.0 * dist * tv / zoom;
                eye = center - f * dist;
            }
        }
    }

    // Near / far from everything that will be drawn.
    let mut cam = CameraInfo {
        view: view.name().to_string(),
        eye: eye.to_array(),
        target: target.to_array(),
        up: up.to_array(),
        fov_deg: fov,
        ortho,
        ortho_height,
        near: 0.01,
        far: 100.0,
        width: size[0],
        height: size[1],
    };
    let f = cam.forward();
    let mut zs = corners(depth).iter().chain(fc.iter()).map(|p| (*p - eye).dot(f)).collect::<Vec<_>>();
    zs.push((target - eye).dot(f));
    let zmin = zs.iter().copied().fold(f32::INFINITY, f32::min);
    let zmax = zs.iter().copied().fold(f32::NEG_INFINITY, f32::max).max(0.1);
    let pad = (zmax - zmin).abs() * 0.05 + 0.01;
    if ortho {
        // Orthographic: shift the eye back so everything sits in front of the near plane.
        let shift = (0.05 - (zmin - pad)).max(0.0);
        cam.eye = (eye - f * shift).to_array();
        cam.near = zmin - pad + shift;
        cam.far = zmax + pad + shift;
    } else {
        cam.far = zmax + pad;
        cam.near = (zmin - pad).max(cam.far * 1e-4).max(1e-4);
    }
    cam
}
