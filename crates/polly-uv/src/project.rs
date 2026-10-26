use std::collections::BTreeSet;
use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec3};
use polly_core::Result;
use polly_core::error::invalid;
use polly_core::mesh::{FaceId, Mesh, newell_vector};
use serde::{Deserialize, Serialize};

use crate::check_scope;

/// Simple UV projections. All except `per_face` are scaled uniformly to fit
/// the 0..1 square.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Projection {
    /// Project along `normal` (default: the faces' average normal).
    Planar {
        #[serde(default)]
        normal: Option<[f32; 3]>,
    },
    /// Each face projects along the axis closest to its normal.
    Box,
    /// Wrap around `axis` (default Y) through the faces' centroid. The wrap
    /// seam is at the back (-Z for the Y axis); caps are projected flat above
    /// and below the strip.
    Cylindrical {
        #[serde(default)]
        axis: Option<[f32; 3]>,
    },
    /// Longitude/latitude around Y through the faces' bounding-box center.
    Spherical,
    /// Every face fills the whole 0..1 square (quads map to its corners,
    /// other polygons stretch to fit).
    PerFace,
}

/// Write corner uvs for `faces` using projection `p`.
pub fn project(mesh: &mut Mesh, faces: &BTreeSet<FaceId>, p: &Projection) -> Result<()> {
    check_scope(mesh, faces)?;
    let layout = match p {
        Projection::Planar { normal } => {
            let n = match normal {
                Some(n) => Vec3::from(*n),
                None => faces.iter().map(|f| newell_vector(&mesh.face_positions(&mesh.faces[f]))).sum(),
            };
            let area: f32 = faces.iter().map(|f| mesh.face_area(&mesh.faces[f])).sum();
            if n.length() < 1e-4 * area.max(1e-12) {
                return invalid("faces have no average normal (closed shape?); pass an explicit normal");
            }
            planar(mesh, faces, n)
        }
        Projection::Box => box_map(mesh, faces),
        Projection::Cylindrical { axis } => {
            let a = axis.map_or(Vec3::Y, Vec3::from);
            if a.length() < 1e-8 {
                return invalid("cylinder axis must be non-zero");
            }
            cylindrical(mesh, faces, a.normalize())
        }
        Projection::Spherical => spherical(mesh, faces),
        Projection::PerFace => {
            for f in faces {
                per_face(mesh, *f);
            }
            return Ok(());
        }
    };
    write_fitted(mesh, layout);
    Ok(())
}

/// Right/up axes for looking at a surface with normal `n` from outside:
/// world Y is up where possible; top/bottom views put -Z/+Z up.
pub(crate) fn view_basis(n: Vec3) -> (Vec3, Vec3) {
    let n = n.normalize_or(Vec3::Z);
    let hint = if n.y.abs() < 0.99 { Vec3::Y } else { -Vec3::Z * n.y.signum() };
    let up = hint.reject_from_normalized(n).normalize();
    (up.cross(n), up)
}

/// Per-face corner coordinates in math space (y up), any scale.
type Layout = Vec<(FaceId, Vec<Vec2>)>;

pub(crate) fn planar(mesh: &Mesh, faces: &BTreeSet<FaceId>, n: Vec3) -> Layout {
    let (r, u) = view_basis(n);
    faces
        .iter()
        .map(|&f| (f, mesh.face_positions(&mesh.faces[&f]).iter().map(|p| Vec2::new(p.dot(r), p.dot(u))).collect()))
        .collect()
}

fn box_map(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Layout {
    faces
        .iter()
        .map(|&f| {
            let face = &mesh.faces[&f];
            let n = mesh.face_normal(face);
            let k = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs())).unwrap();
            let mut axis = Vec3::ZERO;
            axis[k] = if n[k] < 0.0 { -1.0 } else { 1.0 };
            let (r, u) = view_basis(axis);
            (f, mesh.face_positions(face).iter().map(|p| Vec2::new(p.dot(r), p.dot(u))).collect())
        })
        .collect()
}

/// Basis (e1, e2) perpendicular to `a` with e1 x e2 = a; angle 0 faces +Z when possible.
fn ring_basis(a: Vec3) -> (Vec3, Vec3) {
    let hint = if a.z.abs() < 0.99 { Vec3::Z } else { Vec3::X };
    let e1 = hint.reject_from_normalized(a).normalize();
    (e1, a.cross(e1))
}

/// Angles of a face's corners around an axis, unwrapped to be contiguous
/// around the face's mean direction. Corners on the axis (`None`) take that
/// mean angle.
fn face_angles(dirs: &[Option<f32>]) -> Vec<f32> {
    let mean = dirs.iter().flatten().fold(Vec2::ZERO, |acc, &t| acc + Vec2::from_angle(t));
    let c = if mean.length() > 1e-6 { mean.to_angle() } else { dirs.iter().flatten().next().copied().unwrap_or(0.0) };
    dirs.iter().map(|d| d.map_or(c, |t| c + (t - c + PI).rem_euclid(TAU) - PI)).collect()
}

fn centroid(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Vec3 {
    let verts: BTreeSet<_> = faces.iter().flat_map(|f| mesh.faces[f].verts.iter().copied()).collect();
    verts.iter().map(|&v| mesh.pos(v)).sum::<Vec3>() / verts.len().max(1) as f32
}

fn cylindrical(mesh: &Mesh, faces: &BTreeSet<FaceId>, a: Vec3) -> Layout {
    let c = centroid(mesh, faces);
    let (e1, e2) = ring_basis(a);
    let verts: BTreeSet<_> = faces.iter().flat_map(|f| mesh.faces[f].verts.iter().copied()).collect();
    let radius = (verts.iter().map(|&v| (mesh.pos(v) - c).reject_from_normalized(a).length()).sum::<f32>()
        / verts.len().max(1) as f32)
        .max(1e-6);
    let (mut sides, mut top, mut bottom) = (Layout::new(), BTreeSet::new(), BTreeSet::new());
    for &f in faces {
        let face = &mesh.faces[&f];
        let along = mesh.face_normal(face).dot(a);
        if along > 0.75 {
            top.insert(f);
        } else if along < -0.75 {
            bottom.insert(f);
        } else {
            let pts: Vec<Vec3> = mesh.face_positions(face).iter().map(|p| *p - c).collect();
            let dirs: Vec<Option<f32>> = pts
                .iter()
                .map(|d| {
                    let (x, y) = (d.dot(e1), d.dot(e2));
                    (x.hypot(y) > 1e-5 * radius).then(|| y.atan2(x))
                })
                .collect();
            let uv = face_angles(&dirs).iter().zip(&pts).map(|(t, d)| Vec2::new(t * radius, d.dot(a))).collect();
            sides.push((f, uv));
        }
    }
    let (lo, hi) = bounds(&sides).unwrap_or((Vec2::ZERO, Vec2::ZERO));
    let gap = 0.05 * radius;
    for (caps, n, above) in [(top, a, true), (bottom, -a, false)] {
        let mut cap = planar(mesh, &caps, n);
        if let Some((clo, chi)) = bounds(&cap) {
            let dx = (lo.x + hi.x) * 0.5 - (clo.x + chi.x) * 0.5;
            let dy = if above { hi.y + gap - clo.y } else { lo.y - gap - chi.y };
            cap.iter_mut().flat_map(|(_, uv)| uv.iter_mut()).for_each(|p| *p += Vec2::new(dx, dy));
            sides.extend(cap);
        }
    }
    sides
}

fn spherical(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Layout {
    let pts: Vec<Vec3> = faces.iter().flat_map(|f| mesh.face_positions(&mesh.faces[f])).collect();
    let (lo, hi) = pts.iter().fold((pts[0], pts[0]), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let c = (lo + hi) * 0.5;
    let (e1, e2) = ring_basis(Vec3::Y);
    faces
        .iter()
        .map(|&f| {
            let d: Vec<Vec3> = mesh.face_positions(&mesh.faces[&f]).iter().map(|p| *p - c).collect();
            let dirs: Vec<Option<f32>> = d
                .iter()
                .map(|d| {
                    let (x, y) = (d.dot(e1), d.dot(e2));
                    (x.hypot(y) > 1e-5 * d.length().max(1e-9)).then(|| y.atan2(x))
                })
                .collect();
            let uv = face_angles(&dirs)
                .iter()
                .zip(&d)
                .map(|(t, d)| Vec2::new(*t, (d.y / d.length().max(1e-9)).clamp(-1.0, 1.0).asin()))
                .collect();
            (f, uv)
        })
        .collect()
}

fn per_face(mesh: &mut Mesh, f: FaceId) {
    let face = &mesh.faces[&f];
    let (r, u) = view_basis(mesh.face_normal(face));
    // Image space: v down.
    let pts: Vec<Vec2> = mesh.face_positions(face).iter().map(|p| Vec2::new(p.dot(r), -p.dot(u))).collect();
    let (lo, hi) = pts.iter().fold((pts[0], pts[0]), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let size = (hi - lo).max(Vec2::splat(1e-9));
    let mut uv: Vec<Vec2> = pts.iter().map(|p| (*p - lo) / size).collect();
    if uv.len() == 4 {
        // Counter-clockwise from outside = bottom-left, bottom-right, top-right, top-left.
        let target = [Vec2::new(0.0, 1.0), Vec2::ONE, Vec2::new(1.0, 0.0), Vec2::ZERO];
        let cost = |s: usize| (0..4).map(|i| uv[i].distance_squared(target[(i + s) % 4])).sum::<f32>();
        let s = (0..4).min_by(|&a, &b| cost(a).total_cmp(&cost(b))).unwrap();
        uv = (0..4).map(|i| target[(i + s) % 4]).collect();
    }
    for (c, p) in mesh.faces.get_mut(&f).unwrap().corners.iter_mut().zip(uv) {
        c.uv = p;
    }
}

fn bounds(layout: &Layout) -> Option<(Vec2, Vec2)> {
    let mut it = layout.iter().flat_map(|(_, uv)| uv.iter().copied());
    let first = it.next()?;
    Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
}

/// Flip to image space (v down) and scale uniformly into 0..1.
pub(crate) fn write_fitted(mesh: &mut Mesh, mut layout: Layout) {
    layout.iter_mut().flat_map(|(_, uv)| uv.iter_mut()).for_each(|p| p.y = -p.y);
    let Some((lo, hi)) = bounds(&layout) else { return };
    let scale = 1.0 / (hi - lo).max_element().max(1e-9);
    for (f, uv) in layout {
        for (c, p) in mesh.faces.get_mut(&f).unwrap().corners.iter_mut().zip(uv) {
            c.uv = (p - lo) * scale;
        }
    }
}
