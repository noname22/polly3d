//! Builds line / point instances for overlays (wireframe, selection, grid, ...).

use std::collections::BTreeSet;

use glam::Vec3;
use polly_core::mesh::Edge;
use polly_core::scene::Scene;

use crate::options::Overlays;
use crate::prep::{Prepared, world_bounds};

/// One overlay line (or a point when `a == b` and `params[1] == 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineInst {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [f32; 4],
    /// width px, point flag, depth-bias multiplier, on-top flag
    pub params: [f32; 4],
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// sRGB colour (0..1) + alpha -> linear premultiplied-free RGBA for the GPU.
pub(crate) fn lin(c: [f32; 4]) -> [f32; 4] {
    [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]]
}

pub(crate) const WIRE: [f32; 4] = [0.03, 0.03, 0.05, 0.85];
pub(crate) const SELECT: [f32; 4] = [1.0, 0.55, 0.08, 1.0];
pub(crate) const SEAM: [f32; 4] = [0.95, 0.12, 0.12, 1.0];
pub(crate) const SHARP: [f32; 4] = [0.1, 0.85, 0.95, 1.0];
pub(crate) const NORMAL: [f32; 4] = [0.7, 1.0, 0.15, 1.0];
pub(crate) const BOUNDS: [f32; 4] = [1.0, 0.85, 0.2, 0.8];
pub(crate) const BONE: [f32; 4] = [0.98, 0.78, 0.25, 1.0];
pub(crate) const BONE_LINK: [f32; 4] = [0.98, 0.78, 0.25, 0.45];

pub(crate) struct LineBuilder {
    pub out: Vec<LineInst>,
    /// Width multiplier (resolution dependent).
    pub s: f32,
}

impl LineBuilder {
    pub fn line(&mut self, a: Vec3, b: Vec3, color: [f32; 4], width: f32, bias: f32) {
        self.out.push(LineInst { a: a.to_array(), b: b.to_array(), color: lin(color), params: [width * self.s, 0.0, bias, 0.0] });
    }
    pub fn point(&mut self, p: Vec3, color: [f32; 4], size: f32, bias: f32) {
        self.out.push(LineInst { a: p.to_array(), b: p.to_array(), color: lin(color), params: [size * self.s, 1.0, bias, 0.0] });
    }
    /// A line drawn on top of all geometry.
    pub fn line_on_top(&mut self, a: Vec3, b: Vec3, color: [f32; 4], width: f32) {
        self.out.push(LineInst { a: a.to_array(), b: b.to_array(), color: lin(color), params: [width * self.s, 0.0, 0.0, 1.0] });
    }
    pub fn point_on_top(&mut self, p: Vec3, color: [f32; 4], size: f32) {
        self.out.push(LineInst { a: p.to_array(), b: p.to_array(), color: lin(color), params: [size * self.s, 1.0, 0.0, 1.0] });
    }
}

/// Grid parameters: (half extent, step).
pub(crate) fn grid_extent(bounds: Option<(Vec3, Vec3)>) -> (f32, f32) {
    let m = bounds.map_or(1.0, |(lo, hi)| lo.x.abs().max(hi.x.abs()).max(lo.z.abs()).max(hi.z.abs())).max(0.5);
    let target = m / 4.0;
    let p = 10f32.powf(target.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0].iter().map(|k| k * p).find(|s| *s >= target).unwrap_or(10.0 * p);
    let ext = ((m * 1.4) / step).ceil() * step;
    (ext, step)
}

pub(crate) fn build(scene: &Scene, prep: &Prepared, ov: &Overlays, s: f32, frame_radius: f32, frame_bounds: Option<(Vec3, Vec3)>) -> Vec<LineInst> {
    let mut lb = LineBuilder { out: Vec::new(), s };
    let all: Vec<_> = prep.objects.iter().collect();

    if ov.grid {
        let (ext, step) = grid_extent(frame_bounds.or_else(|| world_bounds(&all)));
        let n = (ext / step).round() as i32;
        let minor = [0.55, 0.57, 0.62, 0.35];
        for i in -n..=n {
            let t = i as f32 * step;
            if i != 0 {
                lb.line(Vec3::new(t, 0.0, -ext), Vec3::new(t, 0.0, ext), minor, 1.0, 0.0);
                lb.line(Vec3::new(-ext, 0.0, t), Vec3::new(ext, 0.0, t), minor, 1.0, 0.0);
            }
        }
        lb.line(Vec3::new(-ext, 0.0, 0.0), Vec3::new(ext, 0.0, 0.0), [0.9, 0.2, 0.2, 0.9], 2.0, 0.0);
        lb.line(Vec3::new(0.0, 0.0, -ext), Vec3::new(0.0, 0.0, ext), [0.2, 0.4, 0.95, 0.9], 2.0, 0.0);
        let yh = frame_bounds.or_else(|| world_bounds(&all)).map_or(step, |(_, hi)| hi.y.max(step) * 1.15);
        lb.line(Vec3::ZERO, Vec3::new(0.0, yh, 0.0), [0.25, 0.85, 0.3, 0.6], 1.5, 0.0);
    }

    for po in &prep.objects {
        let mesh = &po.mesh;
        let obj = &scene.objects[&po.id];
        if ov.bone_axes
            && let Some(sk) = &obj.skeleton
        {
            let m = scene.world_matrix(po.id);
            for b in sk.bones.values() {
                let f = m * b.rest_matrix();
                let h = f.transform_point3(Vec3::ZERO);
                let len = (m.transform_point3(b.tail) - m.transform_point3(b.head)).length().max(1e-3) * 0.35;
                for (axis, color) in [(Vec3::X, [0.95, 0.25, 0.2, 1.0]), (Vec3::Y, [0.3, 0.9, 0.3, 1.0]), (Vec3::Z, [0.3, 0.5, 1.0, 1.0])] {
                    let d = f.transform_vector3(axis).normalize_or_zero() * len;
                    lb.line_on_top(h, h + d, color, 2.5);
                }
            }
        }
        if ov.bones {
            if let Some(sk) = &obj.skeleton {
                let m = scene.world_matrix(po.id);
                for b in sk.bones.values() {
                    let (h, t) = (m.transform_point3(b.head), m.transform_point3(b.tail));
                    lb.line_on_top(h, t, [0.05, 0.04, 0.02, 1.0], 6.0);
                    lb.line_on_top(h, t, BONE, 3.5);
                    lb.point_on_top(h, [0.05, 0.04, 0.02, 1.0], 9.0);
                    lb.point_on_top(h, BONE, 6.0);
                    // Dashed-looking thin link from a disconnected parent's tail.
                    if let Some(p) = b.parent.and_then(|p| sk.bones.get(&p)) {
                        if (p.tail - b.head).length() > 1e-4 {
                            lb.line_on_top(m.transform_point3(p.tail), h, BONE_LINK, 1.5);
                        }
                    }
                }
            }
        }
        let valid = |e: &Edge| po.pos.contains_key(&e.0) && po.pos.contains_key(&e.1);
        let edges: BTreeSet<Edge> = mesh.edges().into_iter().filter(valid).collect();
        let p = |v| po.pos[&v];
        if ov.wireframe {
            for e in &edges {
                lb.line(p(e.0), p(e.1), WIRE, 1.5, 1.0);
            }
        }
        if ov.sharp {
            for e in mesh.sharp.iter().filter(|e| edges.contains(e)) {
                lb.line(p(e.0), p(e.1), SHARP, 2.5, 1.2);
            }
        }
        if ov.seams {
            for e in mesh.seams.iter().filter(|e| edges.contains(e)) {
                lb.line(p(e.0), p(e.1), SEAM, 2.5, 1.3);
            }
        }
        if ov.normals {
            for (fid, f) in &mesh.faces {
                let Some(&n) = po.face_normal.get(fid) else { continue };
                let c = po.face_center(f);
                let size = mesh.face_area(f).sqrt();
                let len = (size * 0.4).clamp(frame_radius * 0.04, frame_radius * 0.25);
                lb.line(c, c + n * len, NORMAL, 1.5, 1.0);
                lb.point(c, NORMAL, 4.0, 1.0);
            }
        }
        if ov.bounds {
            if let Some((lo, hi)) = po.bounds() {
                let c = |i: usize| {
                    Vec3::new(
                        if i & 1 == 0 { lo.x } else { hi.x },
                        if i & 2 == 0 { lo.y } else { hi.y },
                        if i & 4 == 0 { lo.z } else { hi.z },
                    )
                };
                for (a, b) in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)] {
                    lb.line(c(a), c(b), BOUNDS, 1.0, 1.0);
                }
            }
        }
        if ov.selection && !obj.selection.is_empty() {
            for e in obj.selection.implied_edges(mesh).iter().filter(|e| edges.contains(e)) {
                lb.line(p(e.0), p(e.1), SELECT, 3.0, 1.5);
            }
        }
        if ov.verts {
            for &v in po.pos.keys() {
                lb.point(p(v), [0.02, 0.02, 0.03, 1.0], 6.0, 1.6);
                lb.point(p(v), [0.95, 0.95, 0.9, 1.0], 3.5, 1.6);
            }
        }
        if ov.selection {
            for v in obj.selection.verts.iter().filter(|v| po.pos.contains_key(v)) {
                lb.point(p(*v), [0.02, 0.02, 0.03, 1.0], 9.0, 1.7);
                lb.point(p(*v), SELECT, 6.5, 1.7);
            }
        }
    }
    lb.out
}
