use std::collections::BTreeSet;

use glam::Vec2;
use polly_core::Result;
use polly_core::error::invalid;
use polly_core::geom::triangulate_polygon;
use polly_core::ids;
use polly_core::mesh::{FaceId, Mesh, newell_vector};
use serde::{Deserialize, Serialize};

use crate::project::{planar, write_fitted};
use crate::report::{UvReport, report};
use crate::{check_scope, uv_islands};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct PackOptions {
    pub texture_size: [u32; 2],
    /// Empty texels kept between islands.
    pub padding_px: u32,
    /// Snap uvs to texel corners so island edges are crisp.
    pub pixel_snap: bool,
    /// Allow 90° rotation of islands for a tighter fit.
    pub rotate: bool,
    /// Texels per world unit; None = as large as fits.
    pub texel_density: Option<f32>,
    /// Empty texels kept along the texture border.
    pub margin_px: u32,
}

impl Default for PackOptions {
    fn default() -> Self {
        PackOptions { texture_size: [64, 64], padding_px: 1, pixel_snap: true, rotate: true, texel_density: None, margin_px: 0 }
    }
}

/// An island as a rigid shape in world units, relative to its uv bbox.
struct Shape {
    faces: BTreeSet<FaceId>,
    origin: Vec2,
    /// Converts current uv units to world units.
    scale: f32,
    size: Vec2,
}

#[derive(Clone, Copy)]
struct Placed {
    x: u32,
    y: u32,
    rotated: bool,
}

/// Pack the uv islands of `faces` into the texture: one common scale for all
/// islands (so relative texel density follows 3-D size), skyline packing,
/// padding and margin in texels, optional pixel snapping.
pub fn pack(mesh: &mut Mesh, faces: &BTreeSet<FaceId>, o: &PackOptions) -> Result<UvReport> {
    check_scope(mesh, faces)?;
    let [tw, th] = o.texture_size;
    if tw == 0 || th == 0 {
        return invalid("texture_size must be non-zero");
    }
    if tw <= 2 * o.margin_px || th <= 2 * o.margin_px {
        return invalid("margin_px leaves no room in the texture");
    }
    let mut warnings = Vec::new();
    let mut shapes = Vec::new();
    for island in uv_islands(mesh, faces) {
        let (mut area_uv, area_3d) = areas(mesh, &island);
        if area_uv < 1e-12 {
            warnings.push(format!(
                "island of {} had no uv area; gave it a planar projection (consider unwrap)",
                ids::face(*island.first().unwrap())
            ));
            let n = island.iter().map(|f| newell_vector(&mesh.face_positions(&mesh.faces[f]))).sum();
            write_fitted(mesh, planar(mesh, &island, n));
            area_uv = areas(mesh, &island).0;
        }
        let (lo, hi) = crate::uv_bounds(mesh, &island);
        let scale = if area_uv > 1e-12 && area_3d > 1e-12 { (area_3d / area_uv).sqrt() } else { 1.0 };
        let size = ((hi - lo) * scale).max(Vec2::splat(1e-6));
        shapes.push(Shape { faces: island, origin: lo, scale, size });
    }

    let avail = [tw - 2 * o.margin_px, th - 2 * o.margin_px];
    let fits = |s: f32| try_pack(&shapes, s, o, avail);
    let fill = || {
        // Upper bound: every island fits alone, and total area fits.
        let each = shapes
            .iter()
            .map(|sh| {
                let straight = (avail[0] as f32 / sh.size.x).min(avail[1] as f32 / sh.size.y);
                let turned = (avail[0] as f32 / sh.size.y).min(avail[1] as f32 / sh.size.x);
                if o.rotate { straight.max(turned) } else { straight }
            })
            .fold(f32::INFINITY, f32::min);
        let total: f32 = shapes.iter().map(|sh| sh.size.x * sh.size.y).sum();
        let mut hi = each.min((avail[0] as f32 * avail[1] as f32 / total).sqrt());
        if fits(hi).is_some() {
            return Some(hi);
        }
        let mut lo = 0.0;
        fits(1e-9)?;
        for _ in 0..40 {
            let mid = (lo + hi) * 0.5;
            if fits(mid).is_some() { lo = mid } else { hi = mid }
        }
        Some(lo.max(1e-9))
    };
    let density = match o.texel_density {
        Some(d) if fits(d).is_some() => d,
        Some(d) => {
            let s = fill();
            if let Some(s) = s {
                warnings.push(format!("texel_density {d} does not fit; reduced to {s:.2} texels/unit"));
            }
            s.unwrap_or(d)
        }
        None => fill().unwrap_or(0.0),
    };
    let Some(placed) = fits(density) else {
        return invalid(format!(
            "{} islands cannot fit in {tw}x{th} px with padding {} and margin {}; use a larger texture, less padding or fewer seams",
            shapes.len(),
            o.padding_px,
            o.margin_px
        ));
    };

    let size = Vec2::new(tw as f32, th as f32);
    for (sh, p) in shapes.iter().zip(placed) {
        let at = Vec2::new((p.x + o.margin_px) as f32, (p.y + o.margin_px) as f32);
        for f in &sh.faces {
            for c in &mut mesh.faces.get_mut(f).unwrap().corners {
                let mut q = (c.uv - sh.origin) * sh.scale;
                if p.rotated {
                    q = Vec2::new(sh.size.y - q.y, q.x);
                }
                let mut px = at + q * density;
                if o.pixel_snap {
                    px = px.round();
                }
                c.uv = px / size;
            }
        }
    }
    let mut r = report(mesh, faces, o.texture_size);
    warnings.append(&mut r.warnings);
    r.warnings = warnings;
    Ok(r)
}

/// (uv area, 3-D area) of a set of faces.
fn areas(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> (f32, f32) {
    let mut uv = 0.0;
    for f in faces {
        let face = &mesh.faces[f];
        for [a, b, c] in triangulate_polygon(&mesh.face_positions(face)) {
            let (a, b, c) = (face.corners[a].uv, face.corners[b].uv, face.corners[c].uv);
            uv += (b - a).perp_dot(c - a).abs() * 0.5;
        }
    }
    (uv, faces.iter().map(|f| mesh.face_area(&mesh.faces[f])).sum())
}

/// Pixel footprint of a shape at density `s`.
fn footprint(size: Vec2, s: f32) -> [u32; 2] {
    let px = |v: f32| ((v * s - 1e-3).ceil() as u32).max(1);
    [px(size.x), px(size.y)]
}

fn try_pack(shapes: &[Shape], s: f32, o: &PackOptions, avail: [u32; 2]) -> Option<Vec<Placed>> {
    let pad = o.padding_px;
    // Every rect is grown by the padding on its right/bottom; so is the bin.
    let mut sky = Skyline::new(avail[0] + pad, avail[1] + pad);
    let dims: Vec<[u32; 2]> = shapes.iter().map(|sh| footprint(sh.size, s).map(|d| d + pad)).collect();
    let mut order: Vec<usize> = (0..shapes.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse((dims[i][0].max(dims[i][1]), dims[i][0] * dims[i][1])));
    let mut out = vec![Placed { x: 0, y: 0, rotated: false }; shapes.len()];
    for i in order {
        let [w, h] = dims[i];
        let mut best: Option<(u32, u32, usize, bool)> = None;
        for node in 0..sky.nodes.len() {
            for (rw, rh, rotated) in [(w, h, false), (h, w, true)] {
                if rotated && (!o.rotate || w == h) {
                    continue;
                }
                if let Some(y) = sky.fit(node, rw, rh) {
                    let x = sky.nodes[node].0;
                    if best.is_none_or(|(by, bx, _, _)| (y + rh, x) < (by, bx)) {
                        best = Some((y + rh, x, node, rotated));
                    }
                }
            }
        }
        let (_, x, node, rotated) = best?;
        let (rw, rh) = if rotated { (h, w) } else { (w, h) };
        let y = sky.fit(node, rw, rh)?;
        sky.insert(node, rw, y + rh);
        out[i] = Placed { x, y, rotated };
    }
    Some(out)
}

/// Skyline bin packer (y grows downward); nodes are (x, y, width) segments.
struct Skyline {
    width: u32,
    height: u32,
    nodes: Vec<(u32, u32, u32)>,
}

impl Skyline {
    fn new(width: u32, height: u32) -> Self {
        Skyline { width, height, nodes: vec![(0, 0, width)] }
    }

    /// Top y at which a w x h rect fits starting at node `i`.
    fn fit(&self, i: usize, w: u32, h: u32) -> Option<u32> {
        let x = self.nodes[i].0;
        if x + w > self.width {
            return None;
        }
        let (mut y, mut covered) = (0, 0);
        for &(_, ny, nw) in &self.nodes[i..] {
            if covered >= w {
                break;
            }
            y = y.max(ny);
            covered += nw;
        }
        (y + h <= self.height).then_some(y)
    }

    fn insert(&mut self, i: usize, w: u32, bottom: u32) {
        let x = self.nodes[i].0;
        self.nodes.insert(i, (x, bottom, w));
        let end = x + w;
        while i + 1 < self.nodes.len() {
            let (nx, ny, nw) = self.nodes[i + 1];
            if nx >= end {
                break;
            }
            if nx + nw <= end {
                self.nodes.remove(i + 1);
            } else {
                self.nodes[i + 1] = (end, ny, nx + nw - end);
                break;
            }
        }
        let mut k = 0;
        while k + 1 < self.nodes.len() {
            if self.nodes[k].1 == self.nodes[k + 1].1 {
                self.nodes[k].2 += self.nodes[k + 1].2;
                self.nodes.remove(k + 1);
            } else {
                k += 1;
            }
        }
    }
}
