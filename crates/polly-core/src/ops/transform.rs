use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::cmp::Reverse;

use glam::{Quat, Vec3};

use crate::error::{Result, invalid};
use crate::mesh::{Mesh, VertId};

pub fn centroid(mesh: &Mesh, verts: &BTreeSet<VertId>) -> Vec3 {
    if verts.is_empty() {
        return Vec3::ZERO;
    }
    verts.iter().map(|v| mesh.pos(*v)).sum::<Vec3>() / verts.len() as f32
}

/// Center of the bounding box of the given vertices.
pub fn bounds_center(mesh: &Mesh, verts: &BTreeSet<VertId>) -> Vec3 {
    let mut it = verts.iter().map(|v| mesh.pos(*v));
    let Some(first) = it.next() else { return Vec3::ZERO };
    let (lo, hi) = it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
    (lo + hi) / 2.0
}

fn require(verts: &BTreeSet<VertId>) -> Result<()> {
    if verts.is_empty() { invalid("no vertices selected") } else { Ok(()) }
}

pub fn translate(mesh: &mut Mesh, verts: &BTreeSet<VertId>, offset: Vec3) -> Result<()> {
    require(verts)?;
    for v in verts {
        mesh.verts.get_mut(v).unwrap().pos += offset;
    }
    Ok(())
}

pub fn rotate(mesh: &mut Mesh, verts: &BTreeSet<VertId>, rotation: Quat, pivot: Vec3) -> Result<()> {
    require(verts)?;
    for v in verts {
        let p = &mut mesh.verts.get_mut(v).unwrap().pos;
        *p = pivot + rotation * (*p - pivot);
    }
    Ok(())
}

pub fn scale(mesh: &mut Mesh, verts: &BTreeSet<VertId>, factor: Vec3, pivot: Vec3) -> Result<()> {
    require(verts)?;
    for v in verts {
        let p = &mut mesh.verts.get_mut(v).unwrap().pos;
        *p = pivot + factor * (*p - pivot);
    }
    if factor.x * factor.y * factor.z < 0.0 {
        // Negative scale mirrors: flip faces lying entirely inside the selection.
        for f in mesh.faces.values_mut() {
            if f.verts.iter().all(|v| verts.contains(v)) {
                f.reverse();
            }
        }
    }
    Ok(())
}

/// Set absolute coordinates; `None` components are left alone.
pub fn set_position(mesh: &mut Mesh, verts: &BTreeSet<VertId>, pos: [Option<f32>; 3]) -> Result<()> {
    require(verts)?;
    for v in verts {
        let p = &mut mesh.verts.get_mut(v).unwrap().pos;
        for (i, c) in pos.iter().enumerate() {
            if let Some(c) = c {
                p[i] = *c;
            }
        }
    }
    Ok(())
}

pub fn snap(mesh: &mut Mesh, verts: &BTreeSet<VertId>, step: f32) -> Result<()> {
    require(verts)?;
    if step <= 0.0 {
        return invalid("step must be > 0");
    }
    for v in verts {
        let p = &mut mesh.verts.get_mut(v).unwrap().pos;
        *p = (*p / step).round() * step;
    }
    Ok(())
}

/// Falloff curve for soft selection, as a function of t = 1 - distance / radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Falloff {
    /// Smooth S-curve (default).
    #[default]
    Smooth,
    Linear,
    /// Drops quickly: affects mostly the nearest vertices.
    Sharp,
    /// Stays strong, then drops at the edge (a broad bulge).
    Root,
}

impl Falloff {
    fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Falloff::Smooth => t * t * (3.0 - 2.0 * t),
            Falloff::Linear => t,
            Falloff::Sharp => t * t,
            Falloff::Root => t.sqrt(),
        }
    }
}

/// Soft-selection weights: 1 for `verts`, falling off to 0 at `radius` for
/// others. With `connected`, distance is measured along mesh edges, so parts
/// that are near in space but not attached are left alone.
pub fn soft_weights(mesh: &Mesh, verts: &BTreeSet<VertId>, radius: f32, curve: Falloff, connected: bool) -> BTreeMap<VertId, f32> {
    let mut w: BTreeMap<VertId, f32> = verts.iter().map(|v| (*v, 1.0)).collect();
    if radius <= 0.0 {
        return w;
    }
    let dist: HashMap<VertId, f32> = if connected {
        // Dijkstra over edges from all selected vertices.
        let ve = mesh.vert_edges();
        let mut best: HashMap<VertId, f32> = verts.iter().map(|v| (*v, 0.0)).collect();
        let mut heap: BinaryHeap<(Reverse<u32>, VertId)> = verts.iter().map(|v| (Reverse(0), *v)).collect();
        let key = |d: f32| (d * 1e6) as u32;
        while let Some((Reverse(dk), v)) = heap.pop() {
            let d = dk as f32 / 1e6;
            if d > best.get(&v).copied().unwrap_or(f32::MAX) + 1e-6 || d > radius {
                continue;
            }
            for e in ve.get(&v).into_iter().flatten() {
                let u = e.other(v);
                let nd = d + mesh.pos(v).distance(mesh.pos(u));
                if nd <= radius && nd < best.get(&u).copied().unwrap_or(f32::MAX) {
                    best.insert(u, nd);
                    heap.push((Reverse(key(nd)), u));
                }
            }
        }
        best
    } else {
        let sel: Vec<glam::Vec3> = verts.iter().map(|v| mesh.pos(*v)).collect();
        mesh.verts
            .iter()
            .filter_map(|(id, v)| {
                let d = sel.iter().map(|p| p.distance(v.pos)).fold(f32::MAX, f32::min);
                (d <= radius).then_some((*id, d))
            })
            .collect()
    };
    for (v, d) in dist {
        if !verts.contains(&v) {
            let k = curve.apply(1.0 - d / radius);
            if k > 1e-4 {
                w.insert(v, k);
            }
        }
    }
    w
}

/// Move each vertex by `offset` scaled by its weight.
pub fn translate_weighted(mesh: &mut Mesh, w: &BTreeMap<VertId, f32>, offset: Vec3) {
    for (v, k) in w {
        mesh.verts.get_mut(v).unwrap().pos += offset * *k;
    }
}

/// Rotate each vertex about `pivot` by the fraction `weight` of `rotation`.
pub fn rotate_weighted(mesh: &mut Mesh, w: &BTreeMap<VertId, f32>, rotation: Quat, pivot: Vec3) {
    for (v, k) in w {
        let q = Quat::IDENTITY.slerp(rotation, *k);
        let p = &mut mesh.verts.get_mut(v).unwrap().pos;
        *p = pivot + q * (*p - pivot);
    }
}

/// Scale each vertex about `pivot`, blending from 1 to `factor` by its weight.
pub fn scale_weighted(mesh: &mut Mesh, w: &BTreeMap<VertId, f32>, factor: Vec3, pivot: Vec3) {
    for (v, k) in w {
        let f = Vec3::ONE.lerp(factor, *k);
        let p = &mut mesh.verts.get_mut(v).unwrap().pos;
        *p = pivot + f * (*p - pivot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_move_falls_off() {
        let mut m = crate::primitives::plane(glam::Vec2::new(4.0, 1.0), 8, 1);
        let centre: BTreeSet<VertId> = m.verts.iter().filter(|(_, v)| v.pos.x.abs() < 1e-5).map(|(i, _)| *i).collect();
        let w = soft_weights(&m, &centre, 1.0, Falloff::Linear, false);
        translate_weighted(&mut m, &w, Vec3::Y);
        let at = |x: f32| m.verts.values().find(|v| (v.pos.x - x).abs() < 1e-5).unwrap().pos.y;
        assert!((at(0.0) - 1.0).abs() < 1e-5);
        assert!((at(0.5) - 0.5).abs() < 1e-5);
        assert!(at(1.5).abs() < 1e-5 && at(-2.0).abs() < 1e-5);
    }

    #[test]
    fn connected_falloff_ignores_separate_parts() {
        let mut m = crate::primitives::cube(Vec3::ONE);
        let mut other = crate::primitives::cube(Vec3::ONE);
        other.transform(glam::Mat4::from_translation(Vec3::new(1.2, 0.0, 0.0)));
        m.append(&other);
        let a: BTreeSet<VertId> = [*m.verts.keys().next().unwrap()].into();
        let near = soft_weights(&m, &a, 3.5, Falloff::Linear, false);
        let conn = soft_weights(&m, &a, 3.5, Falloff::Linear, true);
        assert_eq!(conn.len(), 8, "only the first cube");
        assert!(near.len() > 8);
    }
}
