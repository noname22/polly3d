use std::collections::BTreeSet;

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
