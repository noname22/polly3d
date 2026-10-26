//! Symmetric editing across the object's X = 0 plane: vertices are paired
//! with their mirror image by position, and an edit made to one side is
//! copied, mirrored, to the other.

use std::collections::HashMap;

use glam::Vec3;

use crate::mesh::{Mesh, VertId};

/// Matching tolerance for mirror partners: a small fraction of the mesh size.
pub fn tolerance(mesh: &Mesh) -> f32 {
    let (lo, hi) = mesh.verts.values().fold((Vec3::MAX, Vec3::MIN), |(lo, hi), v| (lo.min(v.pos), hi.max(v.pos)));
    ((hi - lo).max_element() * 1e-3).max(1e-5)
}

fn mirror(p: Vec3) -> Vec3 {
    Vec3::new(-p.x, p.y, p.z)
}

/// Each vertex's mirror partner across X = 0 (itself for vertices on the
/// plane); vertices without a partner are left out.
pub fn partners(mesh: &Mesh, tol: f32) -> HashMap<VertId, VertId> {
    // Grid hash on positions for near-linear matching.
    let cell = tol * 4.0;
    let key = |p: Vec3| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
    let mut grid: HashMap<(i64, i64, i64), Vec<VertId>> = HashMap::new();
    for (id, v) in &mesh.verts {
        grid.entry(key(v.pos)).or_default().push(*id);
    }
    let mut out = HashMap::new();
    for (id, v) in &mesh.verts {
        let want = mirror(v.pos);
        let (kx, ky, kz) = key(want);
        let mut best: Option<(f32, VertId)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for w in grid.get(&(kx + dx, ky + dy, kz + dz)).into_iter().flatten() {
                        let d = mesh.verts[w].pos.distance(want);
                        if d <= tol && best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, *w));
                        }
                    }
                }
            }
        }
        if let Some((_, w)) = best {
            out.insert(*id, w);
        }
    }
    out
}

/// Report of [`mirror_edit`].
#[derive(Debug, Default, Clone)]
pub struct MirrorReport {
    /// Partners that were moved to mirror an edited vertex.
    pub mirrored: usize,
    /// Edited vertices without a partner (their other side stayed put).
    pub unpaired: Vec<VertId>,
}

/// Make an edit symmetric: `before` is the mesh before the edit, `after` the
/// edited mesh (same vertex ids). Each moved vertex's partner gets its
/// mirrored position; when both sides moved, the +X side leads. Vertices that
/// were on the plane stay on it.
pub fn mirror_edit(before: &Mesh, after: &mut Mesh) -> MirrorReport {
    let tol = tolerance(before);
    let pairs = partners(before, tol);
    let moved: Vec<VertId> = after
        .verts
        .iter()
        .filter(|(id, v)| before.verts.get(id).is_some_and(|b| b.pos.distance(v.pos) > 1e-7))
        .map(|(id, _)| *id)
        .collect();
    let moved_set: std::collections::HashSet<VertId> = moved.iter().copied().collect();
    let mut report = MirrorReport::default();
    let mut writes = Vec::new();
    for v in &moved {
        match pairs.get(v) {
            Some(w) if w == v => {
                // On the plane: keep it there.
                writes.push((*v, Vec3 { x: 0.0, ..after.verts[v].pos }));
            }
            Some(w) => {
                // The +X side leads when both moved.
                if moved_set.contains(w) && before.verts[v].pos.x < 0.0 {
                    continue;
                }
                writes.push((*w, mirror(after.verts[v].pos)));
                report.mirrored += 1;
            }
            None => report.unpaired.push(*v),
        }
    }
    for (id, p) in writes {
        if let Some(v) = after.verts.get_mut(&id) {
            v.pos = p;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Primitive;

    #[test]
    fn edits_are_mirrored_to_partners() {
        let m = Primitive::Cylinder { radius_bottom: 1.0, radius_top: 1.0, height: 2.0, segments: 8, height_segments: 2, caps: false }.build().unwrap();
        let pairs = partners(&m, tolerance(&m));
        assert_eq!(pairs.len(), m.verts.len(), "a symmetric cylinder pairs every vertex");
        // Move one +X vertex outward and up: its partner follows, mirrored.
        let (v, p) = m.verts.iter().find(|(_, v)| v.pos.x > 0.5).map(|(i, v)| (*i, v.pos)).unwrap();
        let w = pairs[&v];
        let mut after = m.clone();
        after.verts.get_mut(&v).unwrap().pos = p + Vec3::new(0.3, 0.2, 0.0);
        let r = mirror_edit(&m, &mut after);
        assert_eq!(r.mirrored, 1);
        assert!(after.verts[&w].pos.abs_diff_eq(mirror(p + Vec3::new(0.3, 0.2, 0.0)), 1e-6));
        // A vertex on the plane stays on it.
        let (c, cp) = m.verts.iter().find(|(_, v)| v.pos.x.abs() < 1e-6).map(|(i, v)| (*i, v.pos)).unwrap();
        let mut after = m.clone();
        after.verts.get_mut(&c).unwrap().pos = cp + Vec3::new(0.4, 0.1, 0.0);
        mirror_edit(&m, &mut after);
        assert!(after.verts[&c].pos.abs_diff_eq(cp + Vec3::new(0.0, 0.1, 0.0), 1e-6));
    }
}
