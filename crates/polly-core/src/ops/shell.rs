//! Offsetting surfaces along their normals: inflate vertices in place, or
//! copy faces into a new shell (clothing made from the body it covers).

use std::collections::{BTreeSet, HashMap};

use glam::Vec3;

use crate::error::{Result, invalid};
use crate::mesh::{Edge, FaceId, Mesh, VertId, newell_vector};

/// Smooth vertex normals: area-weighted normals of all faces around each vertex.
pub fn vertex_normals(mesh: &Mesh) -> HashMap<VertId, Vec3> {
    let mut n: HashMap<VertId, Vec3> = HashMap::new();
    for f in mesh.faces.values() {
        let a = newell_vector(&mesh.face_positions(f));
        for v in &f.verts {
            *n.entry(*v).or_default() += a;
        }
    }
    n.into_iter().map(|(v, a)| (v, a.normalize_or_zero())).collect()
}

/// A plane vertices may not cross: those on it stay on it (a garment's
/// centre line), the others stop at it.
#[derive(Debug, Clone, Copy)]
pub struct KeepPlane {
    pub point: Vec3,
    pub normal: Vec3,
}

impl KeepPlane {
    fn constrain(&self, from: Vec3, to: Vec3) -> Vec3 {
        let n = self.normal.normalize_or_zero();
        let (d0, d1) = ((from - self.point).dot(n), (to - self.point).dot(n));
        let eps = 1e-4;
        if d0.abs() < eps || d0 * d1 < 0.0 {
            to - n * d1
        } else {
            to
        }
    }
}

/// Move `verts` along their smooth normals by `distance` (negative shrinks).
pub fn inflate(mesh: &mut Mesh, verts: &BTreeSet<VertId>, distance: f32, keep: Option<KeepPlane>) -> Result<()> {
    if !distance.is_finite() {
        return invalid("distance must be a number");
    }
    let normals = vertex_normals(mesh);
    for v in verts {
        let Some(n) = normals.get(v) else { continue };
        let p = mesh.verts[v].pos;
        let to = p + *n * distance;
        mesh.verts.get_mut(v).unwrap().pos = keep.map_or(to, |k| k.constrain(p, to));
    }
    Ok(())
}

/// A new mesh from `faces` of `mesh`, pushed out by `offset` along the
/// source's smooth vertex normals. Corners (uv, colour), materials, parts,
/// smoothing, seams, sharp edges, skin weights and morphs are carried over.
pub fn shell(mesh: &Mesh, faces: &BTreeSet<FaceId>, offset: f32, keep: Option<KeepPlane>) -> Result<Mesh> {
    if faces.is_empty() {
        return invalid("no faces given");
    }
    let normals = vertex_normals(mesh);
    let mut out = Mesh::new();
    let mut map: HashMap<VertId, VertId> = HashMap::new();
    for fid in faces {
        let Some(f) = mesh.faces.get(fid) else { return invalid(format!("no face f{fid}")) };
        let verts: Vec<VertId> = f
            .verts
            .iter()
            .map(|v| {
                *map.entry(*v).or_insert_with(|| {
                    let p = mesh.verts[v].pos;
                    let to = p + normals[v] * offset;
                    out.add_vert(keep.map_or(to, |k| k.constrain(p, to)))
                })
            })
            .collect();
        let nf = out.add_face_with(verts, f.corners.clone(), f.material, f.smooth)?;
        out.faces.get_mut(&nf).unwrap().part = f.part.clone();
    }
    let edge = |e: &Edge| Some(Edge::new(*map.get(&e.0)?, *map.get(&e.1)?));
    out.seams = mesh.seams.iter().filter_map(edge).collect();
    out.sharp = mesh.sharp.iter().filter_map(edge).collect();
    out.weights = mesh.weights.iter().filter_map(|(v, w)| Some((*map.get(v)?, w.clone()))).collect();
    out.morphs = mesh
        .morphs
        .iter()
        .map(|m| crate::anim::Morph {
            name: m.name.clone(),
            offsets: m.offsets.iter().filter_map(|(v, d)| Some((*map.get(v)?, *d))).collect(),
        })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;

    #[test]
    fn shell_offsets_along_smooth_normals_and_keeps_the_centre_plane() {
        let m = cube(Vec3::splat(2.0));
        let all: BTreeSet<FaceId> = m.faces.keys().copied().collect();
        let s = shell(&m, &all, 0.1, None).unwrap();
        assert_eq!(s.faces.len(), 6);
        // Corners move out along the diagonal: |p| grows by 0.1.
        for v in s.verts.values() {
            assert!((v.pos.length() - (3f32.sqrt() + 0.1)).abs() < 1e-4, "{}", v.pos);
        }
        // Inflate the cube's +X half with a keep plane at x = 0.5: nothing crosses it.
        let mut half = m.clone();
        let right: BTreeSet<VertId> = half.verts.iter().filter(|(_, v)| v.pos.x > 0.0).map(|(i, _)| *i).collect();
        inflate(&mut half, &right, -2.0, Some(KeepPlane { point: Vec3::new(0.5, 0.0, 0.0), normal: Vec3::X })).unwrap();
        assert!(half.verts.values().filter(|v| v.pos.x > 0.0).all(|v| v.pos.x >= 0.5 - 1e-4));
        // A vertex on the plane slides along it.
        let mut flat = m.clone();
        let v = *flat.verts.keys().next().unwrap();
        let p = flat.verts[&v].pos;
        inflate(&mut flat, &BTreeSet::from([v]), 0.3, Some(KeepPlane { point: p, normal: Vec3::X })).unwrap();
        assert!((flat.verts[&v].pos.x - p.x).abs() < 1e-5);
    }
}
