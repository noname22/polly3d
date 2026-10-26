//! CPU-side scene preparation: world-space geometry, triangulation, smooth
//! normals and ray casting. Shared by the GPU renderer, labels and picking.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};
use polly_core::geom::triangulate_polygon;
use polly_core::mesh::{Edge, Face, FaceId, Mesh, VertId, newell_vector};
use polly_core::scene::{ObjectId, Scene};
use serde::{Deserialize, Serialize};

use crate::camera::CameraInfo;

pub(crate) struct PreparedObject {
    pub id: ObjectId,
    pub name: String,
    pub mesh: Arc<Mesh>,
    /// World-space vertex positions.
    pub pos: HashMap<VertId, Vec3>,
    /// Transform mirrors geometry (negative determinant): loops are reversed.
    pub flip: bool,
    /// Outward world-space face normals (unit).
    pub face_normal: BTreeMap<FaceId, Vec3>,
}

impl PreparedObject {
    pub fn face_center(&self, f: &Face) -> Vec3 {
        f.verts.iter().map(|v| self.pos[v]).sum::<Vec3>() / f.verts.len() as f32
    }
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.pos.values().copied();
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
    }
    /// Triangles of a face as corner-index triples, wound CCW-outward in world space.
    pub fn face_tris(&self, f: &Face) -> Vec<[usize; 3]> {
        let pts: Vec<Vec3> = f.verts.iter().map(|v| self.pos[v]).collect();
        let mut t = triangulate_polygon(&pts);
        if self.flip {
            for tri in &mut t {
                tri.swap(1, 2);
            }
        }
        t
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Tri {
    pub obj: usize,
    pub face: FaceId,
    pub p: [Vec3; 3],
    pub uv: [Vec2; 3],
    pub n: Vec3,
    pub two_sided: bool,
}

pub(crate) struct Prepared {
    pub objects: Vec<PreparedObject>,
    pub tris: Vec<Tri>,
}

pub(crate) fn world_bounds(objs: &[&PreparedObject]) -> Option<(Vec3, Vec3)> {
    objs.iter().filter_map(|o| o.bounds()).reduce(|(a, b), (c, d)| (a.min(c), b.max(d)))
}

impl Prepared {
    /// Prepare the given objects (or every visible object when `ids` is None).
    /// Hidden objects are always skipped.
    pub fn new(scene: &Scene, ids: Option<&[ObjectId]>) -> Self {
        let mut objects = Vec::new();
        for (&id, o) in &scene.objects {
            if !o.visible || ids.is_some_and(|ids| !ids.contains(&id)) {
                continue;
            }
            let m: Mat4 = scene.world_matrix(id);
            let flip = m.determinant() < 0.0;
            let pos: HashMap<VertId, Vec3> = o.mesh.verts.iter().map(|(&k, v)| (k, m.transform_point3(v.pos))).collect();
            let face_normal = o
                .mesh
                .faces
                .iter()
                .filter(|(_, f)| f.verts.iter().all(|v| pos.contains_key(v)))
                .map(|(&fid, f)| {
                    let pts: Vec<Vec3> = f.verts.iter().map(|v| pos[v]).collect();
                    let n = newell_vector(&pts).normalize_or_zero();
                    (fid, if flip { -n } else { n })
                })
                .collect();
            objects.push(PreparedObject { id, name: o.name.clone(), mesh: o.mesh.clone(), pos, flip, face_normal });
        }
        let mut tris = Vec::new();
        for (oi, po) in objects.iter().enumerate() {
            for (&fid, f) in &po.mesh.faces {
                let Some(&n) = po.face_normal.get(&fid) else { continue };
                let two_sided = scene.materials.get(&f.material).is_some_and(|m| m.double_sided);
                for [a, b, c] in po.face_tris(f) {
                    tris.push(Tri {
                        obj: oi,
                        face: fid,
                        p: [po.pos[&f.verts[a]], po.pos[&f.verts[b]], po.pos[&f.verts[c]]],
                        uv: [f.corners[a].uv, f.corners[b].uv, f.corners[c].uv],
                        n,
                        two_sided,
                    });
                }
            }
        }
        Prepared { objects, tris }
    }

    /// Closest hit along a ray: (t, triangle index, barycentric b1, b2).
    /// Back faces of single-sided materials are ignored (they are culled when rendering).
    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<(f32, usize, f32, f32)> {
        let mut best: Option<(f32, usize, f32, f32)> = None;
        for (i, t) in self.tris.iter().enumerate() {
            if !t.two_sided && t.n.dot(dir) > 0.0 {
                continue;
            }
            if let Some((d, u, v)) = ray_tri(origin, dir, t.p) {
                if best.is_none_or(|b| d < b.0) {
                    best = Some((d, i, u, v));
                }
            }
        }
        best
    }

    /// Whether world point `p` is visible from the camera (nothing in front of it),
    /// with `own_face` (object index, face) counted as not occluding.
    pub fn visible(&self, cam: &CameraInfo, p: Vec3, own: Option<(usize, FaceId)>, eps: f32) -> bool {
        let (origin, dir) = cam.ray_to(p);
        let tp = (p - origin).dot(dir);
        match self.raycast(origin, dir) {
            None => true,
            Some((t, i, _, _)) => {
                let tri = &self.tris[i];
                own.is_some_and(|o| o == (tri.obj, tri.face)) || t >= tp - eps
            }
        }
    }

    /// Per-corner world normals for `face` honouring `face.smooth` and sharp edges.
    pub fn corner_normals(&self, oi: usize, fid: FaceId, vert_faces: &HashMap<VertId, Vec<FaceId>>) -> Vec<Vec3> {
        let po = &self.objects[oi];
        let mesh = &po.mesh;
        let f = &mesh.faces[&fid];
        let flat = po.face_normal[&fid];
        if !f.smooth {
            return vec![flat; f.len()];
        }
        let weighted = |g: FaceId| -> Vec3 {
            let gf = &mesh.faces[&g];
            let pts: Vec<Vec3> = gf.verts.iter().map(|v| po.pos[v]).collect();
            let n = newell_vector(&pts);
            if po.flip { -n } else { n }
        };
        f.verts
            .iter()
            .map(|&v| {
                let mut set: HashSet<FaceId> = HashSet::from([fid]);
                let mut stack = vec![fid];
                let around = vert_faces.get(&v).map(Vec::as_slice).unwrap_or(&[]);
                while let Some(g) = stack.pop() {
                    let gf = &mesh.faces[&g];
                    let n = gf.len();
                    let i = gf.index_of(v).unwrap();
                    for w in [gf.verts[(i + n - 1) % n], gf.verts[(i + 1) % n]] {
                        let e = Edge::new(v, w);
                        if mesh.sharp.contains(&e) {
                            continue;
                        }
                        for &h in around {
                            if set.contains(&h) || !po.face_normal.contains_key(&h) {
                                continue;
                            }
                            let hf = &mesh.faces[&h];
                            if hf.smooth && hf.edge_index(e).is_some() {
                                set.insert(h);
                                stack.push(h);
                            }
                        }
                    }
                }
                let sum: Vec3 = set.iter().map(|&g| weighted(g)).sum();
                sum.try_normalize().unwrap_or(flat)
            })
            .collect()
    }
}

/// Möller–Trumbore, double-sided. Returns (t, b1, b2).
pub(crate) fn ray_tri(o: Vec3, d: Vec3, p: [Vec3; 3]) -> Option<(f32, f32, f32)> {
    let e1 = p[1] - p[0];
    let e2 = p[2] - p[0];
    let h = d.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-12 {
        return None;
    }
    let f = 1.0 / a;
    let s = o - p[0];
    let u = f * s.dot(h);
    if !(-1e-6..=1.0 + 1e-6).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * d.dot(q);
    if v < -1e-6 || u + v > 1.0 + 1e-6 {
        return None;
    }
    let t = f * e2.dot(q);
    (t > 1e-6).then_some((t, u, v))
}

/// Result of [`pick`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Hit {
    pub object: ObjectId,
    pub object_name: String,
    pub face: FaceId,
    /// World-space hit point.
    pub position: [f32; 3],
    /// World-space face normal, facing the camera.
    pub normal: [f32; 3],
    /// Interpolated corner UV at the hit point.
    pub uv: [f32; 2],
    /// Distance from the ray origin (the eye for perspective cameras).
    pub distance: f32,
}

/// CPU ray cast through pixel (`px`, `py`) (top-left origin, pixel centres at +0.5)
/// against the triangulated faces of all visible objects.
pub fn pick(scene: &Scene, cam: &CameraInfo, px: f32, py: f32) -> Option<Hit> {
    let prep = Prepared::new(scene, None);
    let (o, d) = cam.ray(px, py);
    let (t, i, u, v) = prep.raycast(o, d)?;
    let tri = &prep.tris[i];
    let po = &prep.objects[tri.obj];
    let w = 1.0 - u - v;
    let uv = tri.uv[0] * w + tri.uv[1] * u + tri.uv[2] * v;
    let n = if tri.n.dot(d) > 0.0 { -tri.n } else { tri.n };
    Some(Hit {
        object: po.id,
        object_name: po.name.clone(),
        face: tri.face,
        position: (o + d * t).to_array(),
        normal: n.to_array(),
        uv: uv.to_array(),
        distance: t,
    })
}
