//! Catmull–Clark subdivision with (semi-)sharp creases, as a modifier: the
//! object's mesh stays the editable cage (stable ids), and display / export
//! use a subdivided copy.
//!
//! The scheme is linear, so it is built as stencils (each new vertex is a
//! weighted sum of cage vertices) and applied to positions, morph offsets and
//! skin weights alike. Face-varying data (UVs, colours) is interpolated
//! bilinearly inside each face, which keeps UV seams exact. Boundary edges
//! are sharp. A crease of sharpness `s` is sharp for `floor(s)` levels and
//! blends for the fraction; each level lowers it by one.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, Weak};

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::mesh::{Corner, Edge, Face, FaceId, Mesh, VertId};
use crate::scene::Scene;

/// Highest supported level (each level multiplies faces by about 4).
pub const MAX_LEVEL: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Subdivision {
    /// Level shown in renders and the viewer (0 = the cage).
    pub level: u8,
    /// Level written by export_glb (default: `level`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_level: Option<u8>,
}

impl Subdivision {
    pub fn export(&self) -> u8 {
        self.export_level.unwrap_or(self.level)
    }
}

type Stencil = Vec<(VertId, f32)>;

/// One level of subdivision. Returns the new mesh and, per new face, the
/// cage face it came from.
pub fn subdivide_once(m: &Mesh) -> (Mesh, BTreeMap<FaceId, FaceId>) {
    let mut ef: HashMap<Edge, Vec<FaceId>> = HashMap::with_capacity(m.faces.len() * 2);
    for (&fid, f) in &m.faces {
        for e in f.edges() {
            ef.entry(e).or_default().push(fid);
        }
    }
    let crease = |e: &Edge| -> f32 {
        // Boundary and non-manifold edges are fully sharp.
        if ef.get(e).is_none_or(|f| f.len() != 2) { f32::INFINITY } else { m.creases.get(e).copied().unwrap_or(0.0) }
    };
    let mut out = Mesh::new();
    let mut stencils: Vec<Stencil> = Vec::new();
    let mut add = |out: &mut Mesh, st: Stencil| -> VertId {
        let p = st.iter().map(|(v, w)| m.verts[v].pos * *w).sum::<Vec3>();
        stencils.push(st);
        out.add_vert(p)
    };

    // Face points.
    let mut face_pt: HashMap<FaceId, (VertId, Stencil)> = HashMap::with_capacity(m.faces.len());
    for (&fid, f) in &m.faces {
        let w = 1.0 / f.len() as f32;
        let st: Stencil = f.verts.iter().map(|v| (*v, w)).collect();
        let id = add(&mut out, st.clone());
        face_pt.insert(fid, (id, st));
    }

    // Edge points.
    let mut edges: Vec<Edge> = ef.keys().copied().collect();
    edges.sort();
    let mut edge_pt: HashMap<Edge, VertId> = HashMap::with_capacity(edges.len());
    for e in &edges {
        let s = crease(e);
        let mid: Stencil = vec![(e.0, 0.5), (e.1, 0.5)];
        let st = if s >= 1.0 {
            mid
        } else {
            let fs = &ef[e];
            // Smooth: average of the ends and the adjacent face points.
            let mut smooth: Stencil = vec![(e.0, 0.25), (e.1, 0.25)];
            for f in fs {
                smooth.extend(face_pt[f].1.iter().map(|(v, w)| (*v, w * 0.25)));
            }
            if s > 0.0 { blend(&smooth, &mid, s) } else { smooth }
        };
        edge_pt.insert(*e, add(&mut out, compact(st)));
    }

    // Vertex points.
    let mut vert_edges: HashMap<VertId, Vec<Edge>> = HashMap::new();
    for e in &edges {
        vert_edges.entry(e.0).or_default().push(*e);
        vert_edges.entry(e.1).or_default().push(*e);
    }
    let mut vert_faces: HashMap<VertId, Vec<FaceId>> = HashMap::new();
    for (&fid, f) in &m.faces {
        for v in &f.verts {
            vert_faces.entry(*v).or_default().push(fid);
        }
    }
    let mut vert_pt: HashMap<VertId, VertId> = HashMap::with_capacity(m.verts.len());
    for &v in m.verts.keys() {
        let Some(es) = vert_edges.get(&v) else {
            vert_pt.insert(v, add(&mut out, vec![(v, 1.0)]));
            continue;
        };
        let n = es.len() as f32;
        let sharp: Vec<(&Edge, f32)> = es.iter().map(|e| (e, crease(e))).filter(|(_, s)| *s > 0.0).collect();
        let corner: Stencil = vec![(v, 1.0)];
        let smooth = || -> Stencil {
            // (F + 2R + (n - 3) P) / n
            let fs = &vert_faces[&v];
            let mut st: Stencil = vec![(v, (n - 3.0) / n)];
            let kf = 1.0 / (fs.len() as f32 * n);
            for f in fs {
                st.extend(face_pt[f].1.iter().map(|(u, w)| (*u, w * kf)));
            }
            let kr = 2.0 / (n * n);
            for e in es {
                st.push((e.0, kr * 0.5));
                st.push((e.1, kr * 0.5));
            }
            st
        };
        let st = match sharp.len() {
            0 | 1 => smooth(),
            2 => {
                let crease_rule: Stencil = vec![(v, 0.75), (sharp[0].0.other(v), 0.125), (sharp[1].0.other(v), 0.125)];
                let s = (sharp[0].1 + sharp[1].1) * 0.5;
                if s >= 1.0 { crease_rule } else { blend(&smooth(), &crease_rule, s) }
            }
            _ => {
                let s = sharp.iter().map(|(_, s)| s.min(1.0)).sum::<f32>() / sharp.len() as f32;
                if s >= 1.0 { corner } else { blend(&smooth(), &corner, s) }
            }
        };
        vert_pt.insert(v, add(&mut out, compact(st)));
    }

    // Faces: one quad per cage corner.
    let mut parent = BTreeMap::new();
    for (&fid, f) in &m.faces {
        let n = f.len();
        let fp = face_pt[&fid].0;
        let centre = avg_corner(&f.corners);
        for i in 0..n {
            let prev = (i + n - 1) % n;
            let next = (i + 1) % n;
            let (v, vn, vp) = (f.verts[i], f.verts[next], f.verts[prev]);
            let verts = vec![vert_pt[&v], edge_pt[&Edge::new(v, vn)], fp, edge_pt[&Edge::new(vp, v)]];
            let c = &f.corners;
            let corners = vec![c[i], c[i].lerp(&c[next], 0.5), centre, c[prev].lerp(&c[i], 0.5)];
            let child = out.push_face_unchecked(Face { verts, corners, material: f.material, smooth: f.smooth, part: f.part.clone() });
            parent.insert(child, fid);
        }
    }

    // Edge sets: halves of cage edges inherit; creases lose one level.
    for e in &edges {
        let mid = edge_pt[e];
        let halves = [Edge::new(vert_pt[&e.0], mid), Edge::new(mid, vert_pt[&e.1])];
        if let Some(s) = m.creases.get(e) {
            if *s > 1.0 {
                for h in halves {
                    out.creases.insert(h, s - 1.0);
                }
            }
        }
        if m.sharp.contains(e) {
            out.sharp.extend(halves);
        }
        if m.seams.contains(e) {
            out.seams.extend(halves);
        }
    }

    // Skin weights and morphs through the same stencils.
    if !m.weights.is_empty() {
        for (i, st) in stencils.iter().enumerate() {
            let mut acc: BTreeMap<crate::anim::BoneId, f32> = BTreeMap::new();
            for (v, w) in st {
                for (b, bw) in m.weights.get(v).into_iter().flatten() {
                    *acc.entry(*b).or_default() += bw * w;
                }
            }
            let mut list: Vec<_> = acc.into_iter().filter(|(_, w)| *w > 1e-5).collect();
            list.sort_by(|a, b| b.1.total_cmp(&a.1));
            list.truncate(4);
            if !list.is_empty() {
                out.weights.insert(i as VertId, list);
            }
        }
    }
    for morph in &m.morphs {
        let mut offsets = BTreeMap::new();
        for (i, st) in stencils.iter().enumerate() {
            let d: Vec3 = st.iter().filter_map(|(v, w)| morph.offsets.get(v).map(|o| *o * *w)).sum();
            if d.length_squared() > 0.0 {
                offsets.insert(i as VertId, d);
            }
        }
        out.morphs.push(crate::anim::Morph { name: morph.name.clone(), offsets });
    }
    (out, parent)
}

fn blend(a: &Stencil, b: &Stencil, t: f32) -> Stencil {
    a.iter().map(|(v, w)| (*v, w * (1.0 - t))).chain(b.iter().map(|(v, w)| (*v, w * t))).collect()
}

fn compact(st: Stencil) -> Stencil {
    let mut m: BTreeMap<VertId, f32> = BTreeMap::new();
    for (v, w) in st {
        *m.entry(v).or_default() += w;
    }
    m.into_iter().filter(|(_, w)| w.abs() > 1e-9).collect()
}

fn avg_corner(c: &[Corner]) -> Corner {
    let k = 1.0 / c.len() as f32;
    Corner { uv: c.iter().map(|c| c.uv).sum::<Vec2>() * k, color: c.iter().map(|c| c.color).sum::<Vec4>() * k }
}

/// `level` levels of subdivision; the face map points at the cage faces.
pub fn subdivide(m: &Mesh, level: u8) -> (Mesh, BTreeMap<FaceId, FaceId>) {
    let mut cur = m.clone();
    let mut map: BTreeMap<FaceId, FaceId> = m.faces.keys().map(|f| (*f, *f)).collect();
    for _ in 0..level.min(MAX_LEVEL) {
        let (next, parent) = subdivide_once(&cur);
        map = parent.into_iter().map(|(c, p)| (c, map[&p])).collect();
        cur = next;
    }
    (cur, map)
}

type CacheEntry = (Weak<crate::mesh::Mesh>, u8, Arc<Mesh>);
static CACHE: Mutex<Vec<CacheEntry>> = Mutex::new(Vec::new());

/// Subdivided copy of a shared mesh, cached while the cage is alive and
/// unchanged (repeated renders reuse it, and its `Arc` stays the same).
pub fn subdivided(mesh: &Arc<Mesh>, level: u8) -> Arc<Mesh> {
    if level == 0 {
        return mesh.clone();
    }
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.retain(|(w, _, _)| w.strong_count() > 0);
    if let Some((_, _, out)) = cache.iter().find(|(w, l, _)| *l == level && w.upgrade().is_some_and(|a| Arc::ptr_eq(&a, mesh))) {
        return out.clone();
    }
    let out = Arc::new(subdivide(mesh, level).0);
    cache.push((Arc::downgrade(mesh), level, out.clone()));
    if cache.len() > 64 {
        cache.remove(0);
    }
    out
}

/// The subdivision level of an object for display (with an optional
/// override; 0 = the cage).
pub fn display_level(scene: &Scene, id: crate::scene::ObjectId, over: Option<u8>) -> u8 {
    over.unwrap_or_else(|| scene.objects.get(&id).and_then(|o| o.subdivision).map_or(0, |s| s.level)).min(MAX_LEVEL)
}

/// The scene with every subdivided object's mesh replaced by its subdivided
/// copy at its export level (or `over`), hide-rule face sets remapped to the
/// child faces. Used for export.
pub fn apply_for_export(scene: &Scene, over: Option<u8>) -> Scene {
    let mut out = scene.clone();
    for (id, o) in &scene.objects {
        let level = over.unwrap_or_else(|| o.subdivision.map_or(0, |s| s.export())).min(MAX_LEVEL);
        if level == 0 || o.mesh.faces.is_empty() {
            continue;
        }
        let (mesh, parent) = subdivide(&o.mesh, level);
        let mut children: HashMap<FaceId, Vec<FaceId>> = HashMap::new();
        for (c, p) in &parent {
            children.entry(*p).or_default().push(*c);
        }
        for slot in out.slots.values_mut() {
            for v in &mut slot.variants {
                for h in v.hides.iter_mut().filter(|h| h.object == *id) {
                    if let Some(set) = &h.faces {
                        let mapped: BTreeSet<FaceId> = set.iter().flat_map(|f| children.get(f).into_iter().flatten().copied()).collect();
                        h.faces = Some(mapped);
                    }
                }
            }
        }
        let ob = out.objects.get_mut(id).unwrap();
        ob.mesh = Arc::new(mesh);
        ob.selection = Default::default();
        ob.morph_edit = None;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;

    #[test]
    fn cube_converges_inside_its_cage() {
        let m = cube(Vec3::splat(2.0));
        let (s1, map) = subdivide(&m, 1);
        assert_eq!((s1.verts.len(), s1.faces.len()), (26, 24));
        assert_eq!(map.len(), 24);
        let p = s1.problems();
        assert!(p.nonmanifold == 0 && p.flipped == 0 && p.degenerate.is_empty() && p.loose == 0, "{:?}", s1.validate());
        let (s2, _) = subdivide(&m, 2);
        assert_eq!(s2.faces.len(), 96);
        // A smooth cube shrinks towards a rounded shape inside the cage (half size 1).
        let r = s2.verts.values().map(|v| v.pos.length()).fold(0.0f32, f32::max);
        assert!(r < 1.0 && r > 0.75, "max radius {r}");
    }

    #[test]
    fn creases_keep_edges_sharp() {
        let mut m = cube(Vec3::splat(2.0));
        let all: Vec<Edge> = m.edges().into_iter().collect();
        for e in all {
            m.creases.insert(e, 10.0);
        }
        let (s, _) = subdivide(&m, 2);
        // Fully creased cube keeps its shape: every vertex stays on the cube's surface.
        for v in s.verts.values() {
            let p = v.pos.abs();
            assert!((p.max_element() - 1.0).abs() < 1e-5, "{p:?}");
        }
        // Semi-sharp creases land between smooth and sharp.
        let mut half = cube(Vec3::splat(2.0));
        let edges: Vec<Edge> = half.edges().into_iter().collect();
        for e in edges {
            half.creases.insert(e, 0.5);
        }
        let corner = |m: &Mesh| subdivide(m, 1).0.verts.values().map(|v| v.pos.length()).fold(0.0f32, f32::max);
        let (smooth, semi, sharp) = (corner(&cube(Vec3::splat(2.0))), corner(&half), corner(&m));
        assert!(smooth < semi && semi < sharp, "{smooth} {semi} {sharp}");
    }

    #[test]
    fn open_plane_keeps_its_boundary_and_uvs() {
        let mut m = crate::primitives::plane(Vec2::splat(2.0), 2, 2);
        for f in m.faces.values_mut() {
            for c in &mut f.corners {
                c.uv = Vec2::new(0.25, 0.75);
            }
        }
        let (s, _) = subdivide(&m, 1);
        // Boundary is sharp: the outline stays a 2x2 square.
        let (lo, hi) = s.bounds().unwrap();
        assert!((lo.x + 1.0).abs() < 1e-5 && (hi.z - 1.0).abs() < 1e-5);
        assert!(s.faces.values().flat_map(|f| &f.corners).all(|c| c.uv == Vec2::new(0.25, 0.75)));
    }

    #[test]
    fn weights_and_morphs_follow() {
        let mut m = cube(Vec3::splat(2.0));
        let top: Vec<VertId> = m.verts.iter().filter(|(_, v)| v.pos.y > 0.0).map(|(i, _)| *i).collect();
        for (i, _) in m.verts.clone() {
            m.weights.insert(i, vec![(if top.contains(&i) { 1 } else { 0 }, 1.0)]);
        }
        m.morphs.push(crate::anim::Morph { name: "up".into(), offsets: top.iter().map(|v| (*v, Vec3::Y)).collect() });
        let (s, _) = subdivide(&m, 1);
        for (v, w) in &s.weights {
            let sum: f32 = w.iter().map(|x| x.1).sum();
            assert!((sum - 1.0).abs() < 1e-4, "v{v}: {w:?}");
        }
        // Moving the cage by the morph = subdividing the moved cage.
        let mut moved = m.clone();
        for v in &top {
            moved.verts.get_mut(v).unwrap().pos += Vec3::Y;
        }
        let sm = subdivide(&moved, 1).0;
        for (i, v) in &s.verts {
            let off = s.morphs[0].offsets.get(i).copied().unwrap_or(Vec3::ZERO);
            assert!((v.pos + off - sm.verts[i].pos).length() < 1e-5);
        }
    }

    #[test]
    fn cache_reuses_until_the_cage_changes() {
        let a = Arc::new(cube(Vec3::ONE));
        let s1 = subdivided(&a, 1);
        assert!(Arc::ptr_eq(&s1, &subdivided(&a, 1)));
        assert!(!Arc::ptr_eq(&s1, &subdivided(&a, 2)));
        let b = Arc::new(cube(Vec3::ONE));
        assert!(!Arc::ptr_eq(&s1, &subdivided(&b, 1)));
    }
}
