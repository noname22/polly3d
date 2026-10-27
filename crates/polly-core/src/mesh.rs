//! Polygon (n-gon) mesh with stable, never-reused integer IDs.
//!
//! Faces are ordered vertex loops, counter-clockwise when viewed from the
//! outside. Per-corner attributes (UV, color) live on the face. Edges are not
//! stored; they are derived from face loops as sorted vertex pairs.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result, invalid};

pub type VertId = u32;
pub type FaceId = u32;

/// Undirected edge, always stored with `0 < 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Edge(pub VertId, pub VertId);

impl Edge {
    pub fn new(a: VertId, b: VertId) -> Self {
        if a < b { Edge(a, b) } else { Edge(b, a) }
    }
    pub fn other(&self, v: VertId) -> VertId {
        if self.0 == v { self.1 } else { self.0 }
    }
    pub fn has(&self, v: VertId) -> bool {
        self.0 == v || self.1 == v
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub pos: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Corner {
    pub uv: Vec2,
    pub color: Vec4,
}

impl Default for Corner {
    fn default() -> Self {
        Corner { uv: Vec2::ZERO, color: Vec4::ONE }
    }
}

impl Corner {
    pub fn lerp(&self, other: &Corner, t: f32) -> Corner {
        Corner { uv: self.uv.lerp(other.uv, t), color: self.color.lerp(other.color, t) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Face {
    pub verts: Vec<VertId>,
    pub corners: Vec<Corner>,
    #[serde(default)]
    pub material: u32,
    #[serde(default)]
    pub smooth: bool,
    /// Named part this face belongs to (e.g. "head" after joining objects).
    /// Faces created from a face inherit its part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
}

impl Face {
    pub fn len(&self) -> usize {
        self.verts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.verts.is_empty()
    }
    /// Directed edges in loop order.
    pub fn half_edges(&self) -> impl Iterator<Item = (VertId, VertId)> + '_ {
        let n = self.verts.len();
        (0..n).map(move |i| (self.verts[i], self.verts[(i + 1) % n]))
    }
    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.half_edges().map(|(a, b)| Edge::new(a, b))
    }
    pub fn index_of(&self, v: VertId) -> Option<usize> {
        self.verts.iter().position(|&x| x == v)
    }
    /// Loop index `i` such that `verts[i], verts[i+1]` is the edge (either direction).
    pub fn edge_index(&self, e: Edge) -> Option<usize> {
        let n = self.verts.len();
        (0..n).find(|&i| Edge::new(self.verts[i], self.verts[(i + 1) % n]) == e)
    }
    /// True if the face traverses `a -> b`.
    pub fn has_half_edge(&self, a: VertId, b: VertId) -> bool {
        self.half_edges().any(|h| h == (a, b))
    }
    pub fn reverse(&mut self) {
        self.verts.reverse();
        self.corners.reverse();
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Mesh {
    pub verts: BTreeMap<VertId, Vertex>,
    pub faces: BTreeMap<FaceId, Face>,
    /// UV seams (used by unwrapping).
    #[serde(default)]
    pub seams: BTreeSet<Edge>,
    /// Hard edges for smooth shading.
    #[serde(default)]
    pub sharp: BTreeSet<Edge>,
    /// Subdivision creases: sharpness per edge (1 = sharp for one level,
    /// >= the subdivision level = fully sharp; fractions blend).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub creases: BTreeMap<Edge, f32>,
    /// Skin weights: per vertex, (bone, weight) pairs (see `anim`).
    #[serde(default)]
    pub weights: BTreeMap<VertId, Vec<(crate::anim::BoneId, f32)>>,
    /// Morph targets (shape keys).
    #[serde(default)]
    pub morphs: Vec<crate::anim::Morph>,
    next_vert: VertId,
    next_face: FaceId,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- construction -------------------------------------------------

    pub fn add_vert(&mut self, pos: Vec3) -> VertId {
        let id = self.next_vert;
        self.next_vert += 1;
        self.verts.insert(id, Vertex { pos });
        id
    }

    pub fn add_face(&mut self, verts: Vec<VertId>) -> Result<FaceId> {
        let corners = vec![Corner::default(); verts.len()];
        self.add_face_with(verts, corners, 0, false)
    }

    pub fn add_face_with(
        &mut self,
        verts: Vec<VertId>,
        corners: Vec<Corner>,
        material: u32,
        smooth: bool,
    ) -> Result<FaceId> {
        if verts.len() < 3 {
            return invalid(format!("a face needs at least 3 vertices, got {}", verts.len()));
        }
        if corners.len() != verts.len() {
            return invalid("corner count must match vertex count");
        }
        let unique: BTreeSet<_> = verts.iter().collect();
        if unique.len() != verts.len() {
            return invalid("face uses the same vertex twice");
        }
        for v in &verts {
            if !self.verts.contains_key(v) {
                return Err(Error::NotFound { kind: "vertex", id: crate::ids::vert(*v) });
            }
        }
        let id = self.next_face;
        self.next_face += 1;
        self.faces.insert(id, Face { verts, corners, material, smooth, part: None });
        Ok(id)
    }

    /// Insert a face without validation (callers guarantee a valid loop).
    pub(crate) fn push_face_unchecked(&mut self, face: Face) -> FaceId {
        let id = self.next_face;
        self.next_face += 1;
        self.faces.insert(id, face);
        id
    }

    /// Add a face taking material, smoothing and part from a template face.
    pub fn add_face_from(&mut self, verts: Vec<VertId>, corners: Vec<Corner>, template: &Face) -> Result<FaceId> {
        let id = self.add_face_with(verts, corners, template.material, template.smooth)?;
        self.faces.get_mut(&id).unwrap().part = template.part.clone();
        Ok(id)
    }

    /// Like [`add_face_from`](Self::add_face_from), with default corners
    /// coloured like the template's first corner.
    pub fn add_face_like(&mut self, verts: Vec<VertId>, template: &Face) -> Result<FaceId> {
        let c = Corner { uv: Vec2::ZERO, color: template.corners.first().map_or(Vec4::ONE, |c| c.color) };
        let n = verts.len();
        self.add_face_from(verts, vec![c; n], template)
    }

    /// Append another mesh; returns old->new vertex id map.
    pub fn append(&mut self, other: &Mesh) -> (HashMap<VertId, VertId>, Vec<FaceId>) {
        let mut vmap = HashMap::new();
        for (&id, v) in &other.verts {
            vmap.insert(id, self.add_vert(v.pos));
        }
        let mut faces = Vec::new();
        for f in other.faces.values() {
            let verts = f.verts.iter().map(|v| vmap[v]).collect();
            faces.push(self.add_face_from(verts, f.corners.clone(), f).unwrap());
        }
        for e in &other.seams {
            self.seams.insert(Edge::new(vmap[&e.0], vmap[&e.1]));
        }
        for e in &other.sharp {
            self.sharp.insert(Edge::new(vmap[&e.0], vmap[&e.1]));
        }
        for (v, list) in &other.weights {
            self.weights.insert(vmap[v], list.clone());
        }
        for m in &other.morphs {
            let offsets = m.offsets.iter().map(|(v, d)| (vmap[v], *d));
            match self.morphs.iter_mut().find(|x| x.name == m.name) {
                Some(x) => x.offsets.extend(offsets),
                None => self.morphs.push(crate::anim::Morph { name: m.name.clone(), offsets: offsets.collect() }),
            }
        }
        (vmap, faces)
    }

    // ---- removal ------------------------------------------------------

    pub fn remove_face(&mut self, f: FaceId) -> Option<Face> {
        self.faces.remove(&f)
    }

    /// Remove a vertex and every face using it.
    pub fn remove_vert(&mut self, v: VertId) {
        self.verts.remove(&v);
        self.faces.retain(|_, f| !f.verts.contains(&v));
    }

    /// Remove vertices not referenced by any face. Returns removed ids.
    pub fn remove_loose_verts(&mut self) -> Vec<VertId> {
        let used: BTreeSet<VertId> = self.faces.values().flat_map(|f| f.verts.iter().copied()).collect();
        let loose: Vec<_> = self.verts.keys().copied().filter(|v| !used.contains(v)).collect();
        for v in &loose {
            self.verts.remove(v);
        }
        self.prune_edge_sets();
        loose
    }

    /// Drop seam/sharp flags on edges that no longer exist.
    pub fn prune_edge_sets(&mut self) {
        let edges = self.edges();
        self.seams.retain(|e| edges.contains(e));
        self.sharp.retain(|e| edges.contains(e));
        self.creases.retain(|e, _| edges.contains(e));
    }

    // ---- access -------------------------------------------------------

    pub fn pos(&self, v: VertId) -> Vec3 {
        self.verts[&v].pos
    }

    pub fn try_pos(&self, v: VertId) -> Result<Vec3> {
        self.verts
            .get(&v)
            .map(|v| v.pos)
            .ok_or_else(|| Error::NotFound { kind: "vertex", id: crate::ids::vert(v) })
    }

    pub fn face(&self, f: FaceId) -> Result<&Face> {
        self.faces.get(&f).ok_or_else(|| Error::NotFound { kind: "face", id: crate::ids::face(f) })
    }

    pub fn face_mut(&mut self, f: FaceId) -> Result<&mut Face> {
        self.faces.get_mut(&f).ok_or_else(|| Error::NotFound { kind: "face", id: crate::ids::face(f) })
    }

    pub fn face_positions(&self, f: &Face) -> Vec<Vec3> {
        f.verts.iter().map(|v| self.pos(*v)).collect()
    }

    /// Newell normal (robust for non-planar / concave polygons).
    pub fn face_normal(&self, f: &Face) -> Vec3 {
        newell_normal(&self.face_positions(f))
    }

    pub fn face_center(&self, f: &Face) -> Vec3 {
        let p = self.face_positions(f);
        p.iter().copied().sum::<Vec3>() / p.len() as f32
    }

    pub fn face_area(&self, f: &Face) -> f32 {
        newell_vector(&self.face_positions(f)).length() * 0.5
    }

    pub fn edges(&self) -> BTreeSet<Edge> {
        self.faces.values().flat_map(|f| f.edges().collect::<Vec<_>>()).collect()
    }

    pub fn edge_faces(&self) -> HashMap<Edge, Vec<FaceId>> {
        let mut m: HashMap<Edge, Vec<FaceId>> = HashMap::new();
        for (&id, f) in &self.faces {
            for e in f.edges() {
                m.entry(e).or_default().push(id);
            }
        }
        m
    }

    pub fn vert_faces(&self) -> HashMap<VertId, Vec<FaceId>> {
        let mut m: HashMap<VertId, Vec<FaceId>> = HashMap::new();
        for (&id, f) in &self.faces {
            for &v in &f.verts {
                m.entry(v).or_default().push(id);
            }
        }
        m
    }

    pub fn vert_edges(&self) -> HashMap<VertId, BTreeSet<Edge>> {
        let mut m: HashMap<VertId, BTreeSet<Edge>> = HashMap::new();
        for f in self.faces.values() {
            for e in f.edges() {
                m.entry(e.0).or_default().insert(e);
                m.entry(e.1).or_default().insert(e);
            }
        }
        m
    }

    /// Edges used by exactly one face.
    pub fn boundary_edges(&self) -> BTreeSet<Edge> {
        self.edge_faces().into_iter().filter(|(_, f)| f.len() == 1).map(|(e, _)| e).collect()
    }

    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.verts.values().map(|v| v.pos);
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
    }

    /// Normal at a vertex: area-weighted average of incident face normals.
    pub fn vert_normal(&self, v: VertId, vert_faces: &HashMap<VertId, Vec<FaceId>>) -> Vec3 {
        let n: Vec3 = vert_faces
            .get(&v)
            .map(|fs| fs.iter().map(|f| newell_vector(&self.face_positions(&self.faces[f]))).sum())
            .unwrap_or(Vec3::ZERO);
        n.normalize_or_zero()
    }

    /// Insert points into the loop of `face` between `a` and `b` (in a->b order).
    /// Works regardless of the direction the face traverses the edge. Corner
    /// attributes are interpolated.
    pub fn insert_on_edge(&mut self, face: FaceId, a: VertId, b: VertId, pts: &[(VertId, f32)]) {
        let f = self.faces.get_mut(&face).unwrap();
        let n = f.verts.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let (x, y) = (f.verts[i], f.verts[j]);
            if (x, y) == (a, b) || (x, y) == (b, a) {
                let forward = (x, y) == (a, b);
                let (ci, cj) = (f.corners[i], f.corners[j]);
                let mut seq: Vec<(VertId, Corner)> = pts
                    .iter()
                    .map(|&(v, t)| {
                        // t is measured from a
                        let tt = if forward { t } else { 1.0 - t };
                        (v, ci.lerp(&cj, tt))
                    })
                    .collect();
                if !forward {
                    seq.reverse();
                }
                for (k, (v, c)) in seq.into_iter().enumerate() {
                    f.verts.insert(i + 1 + k, v);
                    f.corners.insert(i + 1 + k, c);
                }
                return;
            }
        }
    }

    /// Structural warnings for the agent.
    /// Topology and geometry problems, structured so edits can report only
    /// the ones they introduced (see [`Problems::new_since`]).
    pub fn problems(&self) -> Problems {
        // One pass over half-edges: per undirected edge, how many faces use it
        // and how many traverse it in its "forward" direction.
        let mut edges: HashMap<(VertId, VertId), (u32, u32)> = HashMap::with_capacity(self.faces.len() * 2);
        let mut used: HashSet<VertId> = HashSet::with_capacity(self.verts.len());
        let mut degenerate = BTreeSet::new();
        let mut nonplanar = BTreeSet::new();
        let mut pts: Vec<Vec3> = Vec::with_capacity(8);
        for (&id, f) in &self.faces {
            let n = f.verts.len();
            pts.clear();
            for i in 0..n {
                let (a, b) = (f.verts[i], f.verts[(i + 1) % n]);
                let e = edges.entry(if a < b { (a, b) } else { (b, a) }).or_default();
                e.0 += 1;
                if a < b {
                    e.1 += 1;
                }
                used.insert(a);
                pts.push(self.verts.get(&a).map_or(Vec3::ZERO, |v| v.pos));
            }
            let nv = newell_vector(&pts);
            if nv.length_squared() < 1e-16 {
                degenerate.insert(id);
                continue;
            }
            if n > 3 {
                let nrm = nv.normalize();
                let c = pts.iter().copied().sum::<Vec3>() / n as f32;
                let (mut size, mut dev) = (0.0f32, 0.0f32);
                for q in &pts {
                    let d = *q - c;
                    size = size.max(d.length_squared());
                    dev = dev.max(d.dot(nrm).abs());
                }
                if dev > 0.02 * size.sqrt().max(1e-6) {
                    nonplanar.insert(id);
                }
            }
        }
        let nonmanifold = edges.values().filter(|(c, _)| *c > 2).count();
        // Inconsistent winding: two faces traversing an edge in the same direction.
        let flipped = edges.values().map(|(c, fwd)| (*fwd > 1) as usize + ((c - fwd) > 1) as usize).sum();
        Problems { nonmanifold, flipped, degenerate, nonplanar, loose: self.verts.len() - used.len() }
    }

    /// [`problems`](Self::problems) of a shared mesh, cached while that mesh
    /// is alive and unchanged (edits compare the before and after meshes, so
    /// each mesh is checked once instead of twice).
    pub fn problems_cached(mesh: &std::sync::Arc<Mesh>) -> Problems {
        type Entry = (std::sync::Weak<Mesh>, Problems);
        static CACHE: std::sync::Mutex<Vec<Entry>> = std::sync::Mutex::new(Vec::new());
        {
            let cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((_, p)) = cache.iter().find(|(w, _)| w.upgrade().is_some_and(|a| std::sync::Arc::ptr_eq(&a, mesh))) {
                return p.clone();
            }
        }
        let p = mesh.problems();
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        cache.retain(|(w, _)| w.strong_count() > 0);
        cache.push((std::sync::Arc::downgrade(mesh), p.clone()));
        if cache.len() > 32 {
            cache.remove(0);
        }
        p
    }

    /// (edge count, boundary edge count) in one pass.
    pub fn edge_stats(&self) -> (usize, usize) {
        let mut count: HashMap<(VertId, VertId), u32> = HashMap::with_capacity(self.faces.len() * 2);
        for f in self.faces.values() {
            for (a, b) in f.half_edges() {
                *count.entry(if a < b { (a, b) } else { (b, a) }).or_default() += 1;
            }
        }
        (count.len(), count.values().filter(|c| **c == 1).count())
    }

    pub fn validate(&self) -> Vec<String> {
        let p = self.problems();
        p.describe(p.nonmanifold, p.flipped, &p.degenerate, &p.nonplanar, p.loose)
    }

    pub fn transform(&mut self, m: glam::Mat4) {
        for v in self.verts.values_mut() {
            v.pos = m.transform_point3(v.pos);
        }
        if m.determinant() < 0.0 {
            for f in self.faces.values_mut() {
                f.reverse();
            }
        }
    }

    /// Shading normal for every face corner, keyed by (face, corner index).
    /// Flat faces use the face normal; smooth faces average with smooth
    /// neighbours around the vertex, without crossing sharp edges.
    pub fn corner_normals(&self) -> HashMap<(FaceId, usize), Vec3> {
        let ef = self.edge_faces();
        let fnv: HashMap<FaceId, Vec3> =
            self.faces.iter().map(|(id, f)| (*id, newell_vector(&self.face_positions(f)))).collect();
        let mut out = HashMap::new();
        for (&fid, f) in &self.faces {
            for (i, &v) in f.verts.iter().enumerate() {
                if !f.smooth {
                    out.insert((fid, i), fnv[&fid].normalize_or_zero());
                    continue;
                }
                // Flood across non-sharp edges at v between smooth faces.
                let mut group = vec![fid];
                let mut k = 0;
                while k < group.len() {
                    let g = &self.faces[&group[k]];
                    k += 1;
                    let n = g.len();
                    let j = g.index_of(v).unwrap();
                    for other in [g.verts[(j + 1) % n], g.verts[(j + n - 1) % n]] {
                        let e = Edge::new(v, other);
                        if self.sharp.contains(&e) {
                            continue;
                        }
                        for h in ef.get(&e).into_iter().flatten() {
                            if !group.contains(h) && self.faces[h].smooth {
                                group.push(*h);
                            }
                        }
                    }
                }
                let n: Vec3 = group.iter().map(|g| fnv[g]).sum();
                out.insert((fid, i), n.normalize_or_zero());
            }
        }
        out
    }

    pub fn triangle_count(&self) -> usize {
        self.faces.values().map(|f| f.len() - 2).sum()
    }
}

pub fn short_list(items: &[String]) -> String {
    if items.len() <= 12 {
        items.join(", ")
    } else {
        format!("{} … (+{} more)", items[..12].join(", "), items.len() - 12)
    }
}

pub fn newell_vector(p: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for i in 0..p.len() {
        let a = p[i];
        let b = p[(i + 1) % p.len()];
        n += Vec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    n
}

pub fn newell_normal(p: &[Vec3]) -> Vec3 {
    newell_vector(p).normalize_or_zero()
}

/// Problems found by [`Mesh::problems`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Problems {
    pub nonmanifold: usize,
    pub flipped: usize,
    pub degenerate: BTreeSet<FaceId>,
    pub nonplanar: BTreeSet<FaceId>,
    pub loose: usize,
}

impl Problems {
    pub fn is_empty(&self) -> bool {
        *self == Problems::default()
    }

    /// All problems as warning lines.
    pub fn warnings(&self) -> Vec<String> {
        self.describe(self.nonmanifold, self.flipped, &self.degenerate, &self.nonplanar, self.loose)
    }

    fn describe(&self, nonmanifold: usize, flipped: usize, degenerate: &BTreeSet<FaceId>, nonplanar: &BTreeSet<FaceId>, loose: usize) -> Vec<String> {
        let mut w = Vec::new();
        if nonmanifold > 0 {
            w.push(format!("{nonmanifold} non-manifold edge(s) (shared by >2 faces)"));
        }
        if flipped > 0 {
            w.push(format!("{flipped} edge(s) with inconsistent winding (adjacent faces disagree on orientation; try recalc_normals)"));
        }
        let list = |s: &BTreeSet<FaceId>| short_list(&s.iter().map(|f| crate::ids::face(*f)).collect::<Vec<_>>());
        if !degenerate.is_empty() {
            w.push(format!("degenerate (zero-area) faces: {}", list(degenerate)));
        }
        if !nonplanar.is_empty() {
            w.push(format!("non-planar faces (consider triangulate): {}", list(nonplanar)));
        }
        if loose > 0 {
            w.push(format!("{loose} loose vertex/vertices not used by any face"));
        }
        w
    }

    /// Warnings for what got worse compared with `before` (problems the edit introduced).
    pub fn new_since(&self, before: &Problems) -> Vec<String> {
        let more = |a: usize, b: usize| a.saturating_sub(b);
        self.describe(
            more(self.nonmanifold, before.nonmanifold),
            more(self.flipped, before.flipped),
            &self.degenerate.difference(&before.degenerate).copied().collect(),
            &self.nonplanar.difference(&before.nonplanar).copied().collect(),
            more(self.loose, before.loose),
        )
    }
}
