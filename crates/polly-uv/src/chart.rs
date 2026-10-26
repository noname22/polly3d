//! Chart topology: faces connected across non-seam edges, with per-corner
//! "wedges" (a vertex split wherever a seam passes through it).

use std::collections::{BTreeMap, BTreeSet};

use glam::Vec3;
use polly_core::geom::triangulate_polygon;
use polly_core::mesh::{Edge, FaceId, Mesh};

pub(crate) struct Dsu(Vec<usize>);

impl Dsu {
    pub fn new(n: usize) -> Self {
        Dsu((0..n).collect())
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    /// Returns true if `a` and `b` were in different sets.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let (a, b) = (self.find(a), self.find(b));
        self.0[a] = b;
        a != b
    }
}

/// Edge -> faces, restricted to `faces`.
pub(crate) fn scope_edges(mesh: &Mesh, faces: impl IntoIterator<Item = FaceId>) -> BTreeMap<Edge, Vec<FaceId>> {
    let mut m: BTreeMap<Edge, Vec<FaceId>> = BTreeMap::new();
    for f in faces {
        if let Some(face) = mesh.faces.get(&f) {
            for e in face.edges() {
                m.entry(e).or_default().push(f);
            }
        }
    }
    m
}

/// Group `faces` into connected components, joining two faces across an
/// edge when `joins(edge, f, g)` holds. Components are sorted by first id.
pub(crate) fn components(
    mesh: &Mesh,
    faces: &BTreeSet<FaceId>,
    mut joins: impl FnMut(Edge, FaceId, FaceId) -> bool,
) -> Vec<BTreeSet<FaceId>> {
    let ids: Vec<FaceId> = faces.iter().copied().filter(|f| mesh.faces.contains_key(f)).collect();
    let index: BTreeMap<FaceId, usize> = ids.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let mut dsu = Dsu::new(ids.len());
    for (e, fs) in scope_edges(mesh, ids.iter().copied()) {
        if let [f, g] = fs[..]
            && joins(e, f, g)
        {
            dsu.union(index[&f], index[&g]);
        }
    }
    let mut groups: BTreeMap<usize, BTreeSet<FaceId>> = BTreeMap::new();
    for (i, &f) in ids.iter().enumerate() {
        groups.entry(dsu.find(i)).or_default().insert(f);
    }
    let mut out: Vec<_> = groups.into_values().collect();
    out.sort_by_key(|g| *g.first().unwrap());
    out
}

pub(crate) struct Tri {
    pub w: [usize; 3],
    pub p: [Vec3; 3],
}

pub(crate) struct Chart {
    pub faces: Vec<FaceId>,
    /// Wedge index of every corner, per face.
    pub corners: BTreeMap<FaceId, Vec<usize>>,
    pub wedge_pos: Vec<Vec3>,
    /// Non-seam edges shared by exactly two chart faces.
    pub interior: Vec<(Edge, FaceId, FaceId)>,
    /// Boundary occurrences (an internal seam counts once per side).
    pub boundary: Vec<(Edge, FaceId)>,
}

impl Chart {
    pub fn build(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Chart {
        let faces: Vec<FaceId> = faces.iter().copied().collect();
        let mut offset = BTreeMap::new();
        let mut n = 0;
        for &f in &faces {
            offset.insert(f, n);
            n += mesh.faces[&f].len();
        }
        let corner = |f: FaceId, v| offset[&f] + mesh.faces[&f].index_of(v).unwrap();
        let mut dsu = Dsu::new(n);
        let (mut interior, mut boundary) = (Vec::new(), Vec::new());
        for (e, fs) in scope_edges(mesh, faces.iter().copied()) {
            match fs[..] {
                [f, g] if !mesh.seams.contains(&e) => {
                    dsu.union(corner(f, e.0), corner(g, e.0));
                    dsu.union(corner(f, e.1), corner(g, e.1));
                    interior.push((e, f, g));
                }
                _ => boundary.extend(fs.iter().map(|&f| (e, f))),
            }
        }
        let mut wedge_of = BTreeMap::new();
        let mut wedge_pos = Vec::new();
        let mut corners = BTreeMap::new();
        for &f in &faces {
            let face = &mesh.faces[&f];
            let ws = (0..face.len())
                .map(|i| {
                    let root = dsu.find(offset[&f] + i);
                    *wedge_of.entry(root).or_insert_with(|| {
                        wedge_pos.push(mesh.pos(face.verts[i]));
                        wedge_pos.len() - 1
                    })
                })
                .collect();
            corners.insert(f, ws);
        }
        Chart { faces, corners, wedge_pos, interior, boundary }
    }

    pub fn euler(&self) -> i64 {
        self.wedge_pos.len() as i64 - (self.interior.len() + self.boundary.len()) as i64 + self.faces.len() as i64
    }

    pub fn is_disk(&self) -> bool {
        self.euler() == 1
    }

    pub fn triangles(&self, mesh: &Mesh) -> Vec<Tri> {
        let mut out = Vec::new();
        for f in &self.faces {
            let p = mesh.face_positions(&mesh.faces[f]);
            let ws = &self.corners[f];
            for [a, b, c] in triangulate_polygon(&p) {
                out.push(Tri { w: [ws[a], ws[b], ws[c]], p: [p[a], p[b], p[c]] });
            }
        }
        out
    }
}

/// Angle between the normals of two faces, in degrees.
pub(crate) fn dihedral_deg(mesh: &Mesh, f: FaceId, g: FaceId) -> f32 {
    let (a, b) = (mesh.face_normal(&mesh.faces[&f]), mesh.face_normal(&mesh.faces[&g]));
    a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
}
