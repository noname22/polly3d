use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

use polly_core::mesh::{Edge, FaceId, Mesh, VertId};

use crate::chart::{Chart, Dsu, components, dihedral_deg, scope_edges};

/// Mark (`on = true`) or clear UV seams.
pub fn mark_seams(mesh: &mut Mesh, edges: &BTreeSet<Edge>, on: bool) {
    for e in edges {
        if on {
            mesh.seams.insert(*e);
        } else {
            mesh.seams.remove(e);
        }
    }
}

/// Charts: faces connected across non-seam edges (an edge shared by more than
/// two of the faces never connects).
pub fn islands(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Vec<BTreeSet<FaceId>> {
    components(mesh, faces, |e, _, _| !mesh.seams.contains(&e))
}

/// UV islands: faces connected across non-seam edges whose corner uvs agree on
/// both sides. Unlike [`islands`] this follows the current layout, so e.g. a
/// box projection yields one island per projected piece.
pub fn uv_islands(mesh: &Mesh, faces: &BTreeSet<FaceId>) -> Vec<BTreeSet<FaceId>> {
    let uv = |f: FaceId, v: VertId| {
        let face = &mesh.faces[&f];
        face.corners[face.index_of(v).unwrap()].uv
    };
    components(mesh, faces, |e, f, g| {
        !mesh.seams.contains(&e)
            && uv(f, e.0).distance(uv(g, e.0)) < 1e-5
            && uv(f, e.1).distance(uv(g, e.1)) < 1e-5
    })
}

/// Mark seams on edges sharper than `angle_deg` (and non-manifold edges), then
/// add the cuts needed so every chart is a topological disk: closed charts are
/// split in two, and charts with holes or handles get a cut graph. Returns the
/// number of newly marked seams.
pub fn auto_seams(mesh: &mut Mesh, faces: &BTreeSet<FaceId>, angle_deg: f32) -> usize {
    let mut added = 0;
    let mut mark = |mesh: &mut Mesh, e: Edge| added += mesh.seams.insert(e) as usize;
    for (e, fs) in scope_edges(mesh, faces.iter().copied()) {
        match fs[..] {
            [f, g] if dihedral_deg(mesh, f, g) > angle_deg => mark(mesh, e),
            [_, _, _, ..] => mark(mesh, e),
            _ => {}
        }
    }
    for _ in 0..8 {
        let mut cuts = Vec::new();
        for chart in islands(mesh, faces) {
            let chart = Chart::build(mesh, &chart);
            if chart.boundary.is_empty() {
                cuts.extend(split_closed(mesh, &chart));
            } else if !chart.is_disk() {
                cuts.extend(cut_graph(mesh, &chart));
            }
        }
        if cuts.is_empty() {
            break;
        }
        for e in cuts {
            mark(mesh, e);
        }
    }
    added
}

/// Grow two regions from the two most distant faces (Dijkstra over face
/// centers) and cut between them.
fn split_closed(mesh: &Mesh, chart: &Chart) -> Vec<Edge> {
    let center: BTreeMap<FaceId, glam::Vec3> =
        chart.faces.iter().map(|&f| (f, mesh.face_center(&mesh.faces[&f]))).collect();
    let farthest = |from: FaceId| {
        *chart.faces.iter().max_by(|a, b| {
            center[a].distance(center[&from]).total_cmp(&center[b].distance(center[&from]))
        })
        .unwrap()
    };
    let a = farthest(chart.faces[0]);
    let b = farthest(a);
    if a == b {
        return vec![];
    }
    let mut adj: HashMap<FaceId, Vec<FaceId>> = HashMap::new();
    for &(_, f, g) in &chart.interior {
        adj.entry(f).or_default().push(g);
        adj.entry(g).or_default().push(f);
    }
    let dist = |seed: FaceId| {
        let mut d: HashMap<FaceId, u64> = HashMap::new();
        let mut heap = BinaryHeap::from([(Reverse(0u64), seed)]);
        while let Some((Reverse(du), f)) = heap.pop() {
            if d.contains_key(&f) {
                continue;
            }
            d.insert(f, du);
            for &g in adj.get(&f).into_iter().flatten() {
                if !d.contains_key(&g) {
                    // Scaled to integers so ties are exact.
                    heap.push((Reverse(du + (center[&f].distance(center[&g]) * 1e4) as u64), g));
                }
            }
        }
        d
    };
    let (da, db) = (dist(a), dist(b));
    // Nearest seed wins; ties go to the smaller side.
    let mut side: HashMap<FaceId, bool> = HashMap::new();
    let mut count = [0usize; 2];
    let mut ties = Vec::new();
    for &f in &chart.faces {
        match da.get(&f).cmp(&db.get(&f)) {
            std::cmp::Ordering::Less => side.insert(f, false),
            std::cmp::Ordering::Greater => side.insert(f, true),
            std::cmp::Ordering::Equal => {
                ties.push(f);
                continue;
            }
        };
        count[side[&f] as usize] += 1;
    }
    for f in ties {
        let s = count[1] < count[0];
        side.insert(f, s);
        count[s as usize] += 1;
    }
    chart.interior.iter().filter(|(_, f, g)| side[f] != side[g]).map(|(e, _, _)| *e).collect()
}

/// Classic cut graph: edges not crossed by a dual spanning tree, with dangling
/// branches pruned, connect all boundaries and cut all handles. The dual tree
/// prefers crossing flat, long edges so cuts land on sharp, short ones.
fn cut_graph(mesh: &Mesh, chart: &Chart) -> Vec<Edge> {
    let index: BTreeMap<FaceId, usize> = chart.faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let mut order: Vec<_> = chart
        .interior
        .iter()
        .map(|&(e, f, g)| (dihedral_deg(mesh, f, g), -mesh.pos(e.0).distance(mesh.pos(e.1)), e, f, g))
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut dsu = Dsu::new(chart.faces.len());
    let mut free: BTreeSet<Edge> = BTreeSet::new();
    for (_, _, e, f, g) in order {
        if !dsu.union(index[&f], index[&g]) {
            free.insert(e);
        }
    }
    let protected: BTreeSet<Edge> = chart.boundary.iter().map(|(e, _)| *e).collect();
    let mut degree: HashMap<VertId, usize> = HashMap::new();
    for e in free.iter().chain(&protected) {
        *degree.entry(e.0).or_default() += 1;
        *degree.entry(e.1).or_default() += 1;
    }
    loop {
        let leaves: Vec<Edge> = free.iter().copied().filter(|e| degree[&e.0] == 1 || degree[&e.1] == 1).collect();
        if leaves.is_empty() {
            break;
        }
        for e in leaves {
            if free.remove(&e) {
                *degree.get_mut(&e.0).unwrap() -= 1;
                *degree.get_mut(&e.1).unwrap() -= 1;
            }
        }
    }
    free.into_iter().collect()
}

/// Edges of the shortest path (by length) from `a` to `b` along mesh edges.
pub fn shortest_path(mesh: &Mesh, a: VertId, b: VertId) -> Option<Vec<Edge>> {
    let mut adj: HashMap<VertId, Vec<(VertId, f32)>> = HashMap::new();
    for e in mesh.edges() {
        let len = mesh.verts[&e.0].pos.distance(mesh.verts[&e.1].pos);
        adj.entry(e.0).or_default().push((e.1, len));
        adj.entry(e.1).or_default().push((e.0, len));
    }
    let mut dist: HashMap<VertId, f32> = HashMap::from([(a, 0.0)]);
    let mut prev: HashMap<VertId, VertId> = HashMap::new();
    // Non-negative f32 bits sort like the values.
    let mut heap = BinaryHeap::from([Reverse((0u32, a))]);
    while let Some(Reverse((d, v))) = heap.pop() {
        let d = f32::from_bits(d);
        if v == b {
            break;
        }
        if d > dist[&v] {
            continue;
        }
        for &(n, len) in adj.get(&v).into_iter().flatten() {
            let nd = d + len;
            if dist.get(&n).is_none_or(|old| nd < *old) {
                dist.insert(n, nd);
                prev.insert(n, v);
                heap.push(Reverse((nd.to_bits(), n)));
            }
        }
    }
    if a != b && !prev.contains_key(&b) {
        return None;
    }
    let mut path = Vec::new();
    let mut v = b;
    while v != a {
        let p = prev[&v];
        path.push(Edge::new(p, v));
        v = p;
    }
    path.reverse();
    Some(path)
}

/// Edges where `faces` cross a plane: those between a face whose centre is in
/// front of the plane and one behind it. With `within` (point, normal), only
/// edges whose midpoint is on its positive side, e.g. the back half for a
/// centre-back seam.
pub fn plane_seams(
    mesh: &Mesh,
    faces: &BTreeSet<FaceId>,
    point: polly_core::glam::Vec3,
    normal: polly_core::glam::Vec3,
    within: Option<(polly_core::glam::Vec3, polly_core::glam::Vec3)>,
) -> BTreeSet<Edge> {
    let n = normal.normalize_or_zero();
    let front = |f: &FaceId| (mesh.face_center(&mesh.faces[f]) - point).dot(n) > 0.0;
    let mut out = BTreeSet::new();
    for (e, fs) in mesh.edge_faces() {
        let fs: Vec<_> = fs.into_iter().filter(|f| faces.contains(f)).collect();
        if fs.len() < 2 || fs.iter().all(front) || !fs.iter().any(front) {
            continue;
        }
        let mid = (mesh.verts[&e.0].pos + mesh.verts[&e.1].pos) * 0.5;
        if within.is_none_or(|(p, wn)| (mid - p).dot(wn) >= 0.0) {
            out.insert(e);
        }
    }
    out
}
