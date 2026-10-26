use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use glam::{DVec2, IVec2, Vec2, Vec3};
use polly_core::Result;
use polly_core::ids;
use polly_core::mesh::{Edge, FaceId, Mesh, short_list};
use serde::{Deserialize, Serialize};

use crate::chart::{Chart, Tri};
use crate::pack::{PackOptions, pack};
use crate::project::view_basis;
use crate::report::UvReport;
use crate::{check_scope, islands};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UnwrapMethod {
    /// Least-squares conformal maps (angle preserving).
    #[default]
    #[serde(alias = "conformal")]
    Lscm,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct UnwrapOptions {
    pub method: UnwrapMethod,
    pub pack: PackOptions,
    /// Lay out charts that form an all-quad grid as axis-aligned rectangles
    /// (column widths / row heights follow edge lengths).
    pub straighten_quads: bool,
}

impl Default for UnwrapOptions {
    fn default() -> Self {
        UnwrapOptions { method: UnwrapMethod::Lscm, pack: PackOptions::default(), straighten_quads: true }
    }
}

/// Unwrap each chart (faces connected across non-seam edges) with LSCM, then pack.
pub fn unwrap(mesh: &mut Mesh, faces: &BTreeSet<FaceId>, o: &UnwrapOptions) -> Result<UvReport> {
    check_scope(mesh, faces)?;
    let mut warnings = Vec::new();
    let mut not_disk = Vec::new();
    for chart_faces in islands(mesh, faces) {
        let chart = Chart::build(mesh, &chart_faces);
        if !chart.is_disk() {
            not_disk.push(ids::face(chart.faces[0]));
        }
        let tris = chart.triangles(mesh);
        let (mut uv, aligned) = match o.straighten_quads.then(|| grid_layout(mesh, &chart)).flatten() {
            Some(uv) => (uv, true),
            None => match lscm(&tris, &chart.wedge_pos) {
                Some((uv, converged)) => {
                    if !converged {
                        warnings.push(format!("LSCM did not fully converge for chart of {}", ids::face(chart.faces[0])));
                    }
                    (uv, false)
                }
                None => (flat_fallback(&chart.wedge_pos), false),
            },
        };
        orient(&mut uv, &tris, aligned);
        for (f, ws) in &chart.corners {
            for (c, w) in mesh.faces.get_mut(f).unwrap().corners.iter_mut().zip(ws) {
                c.uv = Vec2::new(uv[*w].x, -uv[*w].y);
            }
        }
    }
    if !not_disk.is_empty() {
        warnings.push(format!(
            "charts containing {} are not disks (closed, holes or handles), so they fold over; run auto_seams or mark more seams",
            short_list(&not_disk)
        ));
    }
    let mut report = pack(mesh, faces, &o.pack)?;
    warnings.append(&mut report.warnings);
    report.warnings = warnings;
    Ok(report)
}

/// Least-squares conformal map with the two most distant wedges pinned.
/// Returns per-wedge coordinates (math space) and whether the solver converged.
fn lscm(tris: &[Tri], pos: &[Vec3]) -> Option<(Vec<Vec2>, bool)> {
    let far = |from: usize| {
        (0..pos.len()).max_by(|&a, &b| pos[a].distance(pos[from]).total_cmp(&pos[b].distance(pos[from]))).unwrap()
    };
    let p0 = far(0);
    let p1 = far(p0);
    let d = pos[p0].distance(pos[p1]) as f64;
    if d < 1e-9 {
        return None;
    }
    let pinned: HashMap<usize, DVec2> = HashMap::from([(p0, DVec2::ZERO), (p1, DVec2::new(d, 0.0))]);
    let mut var = vec![usize::MAX; pos.len()];
    let mut n = 0;
    for (w, v) in var.iter_mut().enumerate() {
        if !pinned.contains_key(&w) {
            *v = n;
            n += 2;
        }
    }
    let mut normal: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
    let mut rhs = vec![0.0; n];
    let mut used = false;
    for t in tris {
        let [a, b, c] = t.p.map(|p| p.as_dvec3());
        let (ab, ac) = (b - a, c - a);
        let cross = ab.cross(ac);
        let area = cross.length() * 0.5;
        if area < 1e-14 {
            continue;
        }
        used = true;
        let x = ab.normalize();
        let y = (cross.normalize()).cross(x);
        let q = [DVec2::ZERO, DVec2::new(ab.length(), 0.0), DVec2::new(ac.dot(x), ac.dot(y))];
        let k = 0.5 / area.sqrt();
        // Two residual rows of sqrt(A) * (grad v - rot90(grad u)).
        for row in 0..2 {
            let mut free: Vec<(usize, f64)> = Vec::with_capacity(6);
            let mut b = 0.0;
            for j in 0..3 {
                let e = q[(j + 2) % 3] - q[(j + 1) % 3];
                let (cu, cv) = if row == 0 { (e.x, -e.y) } else { (e.y, e.x) };
                let w = t.w[j];
                match pinned.get(&w) {
                    Some(p) => b -= k * (cu * p.x + cv * p.y),
                    None => free.extend([(var[w], k * cu), (var[w] + 1, k * cv)]),
                }
            }
            for &(i, ci) in &free {
                rhs[i] += ci * b;
                for &(j, cj) in &free {
                    *normal[i].entry(j).or_default() += ci * cj;
                }
            }
        }
    }
    if !used {
        return None;
    }
    let (x, converged) = conjugate_gradient(&normal, &rhs);
    let uv = (0..pos.len())
        .map(|w| match pinned.get(&w) {
            Some(p) => p.as_vec2(),
            None => DVec2::new(x[var[w]], x[var[w] + 1]).as_vec2(),
        })
        .collect();
    Some((uv, converged))
}

/// Jacobi-preconditioned CG on a symmetric positive (semi-)definite matrix.
fn conjugate_gradient(a: &[BTreeMap<usize, f64>], b: &[f64]) -> (Vec<f64>, bool) {
    let n = b.len();
    let rows: Vec<Vec<(usize, f64)>> = a.iter().map(|r| r.iter().map(|(&j, &v)| (j, v)).collect()).collect();
    let mean_diag = (0..n).map(|i| a[i].get(&i).copied().unwrap_or(0.0)).sum::<f64>() / n.max(1) as f64;
    let reg = 1e-12 * mean_diag.max(1e-30);
    let diag: Vec<f64> = (0..n).map(|i| a[i].get(&i).copied().unwrap_or(0.0) + reg).collect();
    let mul = |p: &[f64]| -> Vec<f64> {
        rows.iter().enumerate().map(|(i, r)| r.iter().map(|&(j, v)| v * p[j]).sum::<f64>() + reg * p[i]).collect()
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let bnorm = dot(b, b).sqrt();
    if bnorm == 0.0 {
        return (x, true);
    }
    let mut z: Vec<f64> = r.iter().zip(&diag).map(|(r, d)| r / d).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for _ in 0..(4 * n + 200) {
        let ap = mul(&p);
        let alpha = rz / dot(&p, &ap);
        for ((x, r), (p, ap)) in x.iter_mut().zip(&mut r).zip(p.iter().zip(&ap)) {
            *x += alpha * p;
            *r -= alpha * ap;
        }
        if dot(&r, &r).sqrt() < 1e-10 * bnorm {
            return (x, true);
        }
        z = r.iter().zip(&diag).map(|(r, d)| r / d).collect();
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        for (p, z) in p.iter_mut().zip(&z) {
            *p = z + beta * *p;
        }
        rz = rz_next;
    }
    (x, false)
}

fn flat_fallback(pos: &[Vec3]) -> Vec<Vec2> {
    let (r, u) = view_basis(Vec3::Y);
    pos.iter().map(|p| Vec2::new(p.dot(r), p.dot(u))).collect()
}

/// If the chart is an all-quad grid, give each wedge integer grid coordinates
/// and space columns/rows by their mean 3-D edge length.
fn grid_layout(mesh: &Mesh, chart: &Chart) -> Option<Vec<Vec2>> {
    if chart.faces.iter().any(|f| mesh.faces[f].len() != 4) {
        return None;
    }
    let across: HashMap<Edge, (FaceId, FaceId)> = chart.interior.iter().map(|&(e, f, g)| (e, (f, g))).collect();
    let mut grid: Vec<Option<IVec2>> = vec![None; chart.wedge_pos.len()];
    fn assign(grid: &mut [Option<IVec2>], w: usize, p: IVec2) -> bool {
        *grid[w].get_or_insert(p) == p
    }
    let start = chart.faces[0];
    let square = [IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(1, 1), IVec2::new(0, 1)];
    for (w, p) in chart.corners[&start].iter().zip(square) {
        assign(&mut grid, *w, p);
    }
    let mut seen = HashSet::from([start]);
    let mut queue = vec![start];
    while let Some(f) = queue.pop() {
        let face = &mesh.faces[&f];
        for i in 0..4 {
            let (a, b) = (face.verts[i], face.verts[(i + 1) % 4]);
            let Some(&(g0, g1)) = across.get(&Edge::new(a, b)) else { continue };
            let g = if g0 == f { g1 } else { g0 };
            let other = &mesh.faces[&g];
            // A consistently wound neighbor traverses the edge b -> a.
            let j = (0..4).find(|&j| other.verts[j] == b && other.verts[(j + 1) % 4] == a)?;
            let (wa, wb) = (chart.corners[&f][i], chart.corners[&f][(i + 1) % 4]);
            let (ga, gb) = (grid[wa]?, grid[wb]?);
            let d = gb - ga;
            let right = IVec2::new(d.y, -d.x);
            let gw = &chart.corners[&g];
            if !assign(&mut grid, gw[(j + 2) % 4], ga + right) || !assign(&mut grid, gw[(j + 3) % 4], gb + right) {
                return None;
            }
            if seen.insert(g) {
                queue.push(g);
            }
        }
    }
    let grid: Vec<IVec2> = grid.into_iter().collect::<Option<_>>()?;
    if grid.iter().collect::<HashSet<_>>().len() != grid.len() {
        return None;
    }
    // Grid line index -> (summed edge length, count).
    type Lengths = BTreeMap<i32, (f32, u32)>;
    let (mut cols, mut rows) = (Lengths::new(), Lengths::new());
    for (f, ws) in &chart.corners {
        let face = &mesh.faces[f];
        for i in 0..4 {
            let (a, b) = (grid[ws[i]], grid[ws[(i + 1) % 4]]);
            let len = mesh.pos(face.verts[i]).distance(mesh.pos(face.verts[(i + 1) % 4]));
            let slot = if a.y == b.y { cols.entry(a.x.min(b.x)) } else { rows.entry(a.y.min(b.y)) }.or_default();
            slot.0 += len;
            slot.1 += 1;
        }
    }
    let prefix = |m: &Lengths| {
        let mut acc = 0.0;
        let mut out = BTreeMap::new();
        let (&lo, &hi) = (m.keys().next()?, m.keys().last()?);
        for k in lo..=hi + 1 {
            out.insert(k, acc);
            acc += m.get(&k).map_or(0.0, |(s, n)| s / *n as f32);
        }
        Some(out)
    };
    let (xs, ys) = (prefix(&cols)?, prefix(&rows)?);
    Some(grid.iter().map(|g| Vec2::new(xs[&g.x], ys[&g.y])).collect())
}

/// Rotate a chart (math space) to its minimum-area bounding rectangle (unless
/// already axis-aligned), then by a multiple of 90° so world up points up.
fn orient(uv: &mut [Vec2], tris: &[Tri], aligned: bool) {
    if !aligned && let Some(angle) = min_area_angle(uv) {
        let rot = Vec2::from_angle(-angle);
        uv.iter_mut().for_each(|p| *p = rot.rotate(*p));
    }
    let up = uv_gradient(uv, tris, |p| p.y);
    let up = if up.length() > 1e-3 { up } else { uv_gradient(uv, tris, |p| -p.z) };
    if up.length() <= 1e-3 {
        return;
    }
    // Quarter turns needed to bring `up` closest to +y.
    let turns = ((std::f32::consts::FRAC_PI_2 - up.to_angle()) / std::f32::consts::FRAC_PI_2).round() as i32;
    for _ in 0..turns.rem_euclid(4) {
        uv.iter_mut().for_each(|p| *p = Vec2::new(-p.y, p.x));
    }
}

/// Area-weighted direction in uv space along which `f(world pos)` increases,
/// normalized by total area.
fn uv_gradient(uv: &[Vec2], tris: &[Tri], f: impl Fn(Vec3) -> f32) -> Vec2 {
    let (mut sum, mut total) = (Vec2::ZERO, 0.0);
    for t in tris {
        let (a, b) = (uv[t.w[1]] - uv[t.w[0]], uv[t.w[2]] - uv[t.w[0]]);
        let det = a.perp_dot(b);
        if det.abs() < 1e-12 {
            continue;
        }
        let (df1, df2) = (f(t.p[1]) - f(t.p[0]), f(t.p[2]) - f(t.p[0]));
        // gradient * area = sign(det)/2 * adj(M) * df
        sum += 0.5 * det.signum() * Vec2::new(b.y * df1 - a.y * df2, -b.x * df1 + a.x * df2);
        total += det.abs() * 0.5;
    }
    let size = tris.iter().flat_map(|t| t.p).fold(0.0f32, |m, p| m.max(p.length())).max(1e-9);
    if total > 0.0 { sum / total.sqrt() / size } else { Vec2::ZERO }
}

fn min_area_angle(pts: &[Vec2]) -> Option<f32> {
    let hull = convex_hull(pts);
    if hull.len() < 3 {
        return None;
    }
    let area = |angle: f32| {
        let rot = Vec2::from_angle(-angle);
        let (lo, hi) = hull.iter().fold((Vec2::MAX, Vec2::MIN), |(lo, hi), p| {
            let q = rot.rotate(*p);
            (lo.min(q), hi.max(q))
        });
        (hi - lo).x * (hi - lo).y
    };
    (0..hull.len())
        .map(|i| (hull[(i + 1) % hull.len()] - hull[i]).to_angle())
        .min_by(|a, b| area(*a).total_cmp(&area(*b)))
}

fn convex_hull(pts: &[Vec2]) -> Vec<Vec2> {
    let mut p: Vec<Vec2> = pts.to_vec();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    p.dedup_by(|a, b| a.distance(*b) < 1e-9);
    if p.len() < 3 {
        return p;
    }
    let mut hull: Vec<Vec2> = Vec::new();
    for pass in [p.clone(), p.iter().rev().copied().collect()] {
        let base = hull.len();
        for q in pass {
            while hull.len() >= base + 2 && (hull[hull.len() - 1] - hull[hull.len() - 2]).perp_dot(q - hull[hull.len() - 2]) <= 0.0 {
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    hull
}
