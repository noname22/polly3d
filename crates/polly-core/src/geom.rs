//! Small geometry helpers: polygon triangulation and plane projection.

use glam::{Vec2, Vec3};

use crate::mesh::newell_normal;

/// Orthonormal basis (u, v) spanning the plane with normal `n`.
pub fn plane_basis(n: Vec3) -> (Vec3, Vec3) {
    let n = n.normalize_or(Vec3::Y);
    let helper = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = helper.cross(n).normalize();
    let v = n.cross(u);
    (u, v)
}

pub fn project_to_plane(points: &[Vec3], n: Vec3) -> Vec<Vec2> {
    let (u, v) = plane_basis(n);
    points.iter().map(|p| Vec2::new(p.dot(u), p.dot(v))).collect()
}

fn signed_area(p: &[Vec2]) -> f32 {
    let mut a = 0.0;
    for i in 0..p.len() {
        let (x, y) = (p[i], p[(i + 1) % p.len()]);
        a += x.perp_dot(y);
    }
    a * 0.5
}

fn point_in_tri(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = (b - a).perp_dot(p - a);
    let d2 = (c - b).perp_dot(p - b);
    let d3 = (a - c).perp_dot(p - c);
    d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0
}

/// Ear-clipping triangulation of a simple polygon in 2-D. Returns index
/// triples preserving the input winding. Falls back to a fan on failure.
pub fn triangulate_2d(p: &[Vec2]) -> Vec<[usize; 3]> {
    let n = p.len();
    if n < 3 {
        return vec![];
    }
    if n == 3 {
        return vec![[0, 1, 2]];
    }
    let ccw = signed_area(p) >= 0.0;
    // Work in CCW space.
    let q: Vec<Vec2> = if ccw { p.to_vec() } else { p.iter().map(|v| Vec2::new(v.x, -v.y)).collect() };
    let mut idx: Vec<usize> = (0..n).collect();
    let mut tris = Vec::with_capacity(n - 2);
    let mut guard = 0;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let (ia, ib, ic) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (a, b, c) = (q[ia], q[ib], q[ic]);
            if (b - a).perp_dot(c - b) <= 1e-12 {
                continue; // reflex or degenerate
            }
            let blocked = idx.iter().any(|&j| j != ia && j != ib && j != ic && point_in_tri(q[j], a, b, c));
            if blocked {
                continue;
            }
            tris.push([ia, ib, ic]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    } else if idx.len() > 3 {
        for k in 1..idx.len() - 1 {
            tris.push([idx[0], idx[k], idx[k + 1]]);
        }
    }
    tris
}

/// Triangulate a 3-D polygon (indices into `points`), preserving winding.
pub fn triangulate_polygon(points: &[Vec3]) -> Vec<[usize; 3]> {
    if points.len() == 3 {
        return vec![[0, 1, 2]];
    }
    let n = newell_normal(points);
    let mut p2 = project_to_plane(points, n);
    // plane_basis gives a right-handed (u, v, n) frame, so CCW around n stays CCW in 2-D.
    if signed_area(&p2) < 0.0 {
        // Non-simple / self-overlapping; still triangulate.
        p2.iter_mut().for_each(|v| v.y = -v.y);
        return triangulate_2d(&p2);
    }
    triangulate_2d(&p2)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concave_l_shape() {
        let p = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let t = triangulate_2d(&p);
        assert_eq!(t.len(), 4);
        let area: f32 = t.iter().map(|[a, b, c]| (p[*b] - p[*a]).perp_dot(p[*c] - p[*a]) * 0.5).sum();
        assert!((area - 3.0).abs() < 1e-5);
    }
}

/// Extent along `u` of where triangles cross the plane `p·v = at`: the
/// outline of a mesh at that height, as seen along `u × v` (orthographic).
pub fn slice_extent(tris: &[[Vec3; 3]], u: Vec3, v: Vec3, at: f32) -> Option<(f32, f32)> {
    let mut range: Option<(f32, f32)> = None;
    let mut add = |p: Vec3| {
        let x = p.dot(u);
        range = Some(range.map_or((x, x), |(a, b)| (a.min(x), b.max(x))));
    };
    for t in tris {
        let h = t.map(|p| p.dot(v) - at);
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let (a, b) = (h[i], h[j]);
            if a == 0.0 {
                add(t[i]);
            }
            if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
                add(t[i].lerp(t[j], a / (a - b)));
            }
        }
    }
    range
}

#[cfg(test)]
mod slice_tests {
    use super::*;

    #[test]
    fn slice_extent_of_a_tilted_triangle() {
        let t = [[Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 0.0)]];
        let (a, b) = slice_extent(&t, Vec3::X, Vec3::Y, 1.0).unwrap();
        assert!((a - 0.5).abs() < 1e-6 && (b - 1.5).abs() < 1e-6);
        assert!(slice_extent(&t, Vec3::X, Vec3::Y, 3.0).is_none());
    }
}
