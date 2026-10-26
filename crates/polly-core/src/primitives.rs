//! Primitive generators. All are centered on the origin, Y up, faces CCW outward.

use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::error::{Result, invalid};
use crate::mesh::{Mesh, VertId};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Primitive {
    /// Box with the given full extents.
    Cube {
        #[serde(default = "one3")]
        size: [f32; 3],
    },
    /// Flat grid in the XZ plane facing +Y.
    Plane {
        #[serde(default = "one2")]
        size: [f32; 2],
        #[serde(default = "one_u")]
        x_segments: u32,
        #[serde(default = "one_u")]
        z_segments: u32,
    },
    /// Cylinder / truncated cone / cone along Y. A radius of 0 makes an apex.
    Cylinder {
        #[serde(default = "half")]
        radius_bottom: f32,
        #[serde(default = "half")]
        radius_top: f32,
        #[serde(default = "one")]
        height: f32,
        #[serde(default = "eight")]
        segments: u32,
        #[serde(default = "one_u")]
        height_segments: u32,
        #[serde(default = "yes")]
        caps: bool,
    },
    UvSphere {
        #[serde(default = "half")]
        radius: f32,
        #[serde(default = "eight")]
        segments: u32,
        #[serde(default = "six")]
        rings: u32,
    },
    IcoSphere {
        #[serde(default = "half")]
        radius: f32,
        #[serde(default)]
        subdivisions: u32,
    },
    Torus {
        #[serde(default = "half")]
        major_radius: f32,
        #[serde(default = "quarter")]
        minor_radius: f32,
        #[serde(default = "eight")]
        segments: u32,
        #[serde(default = "six")]
        sides: u32,
    },
    /// Revolve a profile of [radius, y] points around the Y axis. The sweep
    /// starts at `start_deg` measured from +X and turns towards +Z (clockwise
    /// seen from above) for `degrees`; less than 360 leaves an opening.
    Lathe {
        profile: Vec<[f32; 2]>,
        #[serde(default = "eight")]
        segments: u32,
        #[serde(default = "full_turn")]
        degrees: f32,
        /// Where the sweep starts: 0 = +X, 90 = +Z, 180 = -X, 270 = -Z.
        #[serde(default)]
        start_deg: f32,
    },
    /// Extrude a 2-D outline of [x, z] points (may be concave) along Y.
    Prism {
        outline: Vec<[f32; 2]>,
        #[serde(default = "one")]
        height: f32,
    },
}

fn one() -> f32 {
    1.0
}
fn half() -> f32 {
    0.5
}
fn quarter() -> f32 {
    0.25
}
fn one3() -> [f32; 3] {
    [1.0; 3]
}
fn one2() -> [f32; 2] {
    [1.0; 2]
}
fn one_u() -> u32 {
    1
}
fn six() -> u32 {
    6
}
fn eight() -> u32 {
    8
}
fn yes() -> bool {
    true
}
fn full_turn() -> f32 {
    360.0
}

impl Primitive {
    pub fn build(&self) -> Result<Mesh> {
        match self {
            Primitive::Cube { size } => Ok(cube(Vec3::from_array(*size))),
            Primitive::Plane { size, x_segments, z_segments } => {
                Ok(plane(Vec2::from_array(*size), (*x_segments).max(1), (*z_segments).max(1)))
            }
            Primitive::Cylinder { radius_bottom, radius_top, height, segments, height_segments, caps } => {
                if *segments < 3 {
                    return invalid("segments must be >= 3");
                }
                let rows: Vec<(f32, f32)> = (0..=(*height_segments).max(1))
                    .map(|i| {
                        let t = i as f32 / (*height_segments).max(1) as f32;
                        (radius_bottom + (radius_top - radius_bottom) * t, -height / 2.0 + height * t)
                    })
                    .collect();
                Ok(revolve(&rows, *segments, 0.0, TAU, *caps))
            }
            Primitive::UvSphere { radius, segments, rings } => {
                if *segments < 3 || *rings < 2 {
                    return invalid("need segments >= 3 and rings >= 2");
                }
                let rows: Vec<(f32, f32)> = (0..=*rings)
                    .map(|i| {
                        let phi = PI - PI * i as f32 / *rings as f32; // bottom to top
                        let (s, c) = phi.sin_cos();
                        (if i == 0 || i == *rings { 0.0 } else { radius * s }, radius * c)
                    })
                    .collect();
                Ok(revolve(&rows, *segments, 0.0, TAU, false))
            }
            Primitive::IcoSphere { radius, subdivisions } => {
                if *subdivisions > 4 {
                    return invalid("subdivisions must be <= 4 for low-poly work");
                }
                Ok(ico_sphere(*radius, *subdivisions))
            }
            Primitive::Torus { major_radius, minor_radius, segments, sides } => {
                if *segments < 3 || *sides < 3 {
                    return invalid("segments and sides must be >= 3");
                }
                Ok(torus(*major_radius, *minor_radius, *segments, *sides))
            }
            Primitive::Lathe { profile, segments, degrees, start_deg } => {
                if profile.len() < 2 || *segments < 2 {
                    return invalid("lathe needs >= 2 profile points and >= 2 segments");
                }
                let rows: Vec<(f32, f32)> = profile.iter().map(|p| (p[0].max(0.0), p[1])).collect();
                let angle = degrees.clamp(1.0, 360.0).to_radians();
                Ok(revolve(&rows, *segments, start_deg.to_radians(), angle, false))
            }
            Primitive::Prism { outline, height } => {
                if outline.len() < 3 {
                    return invalid("prism outline needs >= 3 points");
                }
                prism(outline, *height)
            }
        }
    }
}

pub fn cube(size: Vec3) -> Mesh {
    let h = size / 2.0;
    let mut m = Mesh::new();
    let v: Vec<VertId> = [
        (-1., -1., -1.),
        (1., -1., -1.),
        (1., 1., -1.),
        (-1., 1., -1.),
        (-1., -1., 1.),
        (1., -1., 1.),
        (1., 1., 1.),
        (-1., 1., 1.),
    ]
    .iter()
    .map(|&(x, y, z)| m.add_vert(Vec3::new(x, y, z) * h))
    .collect();
    for f in [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]] {
        m.add_face(f.iter().map(|&i| v[i]).collect()).unwrap();
    }
    m
}

pub fn plane(size: Vec2, nx: u32, nz: u32) -> Mesh {
    let mut m = Mesh::new();
    let mut grid = vec![vec![0; (nz + 1) as usize]; (nx + 1) as usize];
    for i in 0..=nx {
        for j in 0..=nz {
            let x = -size.x / 2.0 + size.x * i as f32 / nx as f32;
            let z = -size.y / 2.0 + size.y * j as f32 / nz as f32;
            grid[i as usize][j as usize] = m.add_vert(Vec3::new(x, 0.0, z));
        }
    }
    for i in 0..nx as usize {
        for j in 0..nz as usize {
            m.add_face(vec![grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]]).unwrap();
        }
    }
    m
}

/// Revolve rows of (radius, y) around Y, from angle `start` (radians, 0 = +X,
/// turning towards +Z) through `angle`. Rows with radius 0 become poles.
/// If `angle < TAU` the result is open (no wrap-around).
fn revolve(rows: &[(f32, f32)], segments: u32, start: f32, angle: f32, caps: bool) -> Mesh {
    let mut m = Mesh::new();
    let full = (angle - TAU).abs() < 1e-4;
    let cols = if full { segments } else { segments + 1 };
    let ring: Vec<Vec<VertId>> = rows
        .iter()
        .map(|&(r, y)| {
            if r <= 1e-6 {
                let p = m.add_vert(Vec3::new(0.0, y, 0.0));
                vec![p; cols as usize]
            } else {
                (0..cols)
                    .map(|k| {
                        let t = start + angle * k as f32 / segments as f32;
                        m.add_vert(Vec3::new(r * t.cos(), y, r * t.sin()))
                    })
                    .collect()
            }
        })
        .collect();
    let col = |k: u32| (k % cols) as usize;
    for i in 0..rows.len() - 1 {
        let (b, t) = (&ring[i], &ring[i + 1]);
        for k in 0..segments {
            let mut f = vec![b[col(k)], t[col(k)], t[col(k + 1)], b[col(k + 1)]];
            f.dedup();
            if f.first() == f.last() && f.len() > 1 {
                f.pop();
            }
            if f.len() >= 3 {
                let _ = m.add_face(f);
            }
        }
    }
    if caps && full {
        let bottom = &ring[0];
        if bottom[0] != bottom[1] {
            m.add_face(bottom.clone()).unwrap();
        }
        let top = ring.last().unwrap();
        if top[0] != top[1] {
            m.add_face(top.iter().rev().copied().collect()).unwrap();
        }
    }
    orient_outward(&mut m);
    m
}

fn ico_sphere(radius: f32, subdivisions: u32) -> Mesh {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut pts: Vec<Vec3> = [
        (-1., t, 0.),
        (1., t, 0.),
        (-1., -t, 0.),
        (1., -t, 0.),
        (0., -1., t),
        (0., 1., t),
        (0., -1., -t),
        (0., 1., -t),
        (t, 0., -1.),
        (t, 0., 1.),
        (-t, 0., -1.),
        (-t, 0., 1.),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let mut tris: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut cache = std::collections::HashMap::new();
        let mut mid = |a: usize, b: usize, pts: &mut Vec<Vec3>| {
            *cache.entry((a.min(b), a.max(b))).or_insert_with(|| {
                pts.push(((pts[a] + pts[b]) / 2.0).normalize());
                pts.len() - 1
            })
        };
        let mut next = Vec::new();
        for [a, b, c] in tris {
            let ab = mid(a, b, &mut pts);
            let bc = mid(b, c, &mut pts);
            let ca = mid(c, a, &mut pts);
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }
    let mut m = Mesh::new();
    let ids: Vec<VertId> = pts.iter().map(|p| m.add_vert(*p * radius)).collect();
    for [a, b, c] in tris {
        let mut f = vec![ids[a], ids[b], ids[c]];
        let n = (pts[b] - pts[a]).cross(pts[c] - pts[a]);
        if n.dot(pts[a]) < 0.0 {
            f.reverse();
        }
        m.add_face(f).unwrap();
    }
    m
}

fn torus(big_r: f32, r: f32, segments: u32, sides: u32) -> Mesh {
    let mut m = Mesh::new();
    let mut g = vec![vec![0; sides as usize]; segments as usize];
    for i in 0..segments {
        let u = TAU * i as f32 / segments as f32;
        for j in 0..sides {
            let v = TAU * j as f32 / sides as f32;
            let rr = big_r + r * v.cos();
            g[i as usize][j as usize] = m.add_vert(Vec3::new(rr * u.cos(), r * v.sin(), rr * u.sin()));
        }
    }
    for i in 0..segments as usize {
        let i2 = (i + 1) % segments as usize;
        for j in 0..sides as usize {
            let j2 = (j + 1) % sides as usize;
            let mut f = vec![g[i][j], g[i][j2], g[i2][j2], g[i2][j]];
            // Orient outward from the tube center.
            let u = TAU * i as f32 / segments as f32;
            let tube = Vec3::new(big_r * u.cos(), 0.0, big_r * u.sin());
            let p: Vec<Vec3> = f.iter().map(|v| m.pos(*v)).collect();
            let c = p.iter().copied().sum::<Vec3>() / 4.0;
            if crate::mesh::newell_vector(&p).dot(c - tube) < 0.0 {
                f.reverse();
            }
            m.add_face(f).unwrap();
        }
    }
    m
}

fn prism(outline: &[[f32; 2]], height: f32) -> Result<Mesh> {
    let mut pts: Vec<Vec2> = outline.iter().map(|p| Vec2::new(p[0], p[1])).collect();
    // Make the outline CCW when viewed from +Y (x -> -z ... ). In XZ seen from
    // above (+Y), CCW means the 2-D signed area in (x, -z) is positive.
    let area: f32 = (0..pts.len()).map(|i| pts[i].perp_dot(pts[(i + 1) % pts.len()])).sum();
    if area > 0.0 {
        // (x, z) CCW == clockwise seen from +Y
        pts.reverse();
    }
    let mut m = Mesh::new();
    let h = height / 2.0;
    let bot: Vec<VertId> = pts.iter().map(|p| m.add_vert(Vec3::new(p.x, -h, p.y))).collect();
    let top: Vec<VertId> = pts.iter().map(|p| m.add_vert(Vec3::new(p.x, h, p.y))).collect();
    let n = pts.len();
    m.add_face(top.clone())?;
    m.add_face(bot.iter().rev().copied().collect())?;
    for i in 0..n {
        let j = (i + 1) % n;
        m.add_face(vec![bot[i], bot[j], top[j], top[i]])?;
    }
    Ok(m)
}

/// Heuristic: flip all faces if they mostly point toward the centroid.
pub fn orient_outward(m: &mut Mesh) {
    let Some((lo, hi)) = m.bounds() else { return };
    let centroid = (lo + hi) / 2.0;
    let score: f32 = m
        .faces
        .values()
        .map(|f| {
            let p = m.face_positions(f);
            let c = p.iter().copied().sum::<Vec3>() / p.len() as f32;
            crate::mesh::newell_vector(&p).dot(c - centroid)
        })
        .sum();
    if score < 0.0 {
        for f in m.faces.values_mut() {
            f.reverse();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed_and_consistent(m: &Mesh) {
        let w = m.validate();
        assert!(w.is_empty(), "{w:?}");
        assert!(m.boundary_edges().is_empty());
    }

    #[test]
    fn lathe_sweep_direction_and_start() {
        let half = |start: f32| {
            Primitive::Lathe { profile: vec![[1.0, 0.0], [1.0, 1.0]], segments: 4, degrees: 180.0, start_deg: start }.build().unwrap()
        };
        // 0..180 starting at +X sweeps through +Z: every vertex has z >= 0.
        assert!(half(0.0).verts.values().all(|v| v.pos.z > -1e-5));
        assert!(half(0.0).verts.values().any(|v| v.pos.z > 0.99));
        // Starting at 180 (-X) the half goes through -Z instead.
        assert!(half(180.0).verts.values().all(|v| v.pos.z < 1e-5));
    }

    #[test]
    fn primitives_are_clean() {
        let c = cube(Vec3::ONE);
        closed_and_consistent(&c);
        assert_eq!((c.verts.len(), c.edges().len(), c.faces.len()), (8, 12, 6));
        for f in c.faces.values() {
            let n = c.face_normal(f);
            assert!(n.dot(c.face_center(f)) > 0.0);
        }
        let cyl = Primitive::Cylinder {
            radius_bottom: 0.5,
            radius_top: 0.5,
            height: 1.0,
            segments: 6,
            height_segments: 1,
            caps: true,
        }
        .build()
        .unwrap();
        closed_and_consistent(&cyl);
        let cone = Primitive::Cylinder {
            radius_bottom: 0.5,
            radius_top: 0.0,
            height: 1.0,
            segments: 5,
            height_segments: 1,
            caps: true,
        }
        .build()
        .unwrap();
        closed_and_consistent(&cone);
        assert_eq!(cone.faces.len(), 6);
        closed_and_consistent(&Primitive::UvSphere { radius: 1.0, segments: 8, rings: 5 }.build().unwrap());
        closed_and_consistent(&ico_sphere(1.0, 1));
        closed_and_consistent(&torus(1.0, 0.3, 8, 5));
        let p = prism(&[[0., 0.], [2., 0.], [2., 1.], [1., 1.], [1., 2.], [0., 2.]], 1.0).unwrap();
        closed_and_consistent(&p);
        for f in p.faces.values() {
            let n = p.face_normal(f);
            assert!(n.dot(p.face_center(f) - Vec3::new(0.8, 0.0, 0.8)) > 0.0);
        }
    }
}
