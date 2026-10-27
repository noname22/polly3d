//! Region brushes for organic detail: smooth, inflate, crease and displace
//! vertices near a stroke (a path of points), fading out with distance.

use std::collections::{BTreeMap, HashMap};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::error::{Result, invalid};
use crate::mesh::{Mesh, VertId};
use crate::ops::transform::Falloff;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Brush {
    /// Relax towards the average of neighbours (removes lumps and pinches).
    Smooth,
    /// Push out along the normals (negative strength pulls in).
    Inflate,
    /// Pull towards the stroke and sink along the normal: a groove (negative strength = a ridge).
    Crease,
    /// Move along the normals by 3-D noise: skin pores, bark, rough stone.
    Displace {
        /// Noise feature size in object units (default: radius / 4).
        #[serde(default)]
        scale: Option<f32>,
        #[serde(default)]
        seed: u64,
    },
    /// Move towards the average plane of the affected area.
    Flatten,
}

/// Stroke settings.
#[derive(Debug, Clone)]
pub struct Stroke {
    /// Path in object space (one point = a dab).
    pub path: Vec<Vec3>,
    pub radius: f32,
    /// 0..1 typical (fraction of the radius for displacement-like brushes).
    pub strength: f32,
    pub falloff: Falloff,
    /// Smoothing passes (smooth / crease).
    pub iterations: u32,
}

fn closest_on_path(path: &[Vec3], p: Vec3) -> (Vec3, f32) {
    if path.len() == 1 {
        return (path[0], p.distance(path[0]));
    }
    let mut best = (path[0], f32::MAX);
    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
        let q = a + ab * t;
        let d = p.distance(q);
        if d < best.1 {
            best = (q, d);
        }
    }
    best
}

fn falloff(f: Falloff, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match f {
        Falloff::Smooth => t * t * (3.0 - 2.0 * t),
        Falloff::Linear => t,
        Falloff::Sharp => t * t,
        Falloff::Root => t.sqrt(),
    }
}

fn hash3(seed: u64, x: i32, y: i32, z: i32) -> f32 {
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    for v in [x, y, z] {
        h ^= v as u32 as u64;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(31);
    }
    h ^= h >> 29;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

fn noise3(seed: u64, p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let s = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let (x, y, z) = (i.x as i32, i.y as i32, i.z as i32);
    let mut acc = 0.0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let w = (if dx == 1 { s.x } else { 1.0 - s.x }) * (if dy == 1 { s.y } else { 1.0 - s.y }) * (if dz == 1 { s.z } else { 1.0 - s.z });
                acc += w * hash3(seed, x + dx, y + dy, z + dz);
            }
        }
    }
    acc
}

/// Fractal noise in -1..1.
fn fbm(seed: u64, p: Vec3) -> f32 {
    let (mut a, mut f, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
    for o in 0..4 {
        sum += a * noise3(seed.wrapping_add(o), p * f);
        norm += a;
        a *= 0.5;
        f *= 2.0;
    }
    sum / norm * 2.0 - 1.0
}

/// Weights of the vertices the stroke reaches (optionally only `allowed`).
pub fn stroke_weights(mesh: &Mesh, s: &Stroke, allowed: Option<&std::collections::BTreeSet<VertId>>) -> BTreeMap<VertId, f32> {
    mesh.verts
        .iter()
        .filter(|(id, _)| allowed.is_none_or(|a| a.contains(id)))
        .filter_map(|(id, v)| {
            let d = closest_on_path(&s.path, v.pos).1;
            (d < s.radius).then(|| (*id, falloff(s.falloff, 1.0 - d / s.radius)))
        })
        .filter(|(_, w)| *w > 1e-4)
        .collect()
}

/// Apply a brush stroke; returns the number of vertices moved.
pub fn apply(mesh: &mut Mesh, brush: Brush, s: &Stroke, allowed: Option<&std::collections::BTreeSet<VertId>>) -> Result<usize> {
    if s.path.is_empty() || !(s.radius > 0.0) || !s.strength.is_finite() {
        return invalid("a stroke needs at least one point, a positive radius and a strength");
    }
    let w = stroke_weights(mesh, s, allowed);
    if w.is_empty() {
        return Ok(0);
    }
    let vf = mesh.vert_faces();
    let ve = mesh.vert_edges();
    let normals: HashMap<VertId, Vec3> = w.keys().map(|v| (*v, mesh.vert_normal(*v, &vf))).collect();
    let k = s.strength;
    match brush {
        Brush::Smooth | Brush::Crease => {
            if brush == Brush::Crease {
                for (v, wt) in &w {
                    let p = mesh.verts[v].pos;
                    let (q, _) = closest_on_path(&s.path, p);
                    let to_line = q - p;
                    let n = normals[v];
                    // Pull sideways onto the stroke and sink along the normal.
                    let side = to_line - n * to_line.dot(n);
                    mesh.verts.get_mut(v).unwrap().pos = p + side * (0.5 * k.abs() * wt) - n * (s.radius * 0.15 * k * wt);
                }
            }
            let passes = if brush == Brush::Smooth { s.iterations.max(1) } else { s.iterations };
            for _ in 0..passes {
                let pos: HashMap<VertId, Vec3> = w.keys().map(|v| (*v, mesh.verts[v].pos)).collect();
                for (v, wt) in &w {
                    let Some(es) = ve.get(v) else { continue };
                    let boundary = es.iter().any(|e| mesh_edge_faces(&vf, e.0, e.1) < 2);
                    if boundary {
                        continue;
                    }
                    let avg = es.iter().map(|e| pos.get(&e.other(*v)).copied().unwrap_or_else(|| mesh.verts[&e.other(*v)].pos)).sum::<Vec3>() / es.len() as f32;
                    let t = if brush == Brush::Smooth { k.clamp(0.0, 1.0) } else { 0.3 };
                    let p = pos[v];
                    mesh.verts.get_mut(v).unwrap().pos = p + (avg - p) * (t * wt);
                }
            }
        }
        Brush::Inflate => {
            for (v, wt) in &w {
                mesh.verts.get_mut(v).unwrap().pos += normals[v] * (s.radius * 0.2 * k * wt);
            }
        }
        Brush::Displace { scale, seed } => {
            let sc = scale.unwrap_or(s.radius * 0.25).max(1e-5);
            for (v, wt) in &w {
                let p = mesh.verts[v].pos;
                let h = fbm(seed, p / sc);
                mesh.verts.get_mut(v).unwrap().pos = p + normals[v] * (h * sc * k * wt);
            }
        }
        Brush::Flatten => {
            let total: f32 = w.values().sum();
            let c = w.iter().map(|(v, wt)| mesh.verts[v].pos * *wt).sum::<Vec3>() / total;
            let n = w.iter().map(|(v, wt)| normals[v] * *wt).sum::<Vec3>().normalize_or(Vec3::Y);
            for (v, wt) in &w {
                let p = mesh.verts[v].pos;
                mesh.verts.get_mut(v).unwrap().pos = p - n * ((p - c).dot(n) * k.clamp(0.0, 1.0) * wt);
            }
        }
    }
    Ok(w.len())
}

fn mesh_edge_faces(vf: &HashMap<VertId, Vec<crate::mesh::FaceId>>, a: VertId, b: VertId) -> usize {
    match (vf.get(&a), vf.get(&b)) {
        (Some(fa), Some(fb)) => fa.iter().filter(|f| fb.contains(f)).count(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Primitive;

    fn ball() -> Mesh {
        Primitive::UvSphere { radius: 1.0, segments: 24, rings: 12 }.build().unwrap()
    }

    fn stroke(path: Vec<Vec3>, strength: f32) -> Stroke {
        Stroke { path, radius: 0.6, strength, falloff: Falloff::Smooth, iterations: 3 }
    }

    #[test]
    fn inflate_pushes_out_only_near_the_stroke() {
        let mut m = ball();
        let before = m.clone();
        let n = apply(&mut m, Brush::Inflate, &stroke(vec![Vec3::new(0.0, 0.0, 1.0)], 1.0), None).unwrap();
        assert!(n > 0);
        for (id, v) in &m.verts {
            let (a, b) = (before.verts[id].pos.length(), v.pos.length());
            if before.verts[id].pos.z < 0.0 {
                assert!((a - b).abs() < 1e-6, "far side moved");
            }
        }
        assert!(m.verts.values().map(|v| v.pos.length()).fold(0.0, f32::max) > 1.05);
    }

    #[test]
    fn smooth_reduces_noise_and_crease_cuts_a_groove() {
        let mut m = ball();
        apply(&mut m, Brush::Displace { scale: Some(0.1), seed: 3 }, &Stroke { radius: 3.0, ..stroke(vec![Vec3::ZERO], 1.0) }, None).unwrap();
        // Spread of the radii (smoothing also shrinks a little, so not distance to 1).
        let rough = |m: &Mesh| {
            let r: Vec<f32> = m.verts.values().map(|v| v.pos.length()).collect();
            let mean = r.iter().sum::<f32>() / r.len() as f32;
            r.iter().map(|x| (x - mean).abs()).sum::<f32>()
        };
        let r0 = rough(&m);
        assert!(r0 > 0.1, "displace roughened the ball: {r0}");
        apply(&mut m, Brush::Smooth, &Stroke { radius: 3.0, ..stroke(vec![Vec3::ZERO], 0.8) }, None).unwrap();
        assert!(rough(&m) < r0 * 0.8, "smooth evened it out");
        let mut g = ball();
        let path = vec![Vec3::new(-0.5, 0.0, 0.87), Vec3::new(0.5, 0.0, 0.87)];
        apply(&mut g, Brush::Crease, &stroke(path, 1.0), None).unwrap();
        let front = g.verts.values().filter(|v| v.pos.z > 0.9 && v.pos.y.abs() < 0.1).map(|v| v.pos.length()).fold(1.0, f32::min);
        assert!(front < 0.98, "a groove along the stroke: {front}");
    }

    #[test]
    fn flatten_and_limits() {
        let mut m = ball();
        apply(&mut m, Brush::Flatten, &Stroke { radius: 0.5, ..stroke(vec![Vec3::Y], 1.0) }, None).unwrap();
        let top: Vec<f32> = m.verts.values().filter(|v| v.pos.y > 0.95).map(|v| v.pos.y).collect();
        assert!(top.windows(2).all(|w| (w[0] - w[1]).abs() < 0.02), "{top:?}");
        assert!(apply(&mut m, Brush::Smooth, &Stroke { radius: 0.0, ..stroke(vec![Vec3::Y], 1.0) }, None).is_err());
        // Only allowed vertices move.
        let allowed = std::collections::BTreeSet::new();
        assert_eq!(apply(&mut m, Brush::Inflate, &stroke(vec![Vec3::Y], 1.0), Some(&allowed)).unwrap(), 0);
    }
}
