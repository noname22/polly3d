use std::collections::BTreeSet;

use glam::{Vec2, Vec3};
use polly_core::mesh::{FaceId, Mesh};
use polly_core::primitives::{Primitive, cube, plane};

use crate::*;

fn all(m: &Mesh) -> BTreeSet<FaceId> {
    m.faces.keys().copied().collect()
}

fn assert_clean(r: &UvReport) {
    assert!(r.overlapping_faces.is_empty(), "overlaps: {r:#?}");
    assert!(r.out_of_bounds_faces.is_empty(), "out of bounds: {r:#?}");
    assert!(r.flipped_faces.is_empty(), "flipped: {r:#?}");
}

fn cylinder(caps: bool) -> Mesh {
    Primitive::Cylinder { radius_bottom: 0.5, radius_top: 0.5, height: 1.0, segments: 8, height_segments: 2, caps }
        .build()
        .unwrap()
}

#[test]
fn cube_auto_seams_unwrap() {
    let mut m = cube(Vec3::ONE);
    let faces = all(&m);
    assert_eq!(auto_seams(&mut m, &faces, 30.0), 12);
    let r = unwrap(&mut m, &faces, &UnwrapOptions::default()).unwrap();
    assert_eq!(r.islands.len(), 6);
    assert_clean(&r);
    assert!(r.max_stretch < 1.05, "{r:#?}");
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
}

#[test]
fn closed_cube_gets_cut_into_disks() {
    let mut m = cube(Vec3::ONE);
    let faces = all(&m);
    assert!(auto_seams(&mut m, &faces, 120.0) > 0);
    for isl in islands(&m, &faces) {
        assert!(chart::Chart::build(&m, &isl).is_disk());
    }
    let r = unwrap(&mut m, &faces, &UnwrapOptions::default()).unwrap();
    assert_clean(&r);
}

#[test]
fn plane_grid_lscm_has_no_stretch() {
    let mut m = plane(Vec2::new(2.0, 1.0), 4, 2);
    let faces = all(&m);
    let o = UnwrapOptions {
        straighten_quads: false,
        pack: PackOptions { pixel_snap: false, ..Default::default() },
        ..Default::default()
    };
    let r = unwrap(&mut m, &faces, &o).unwrap();
    assert_eq!(r.islands.len(), 1);
    assert!(r.max_stretch < 1.01, "{r:#?}");
    assert_clean(&r);
    // 2:1 plane fills the width of a square texture.
    assert!(r.coverage > 0.45, "{r:#?}");
}

#[test]
fn cylinder_unwraps_without_overlaps() {
    for (caps, angle) in [(true, 60.0), (false, 60.0), (true, 30.0)] {
        let mut m = cylinder(caps);
        let faces = all(&m);
        auto_seams(&mut m, &faces, angle);
        for isl in islands(&m, &faces) {
            assert!(chart::Chart::build(&m, &isl).is_disk());
        }
        for straighten_quads in [true, false] {
            let o = UnwrapOptions { straighten_quads, ..Default::default() };
            let r = unwrap(&mut m, &faces, &o).unwrap();
            assert_clean(&r);
            assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        }
    }
}

#[test]
fn torus_and_sphere_unwrap() {
    for p in [
        Primitive::Torus { major_radius: 0.5, minor_radius: 0.2, segments: 8, sides: 6 },
        Primitive::UvSphere { radius: 0.5, segments: 8, rings: 6 },
        Primitive::IcoSphere { radius: 0.5, subdivisions: 1 },
    ] {
        let mut m = p.build().unwrap();
        let faces = all(&m);
        auto_seams(&mut m, &faces, 89.0);
        for isl in islands(&m, &faces) {
            assert!(chart::Chart::build(&m, &isl).is_disk(), "{p:?}");
        }
        let r = unwrap(&mut m, &faces, &UnwrapOptions::default()).unwrap();
        assert!(r.out_of_bounds_faces.is_empty());
        assert!(r.warnings.iter().all(|w| !w.contains("not disks")), "{:?}", r.warnings);
    }
}

#[test]
fn pack_respects_padding_and_snaps() {
    let mut m = cylinder(true);
    let faces = all(&m);
    auto_seams(&mut m, &faces, 30.0);
    let size = 48;
    let o = UnwrapOptions {
        pack: PackOptions { texture_size: [size, size], padding_px: 2, ..Default::default() },
        ..Default::default()
    };
    let r = unwrap(&mut m, &faces, &o).unwrap();
    assert!(r.islands.len() >= 10);
    for (i, a) in r.islands.iter().enumerate() {
        let [ax, ay, aw, ah] = a.pixel_rect;
        assert!(ax + aw <= size && ay + ah <= size);
        for b in &r.islands[i + 1..] {
            let [bx, by, bw, bh] = b.pixel_rect;
            let apart = ax + aw + 2 <= bx || bx + bw + 2 <= ax || ay + ah + 2 <= by || by + bh + 2 <= ay;
            assert!(apart, "{:?} vs {:?}", a.pixel_rect, b.pixel_rect);
        }
    }
    for f in m.faces.values() {
        for c in &f.corners {
            let p = c.uv * size as f32;
            assert!((p - p.round()).abs().max_element() < 1e-3, "{p}");
        }
    }
}

#[test]
fn pack_with_texel_density() {
    let mut m = cube(Vec3::ONE);
    let faces = all(&m);
    auto_seams(&mut m, &faces, 30.0);
    let o = UnwrapOptions {
        pack: PackOptions { texture_size: [64, 64], texel_density: Some(8.0), ..Default::default() },
        ..Default::default()
    };
    let r = unwrap(&mut m, &faces, &o).unwrap();
    assert!((r.texel_density - 8.0).abs() < 0.01, "{r:#?}");
    assert!(r.islands.iter().all(|i| i.pixel_rect[2] == 8 && i.pixel_rect[3] == 8));
}

#[test]
fn per_face_quad_hits_corners() {
    let mut m = cube(Vec3::ONE);
    let faces = all(&m);
    project(&mut m, &faces, &Projection::PerFace).unwrap();
    let corners: BTreeSet<[i32; 2]> = [[0, 0], [0, 1], [1, 0], [1, 1]].into();
    for f in m.faces.values() {
        let got: BTreeSet<[i32; 2]> = f.corners.iter().map(|c| [c.uv.x as i32, c.uv.y as i32]).collect();
        assert_eq!(got, corners);
        assert!(f.corners.iter().all(|c| c.uv == c.uv.round()));
    }
    // Side face: the top edge of the texture (v = 0) is at the top of the cube.
    let r = report(&m, &faces, [16, 16]);
    assert!(r.flipped_faces.is_empty());
    for f in m.faces.values() {
        if m.face_normal(f).y.abs() < 0.5 {
            for (c, v) in f.corners.iter().zip(&f.verts) {
                assert_eq!(c.uv.y == 0.0, m.pos(*v).y > 0.0);
            }
        }
    }
}

#[test]
fn projections_fit_unit_square() {
    let mut m = cylinder(true);
    let faces = all(&m);
    for p in [
        Projection::Planar { normal: None },
        Projection::Planar { normal: Some([0.0, 0.0, 1.0]) },
        Projection::Box,
        Projection::Cylindrical { axis: None },
        Projection::Spherical,
    ] {
        let res = project(&mut m, &faces, &p);
        if matches!(p, Projection::Planar { normal: None }) {
            assert!(res.is_err()); // closed mesh: average normal is zero
            continue;
        }
        res.unwrap();
        let r = report(&m, &faces, [64, 64]);
        assert!(r.out_of_bounds_faces.is_empty(), "{p:?}");
        let (lo, hi) = uv_bounds(&m, &faces);
        assert!(lo.min_element().abs() < 1e-5 && (hi.max_element() - 1.0).abs() < 1e-5, "{p:?}");
    }
    // Cylindrical: sides and caps are separate and nothing overlaps or wraps.
    project(&mut m, &faces, &Projection::Cylindrical { axis: None }).unwrap();
    let r = report(&m, &faces, [64, 64]);
    assert_clean(&r);
    for f in m.faces.values() {
        let (lo, hi) = f.corners.iter().fold((Vec2::MAX, Vec2::MIN), |(lo, hi), c| (lo.min(c.uv), hi.max(c.uv)));
        assert!(hi.x - lo.x < 0.5);
    }
}

#[test]
fn transform_rotates_about_center() {
    let mut m = plane(Vec2::ONE, 1, 1);
    let faces = all(&m);
    project(&mut m, &faces, &Projection::PerFace).unwrap();
    let before: Vec<Vec2> = m.faces.values().next().unwrap().corners.iter().map(|c| c.uv).collect();
    let t = TransformUv { rotate_deg: 90.0, scale: [0.5, 0.5], translate: [0.25, 0.0], ..Default::default() };
    transform_uvs(&mut m, &faces, &t).unwrap();
    let after: Vec<Vec2> = m.faces.values().next().unwrap().corners.iter().map(|c| c.uv).collect();
    for (b, a) in before.iter().zip(&after) {
        let d = (*b - 0.5) * 0.5;
        // Visually counter-clockwise: right (+u) goes to up (-v).
        let expect = Vec2::new(0.5 + d.y + 0.25, 0.5 - d.x);
        assert!(a.distance(expect) < 1e-5, "{a} vs {expect}");
    }
}

#[test]
fn options_deserialize_from_partial_json() {
    let o: UnwrapOptions = serde_json::from_str(r#"{"method":"conformal","pack":{"texture_size":[32,32]}}"#).unwrap();
    assert_eq!(o.pack.texture_size, [32, 32]);
    assert_eq!(o.pack.padding_px, 1);
    let p: Projection = serde_json::from_str(r#"{"method":"cylindrical"}"#).unwrap();
    assert!(matches!(p, Projection::Cylindrical { axis: None }));
    let t: TransformUv = serde_json::from_str(r#"{"rotate_deg":90}"#).unwrap();
    assert_eq!(t.scale, [1.0, 1.0]);
}

