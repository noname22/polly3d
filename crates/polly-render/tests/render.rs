//! CPU tests: picking, cameras, UV layout, options. GPU render tests live in polly-bevy.

use glam::{Quat, Vec3};
use polly_core::primitives::Primitive;
use polly_core::scene::{Scene, Texture};
use polly_render::{RenderOptions, View, pick, render_uv_layout, resolve_camera};

fn cube_scene() -> (Scene, u32) {
    let mut s = Scene::default();
    let id = s.add_object("Cube", Primitive::Cube { size: [1.0, 1.0, 1.0] }.build().unwrap());
    (s, id)
}

#[test]
fn pick_front_centre_hits_plus_z_face() {
    let (scene, id) = cube_scene();
    let cam = resolve_camera(&scene, &RenderOptions { view: View::Front, width: 256, height: 256, ..Default::default() });
    let hit = pick(&scene, &cam, 128.0, 128.0).expect("hit");
    assert_eq!(hit.object, id);
    let mesh = &scene.objects[&id].mesh;
    let n = mesh.face_normal(&mesh.faces[&hit.face]);
    assert!(n.z > 0.99, "expected +Z face, got f{} normal {n:?}", hit.face);
    assert!((hit.position[2] - 0.5).abs() < 1e-4);
    assert!(hit.normal[2] > 0.99);
    assert!(pick(&scene, &cam, 2.0, 2.0).is_none());
}

#[test]
fn pick_rotated_persp() {
    let (mut scene, id) = cube_scene();
    scene.object_mut(id).unwrap().transform.rotation = Quat::from_rotation_y(0.4);
    let cam = resolve_camera(&scene, &RenderOptions::default());
    let hit = pick(&scene, &cam, 256.0, 256.0).expect("hit");
    let d = (Vec3::from(hit.position) - cam.eye_v()).length();
    assert!((d - hit.distance).abs() < 1e-3);
}

#[test]
fn project_ray_roundtrip() {
    let (scene, _) = cube_scene();
    for view in [View::Persp, View::Front, View::Iso, View::Orbit { yaw_deg: 200.0, pitch_deg: -30.0, distance: None }] {
        let cam = resolve_camera(&scene, &RenderOptions { view: view.clone(), width: 320, height: 200, ..Default::default() });
        for p in [Vec3::new(0.3, 0.2, 0.5), Vec3::new(-0.5, -0.5, -0.5), Vec3::ZERO] {
            let [x, y] = cam.project(p).unwrap();
            let (o, d) = cam.ray(x, y);
            let closest = o + d * (p - o).dot(d);
            assert!((closest - p).length() < 1e-3, "{view:?} {p:?}: {closest:?}");
            assert!((d.length() - 1.0).abs() < 1e-4);
        }
        // Screen centre projects the target.
        let [x, y] = cam.project(Vec3::from(cam.target)).unwrap();
        assert!((x - 160.0).abs() < 0.01 && (y - 100.0).abs() < 0.01);
    }
}

#[test]
fn view_axis_hints() {
    let (scene, _) = cube_scene();
    let hint = |v| resolve_camera(&scene, &RenderOptions { view: v, ..Default::default() }).axis_hint();
    assert_eq!(hint(View::Front).as_deref(), Some("+X right, +Y up"));
    assert_eq!(hint(View::Right).as_deref(), Some("-Z right, +Y up"));
    assert_eq!(hint(View::Top).as_deref(), Some("+X right, -Z up"));
    assert_eq!(hint(View::Back).as_deref(), Some("-X right, +Y up"));
    assert_eq!(hint(View::Persp), None);
}

#[test]
fn uv_layout_dimensions() {
    let (mut scene, id) = cube_scene();
    let img = render_uv_layout(&scene, id, None, 256, true);
    assert_eq!(img.dimensions(), (256, 256));
    let t = scene.add_texture(Texture::new("wide", 32, 16, [10, 200, 30, 255]));
    let img = render_uv_layout(&scene, id, Some(t), 256, true);
    assert_eq!(img.dimensions(), (256, 128));
    let img = render_uv_layout(&scene, id, Some(t), 10_000, false);
    assert_eq!(img.dimensions(), (2048, 1024));
}

#[test]
fn options_deserialize_with_defaults() {
    let o: RenderOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(o.width, 512);
    assert!(o.overlays.wireframe && o.overlays.selection && o.overlays.grid && !o.overlays.face_ids);
    assert_eq!(o.view, View::Persp);
    let o: RenderOptions =
        serde_json::from_str(r#"{"view":{"orbit":{"yaw_deg":30,"pitch_deg":10}},"overlays":{"face_ids":true},"zoom":2}"#).unwrap();
    assert!(o.overlays.face_ids && o.overlays.wireframe);
    assert_eq!(o.view, View::Orbit { yaw_deg: 30.0, pitch_deg: 10.0, distance: None });
    let o: RenderOptions = serde_json::from_str(r#"{"view":"front","filter":"linear"}"#).unwrap();
    assert_eq!(o.view, View::Front);
    let _schema = schemars::schema_for!(RenderOptions);
}

/// Orbiting must not "breathe": the eye stays at a constant distance from the
/// target regardless of how the (here very elongated) model is oriented.
#[test]
fn orbit_distance_is_independent_of_angle() {
    let mut scene = Scene::default();
    scene.add_object("Plank", polly_core::primitives::cube(glam::Vec3::new(4.0, 0.3, 0.6)));
    let dist = |yaw: f32, pitch: f32| {
        let o = RenderOptions { view: View::Orbit { yaw_deg: yaw, pitch_deg: pitch, distance: None }, ..RenderOptions::default() };
        let c = polly_render::resolve_camera(&scene, &o);
        (glam::Vec3::from(c.eye) - glam::Vec3::from(c.target)).length()
    };
    let d0 = dist(0.0, 20.0);
    for yaw in (0..360).step_by(15) {
        for pitch in [-60.0, 0.0, 20.0, 80.0] {
            let d = dist(yaw as f32, pitch);
            assert!((d - d0).abs() < 1e-4 * d0, "yaw {yaw} pitch {pitch}: {d} vs {d0}");
        }
    }
}
