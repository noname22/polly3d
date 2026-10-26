use glam::{Quat, Vec3};
use polly_core::primitives::Primitive;
use polly_core::scene::{Scene, Texture};
use polly_render::{Overlays, RenderOptions, Renderer, View, pick, render_uv_layout, resolve_camera};

fn renderer() -> Option<Renderer> {
    match Renderer::headless() {
        Ok(r) => {
            eprintln!("adapter: {:?}", r.adapter_info().name);
            Some(r)
        }
        Err(e) => {
            eprintln!("skipping GPU test: no adapter ({e:#})");
            None
        }
    }
}

fn cube_scene() -> (Scene, u32) {
    let mut s = Scene::default();
    let id = s.add_object("Cube", Primitive::Cube { size: [1.0, 1.0, 1.0] }.build().unwrap());
    (s, id)
}

fn bg_u8(s: &Scene) -> [u8; 3] {
    let b = s.background;
    [(b[0] * 255.0).round() as u8, (b[1] * 255.0).round() as u8, (b[2] * 255.0).round() as u8]
}

fn close(a: [u8; 4], b: [u8; 3], tol: i32) -> bool {
    (0..3).all(|i| (a[i] as i32 - b[i] as i32).abs() <= tol)
}

fn plain(view: View) -> RenderOptions {
    RenderOptions { view, width: 256, height: 256, overlays: Overlays::none(), legend: false, ..Default::default() }
}

#[test]
fn renders_cube_centre_and_background_corners() {
    let Some(mut r) = renderer() else { return };
    let (scene, _) = cube_scene();
    let bg = bg_u8(&scene);
    for view in [View::Front, View::Persp, View::Top, View::Iso] {
        let out = r.render(&scene, &plain(view.clone())).unwrap();
        let img = &out.image;
        assert_eq!(img.dimensions(), (256, 256));
        let c = img.get_pixel(128, 128).0;
        assert!(!close(c, bg, 6), "{view:?}: centre should be geometry, got {c:?}");
        for (x, y) in [(1, 1), (254, 1), (1, 254), (254, 254)] {
            let p = img.get_pixel(x, y).0;
            assert!(close(p, bg, 2), "{view:?}: corner ({x},{y}) should be background {bg:?}, got {p:?}");
        }
    }
}

#[test]
fn mirrored_object_is_not_culled() {
    let Some(mut r) = renderer() else { return };
    let (mut scene, id) = cube_scene();
    scene.object_mut(id).unwrap().transform.scale = Vec3::new(-1.0, 1.0, 1.0);
    let out = r.render(&scene, &plain(View::Front)).unwrap();
    assert!(!close(out.image.get_pixel(128, 128).0, bg_u8(&scene), 6));
}

#[test]
fn face_id_labels_are_drawn() {
    let Some(mut r) = renderer() else { return };
    let (scene, _) = cube_scene();
    let mut o = RenderOptions { overlays: Overlays { face_ids: true, ..Default::default() }, ..Default::default() };
    let persp = r.render(&scene, &o).unwrap();
    assert_eq!(persp.labels, 3, "three cube faces visible in the 3/4 view; notes {:?}", persp.notes);
    o.view = View::Front;
    let front = r.render(&scene, &o).unwrap();
    assert_eq!(front.labels, 1);
    o.overlays.backface_labels = true;
    let all = r.render(&scene, &o).unwrap();
    assert!(all.labels > 1);
    o.overlays = Overlays { face_ids: true, vert_ids: true, edge_ids: true, ..Default::default() };
    o.view = View::Persp;
    let many = r.render(&scene, &o).unwrap();
    assert!(many.labels >= 3 + 7, "labels {} notes {:?}", many.labels, many.notes);
}

#[test]
fn contact_sheet_size() {
    let Some(mut r) = renderer() else { return };
    let (scene, _) = cube_scene();
    let out = r.contact_sheet(&scene, &RenderOptions { width: 512, height: 384, ..Default::default() }).unwrap();
    assert_eq!(out.image.dimensions(), (512, 384));
    assert_eq!(out.tiles.len(), 4);
}

#[test]
fn camera_matches_render_and_textures_render() {
    let Some(mut r) = renderer() else { return };
    let (mut scene, id) = cube_scene();
    let t = scene.add_texture(Texture::new("t", 4, 4, [255, 0, 0, 255]));
    let mut m = polly_core::scene::Material::new("m", [1.0; 4]);
    m.texture = Some(t);
    m.unlit = true;
    let mid = scene.add_material(m);
    let obj = scene.object_mut(id).unwrap();
    for f in std::sync::Arc::make_mut(&mut obj.mesh).faces.values_mut() {
        f.material = mid;
    }
    let o = plain(View::Front);
    let out = r.render(&scene, &o).unwrap();
    assert_eq!(out.camera, resolve_camera(&scene, &o));
    let c = out.image.get_pixel(128, 128).0;
    assert!(c[0] > 240 && c[1] < 10 && c[2] < 10, "unlit red texture, got {c:?}");
}

#[test]
fn render_to_view_offscreen_texture() {
    let Some(mut r) = renderer() else { return };
    let (scene, _) = cube_scene();
    for format in [wgpu::TextureFormat::Bgra8UnormSrgb, wgpu::TextureFormat::Rgba8Unorm] {
        let tex = r.device().create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: 200, height: 100, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        r.render_to_view(&scene, &RenderOptions::default(), &view, format, [200, 100]).unwrap();
    }
    r.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

// ---- CPU-only tests ----

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
