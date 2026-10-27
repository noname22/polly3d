//! GPU render tests through the Bevy service (ported from polly-render).
//! They skip when no adapter (not even a software one) is available.

use polly_bevy::BevyRenderer;
use polly_core::glam::Vec3;
use polly_core::primitives::Primitive;
use polly_core::scene::{Scene, Texture};
use polly_render::{Overlays, RenderOptions, View, resolve_camera};

fn renderer() -> Option<BevyRenderer> {
    match BevyRenderer::global() {
        Ok(r) => {
            eprintln!("adapter: {}", r.adapter_name());
            Some(r)
        }
        Err(e) => {
            eprintln!("skipping GPU test: {e:#}");
            None
        }
    }
}

fn out_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
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
    let Some(r) = renderer() else { return };
    let (scene, _) = cube_scene();
    let bg = bg_u8(&scene);
    for view in [View::Front, View::Persp, View::Top, View::Iso] {
        let out = r.render(&scene, &plain(view.clone())).unwrap();
        let img = &out.image;
        img.save(out_dir().join(format!("bevy_cube_{}.png", view.name()))).unwrap();
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
    let Some(r) = renderer() else { return };
    let (mut scene, id) = cube_scene();
    scene.object_mut(id).unwrap().transform.scale = Vec3::new(-1.0, 1.0, 1.0);
    let out = r.render(&scene, &plain(View::Front)).unwrap();
    assert!(!close(out.image.get_pixel(128, 128).0, bg_u8(&scene), 6));
}

#[test]
fn face_id_labels_are_drawn() {
    let Some(r) = renderer() else { return };
    let (scene, _) = cube_scene();
    let mut o = RenderOptions { overlays: Overlays { face_ids: true, ..Default::default() }, ..Default::default() };
    let persp = r.render(&scene, &o).unwrap();
    persp.image.save(out_dir().join("bevy_face_ids.png")).unwrap();
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
    let Some(r) = renderer() else { return };
    let (scene, _) = cube_scene();
    let out = r.contact_sheet(&scene, &RenderOptions { width: 512, height: 384, ..Default::default() }).unwrap();
    out.image.save(out_dir().join("bevy_sheet.png")).unwrap();
    assert_eq!(out.image.dimensions(), (512, 384));
    assert_eq!(out.tiles.len(), 4);
}

#[test]
fn camera_matches_render_and_textures_render() {
    let Some(r) = renderer() else { return };
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

/// Large (viewer-sized) renders beyond the agent's 2048 px limit.
#[test]
fn renders_at_viewer_size() {
    let Some(r) = renderer() else { return };
    let (scene, _) = cube_scene();
    let out = r.render_sized(&scene, &RenderOptions::default(), [2200, 300]).unwrap();
    assert_eq!(out.image.dimensions(), (2200, 300));
}

fn unlit_scene(color: [f32; 4]) -> (Scene, u32) {
    let (mut s, id) = cube_scene();
    let mid = s.add_material(polly_core::scene::Material { unlit: true, ..polly_core::scene::Material::new("u", color) });
    for f in std::sync::Arc::make_mut(&mut s.object_mut(id).unwrap().mesh).faces.values_mut() {
        f.material = mid;
    }
    (s, id)
}

/// Rendered silhouettes line up with polly's projection (picking, labels).
#[test]
fn silhouette_matches_camera_projection() {
    let Some(r) = renderer() else { return };
    let (mut scene, id) = unlit_scene([1.0; 4]);
    scene.object_mut(id).unwrap().transform.rotation = polly_core::glam::Quat::from_rotation_y(0.5);
    for view in [View::Persp, View::Front, View::Orbit { yaw_deg: 200.0, pitch_deg: -20.0, distance: None }] {
        let o = RenderOptions { view: view.clone(), width: 320, height: 200, overlays: Overlays::none(), legend: false, ..Default::default() };
        let out = r.render(&scene, &o).unwrap();
        // Every pixel centre whose ray hits the cube is white, every miss is background.
        let mut wrong = 0;
        for y in (0..200).step_by(3) {
            for x in (0..320).step_by(3) {
                let hit = polly_render::pick(&scene, &out.camera, x as f32 + 0.5, y as f32 + 0.5).is_some();
                let p = out.image.get_pixel(x, y).0;
                let white = p[0] > 200 && p[1] > 200 && p[2] > 200;
                let bg = close(p, bg_u8(&scene), 3);
                // Partly covered (anti-aliased) edge pixels are neither: skip them.
                if (hit && bg) || (!hit && white) {
                    wrong += 1;
                    eprintln!("{view:?} ({x},{y}) hit {hit} pixel {p:?}");
                }
            }
        }
        assert!(wrong == 0, "{view:?}: {wrong} pixels disagree with picking");
    }
}

#[test]
fn reference_images_behind_and_on_top() {
    let Some(r) = renderer() else { return };
    let (mut scene, _) = unlit_scene([1.0; 4]);
    let red = polly_core::reference::Reference {
        name: "red".into(),
        view: polly_core::reference::RefView::Front,
        width: 4,
        height: 4,
        pixels: std::sync::Arc::new([255, 0, 0, 255].repeat(16)),
        crop: None,
        center: [0.0, 0.0],
        size: 3.0,
        layer: polly_core::reference::Layer::Behind,
        opacity: 1.0,
        visible: true,
        flip: false,
    };
    scene.references.push(red);
    let o = RenderOptions { view: View::Front, width: 200, height: 200, overlays: Overlays::none(), legend: false, ..Default::default() };
    let behind = r.render(&scene, &o).unwrap().image;
    behind.save(out_dir().join("bevy_ref_behind.png")).unwrap();
    let (centre, edge) = (behind.get_pixel(100, 100).0, behind.get_pixel(40, 100).0);
    assert!(centre[0] > 240 && centre[1] > 240, "cube in front of the reference: {centre:?}");
    assert!(edge[0] > 200 && edge[1] < 40, "reference around the cube: {edge:?}");
    let front = r.render(&scene, &RenderOptions { reference_layer: Some(polly_core::reference::Layer::Front), reference_opacity: Some(0.5), ..o.clone() }).unwrap().image;
    let c = front.get_pixel(100, 100).0;
    assert!(c[0] > 200 && c[1] > 100 && c[1] < 230, "half-transparent red over the white cube: {c:?}");
    // Not shown in other views.
    let side = r.render(&scene, &RenderOptions { view: View::Right, ..o }).unwrap().image;
    assert!(!side.pixels().any(|p| p[0] > 200 && p[1] < 50), "right view shows no front reference");
}

#[test]
fn pbr_channels_reach_the_shader() {
    let Some(r) = renderer() else { return };
    let mut scene = Scene::default();
    let ball = Primitive::UvSphere { radius: 0.5, segments: 24, rings: 12 }.build().unwrap();
    let id = scene.add_object("Ball", ball);
    for f in std::sync::Arc::make_mut(&mut scene.object_mut(id).unwrap().mesh).faces.values_mut() {
        f.smooth = true;
    }
    let o = RenderOptions { view: View::Front, width: 160, height: 160, overlays: Overlays::none(), legend: false, ..Default::default() };
    let mut shade = |metallic: f32, roughness: f32| {
        let m = scene.materials.get_mut(&0).unwrap();
        m.metallic = metallic;
        m.roughness = roughness;
        r.render(&scene, &o).unwrap().image
    };
    let rough = shade(0.0, 1.0);
    let shiny = shade(1.0, 0.2);
    let diff: u64 = rough.pixels().zip(shiny.pixels()).map(|(a, b)| (a[0] as i32 - b[0] as i32).unsigned_abs() as u64).sum();
    assert!(diff > 160 * 160 * 5, "metallic/roughness changed little: {diff}");
}

#[test]
fn selected_faces_are_tinted() {
    let Some(r) = renderer() else { return };
    let (mut scene, id) = unlit_scene([1.0; 4]);
    let mesh = scene.objects[&id].mesh.clone();
    let front_face = *mesh.faces.iter().find(|(_, f)| mesh.face_normal(f).z > 0.9).unwrap().0;
    scene.object_mut(id).unwrap().selection.faces.insert(front_face);
    let o = RenderOptions { view: View::Front, width: 128, height: 128, overlays: Overlays { selection: true, ..Overlays::none() }, legend: false, ..Default::default() };
    let p = r.render(&scene, &o).unwrap().image.get_pixel(64, 64).0;
    // 50 % of linear (1, 0.33, 0.02) over white, as the old renderer mixed it.
    assert!(p[0] > 245 && (200..226).contains(&p[1]) && (175..200).contains(&p[2]), "orange tint on white: {p:?}");
}
