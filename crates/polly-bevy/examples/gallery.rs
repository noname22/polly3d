//! Render a gallery of example images: `cargo run -p polly-bevy --example gallery -- <out_dir>`.

use polly_core::glam::{Quat, Vec2, Vec3, Vec4};
use polly_core::ids::Selection;
use polly_core::mesh::{Edge, Mesh};
use polly_core::primitives::Primitive;
use polly_core::scene::{Material, Scene, Texture};
use polly_bevy::BevyRenderer;
use polly_render::{Overlays, RenderOptions, View, render_uv_layout};

fn box_uvs(m: &mut Mesh) {
    for f in m.faces.values_mut() {
        if f.len() == 4 {
            let uv = [Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)];
            for (c, u) in f.corners.iter_mut().zip(uv) {
                c.uv = u;
            }
        }
    }
}

fn crate_texture() -> Texture {
    let mut t = Texture::new("crate", 16, 16, [150, 100, 50, 255]);
    let px = t.pixels_mut();
    for y in 0..16u32 {
        for x in 0..16u32 {
            let i = ((y * 16 + x) * 4) as usize;
            let border = x == 0 || y == 0 || x == 15 || y == 15;
            let diag = x == y || x + y == 15;
            let plank = y % 5 == 0;
            let c: [u8; 4] = if border {
                [70, 45, 25, 255]
            } else if diag {
                [120, 80, 40, 255]
            } else if plank {
                [135, 90, 45, 255]
            } else {
                [180, 125, 65, 255]
            };
            px[i..i + 4].copy_from_slice(&c);
        }
    }
    t
}

fn main() -> anyhow::Result<()> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "gallery".into());
    std::fs::create_dir_all(&out)?;
    let r = BevyRenderer::global()?;
    eprintln!("adapter: {}", r.adapter_name());

    let mut scene = Scene::default();
    let red = scene.add_material(Material::new("red", [0.8, 0.15, 0.1, 1.0]));
    let tex = scene.add_texture(crate_texture());
    let mut crate_mat = Material::new("crate", [1.0, 1.0, 1.0, 1.0]);
    crate_mat.texture = Some(tex);
    let crate_mat = scene.add_material(crate_mat);

    // Grey cube with a selection.
    let cube = scene.add_object("Cube", Primitive::Cube { size: [1.0, 1.0, 1.0] }.build()?);
    {
        let o = scene.object_mut(cube)?;
        o.transform.translation = Vec3::new(0.0, 0.5, 0.0);
        o.selection = Selection { faces: [3].into(), edges: [Edge(1, 5)].into(), verts: [6].into() };
    }
    // Smooth red sphere.
    let mut sphere = Primitive::UvSphere { radius: 0.5, segments: 12, rings: 8 }.build()?;
    for f in sphere.faces.values_mut() {
        f.smooth = true;
        f.material = red;
    }
    let s = scene.add_object("Sphere", sphere);
    scene.object_mut(s)?.transform.translation = Vec3::new(1.4, 0.5, -0.3);
    // Textured crate, rotated.
    let mut cm = Primitive::Cube { size: [0.8, 0.8, 0.8] }.build()?;
    box_uvs(&mut cm);
    for f in cm.faces.values_mut() {
        f.material = crate_mat;
    }
    let c = scene.add_object("Crate", cm);
    {
        let o = scene.object_mut(c)?;
        o.transform.translation = Vec3::new(-1.3, 0.4, 0.3);
        o.transform.rotation = Quat::from_rotation_y(0.5);
    }
    // Vertex-coloured cone.
    let mut cone = Primitive::Cylinder { radius_bottom: 0.4, radius_top: 0.0, height: 1.0, segments: 6, height_segments: 1, caps: true }.build()?;
    for f in cone.faces.values_mut() {
        for (i, cn) in f.corners.iter_mut().enumerate() {
            cn.color = if i % 2 == 0 { Vec4::new(0.2, 0.9, 0.3, 1.0) } else { Vec4::new(0.1, 0.4, 0.9, 1.0) };
        }
    }
    let k = scene.add_object("Cone", cone);
    scene.object_mut(k)?.transform.translation = Vec3::new(0.2, 0.5, 1.4);
    scene.active = Some(cube);

    let save = |name: &str, img: &image::RgbaImage| -> anyhow::Result<()> {
        let p = format!("{out}/{name}.png");
        img.save(&p)?;
        eprintln!("wrote {p}");
        Ok(())
    };

    let rend = r.render(&scene, &RenderOptions::default())?;
    eprintln!("persp notes: {:?}", rend.notes);
    save("scene_persp", &rend.image)?;

    let ids = Overlays { face_ids: true, vert_ids: true, verts: true, ..Default::default() };
    let rend = r.render(&scene, &RenderOptions { focus: Some(vec![cube]), overlays: ids.clone(), ..Default::default() })?;
    eprintln!("cube ids: labels={} notes={:?}", rend.labels, rend.notes);
    save("cube_ids_persp", &rend.image)?;

    let all = Overlays { face_ids: true, vert_ids: true, edge_ids: true, verts: true, ..Default::default() };
    let rend = r.render(&scene, &RenderOptions { view: View::Front, focus: Some(vec![cube]), isolate: true, overlays: all, ..Default::default() })?;
    eprintln!("cube front all ids: labels={} notes={:?}", rend.labels, rend.notes);
    save("cube_front_all_ids", &rend.image)?;

    let rend = r.render(&scene, &RenderOptions { focus: Some(vec![s]), overlays: Overlays { face_ids: true, ..Default::default() }, ..Default::default() })?;
    eprintln!("sphere ids: labels={} notes={:?}", rend.labels, rend.notes);
    save("sphere_face_ids", &rend.image)?;

    let extra = Overlays { seams: true, sharp: true, normals: true, bounds: true, object_names: true, ..Default::default() };
    {
        let o = scene.object_mut(cube)?;
        let m = std::sync::Arc::make_mut(&mut o.mesh);
        m.seams.extend([Edge(0, 1), Edge(1, 2), Edge(2, 3)]);
        m.sharp.extend([Edge(4, 5), Edge(5, 6)]);
    }
    let rend = r.render(&scene, &RenderOptions { view: View::Iso, overlays: extra, ..Default::default() })?;
    save("overlays_iso", &rend.image)?;

    let rend = r.contact_sheet(&scene, &RenderOptions { width: 1024, height: 1024, ..Default::default() })?;
    eprintln!("sheet notes: {:?}", rend.notes);
    save("contact_sheet", &rend.image)?;

    let rend = r.render(&scene, &RenderOptions { view: View::Top, filter: Some(polly_core::scene::Filter::Linear), ..Default::default() })?;
    save("top_linear", &rend.image)?;

    save("uv_crate", &render_uv_layout(&scene, c, Some(tex), 512, true))?;
    save("uv_cube_checker", &render_uv_layout(&scene, cube, None, 256, true))?;
    Ok(())
}
