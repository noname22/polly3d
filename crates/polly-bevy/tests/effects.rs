//! Lighting and camera effects: environments, sun + shadows, SSAO, tonemapping, bloom.

use polly_bevy::BevyRenderer;
use polly_core::glam::Vec3;
use polly_core::lighting::{Lighting, Sun, Tonemap};
use polly_core::primitives::Primitive;
use polly_core::scene::{Material, Scene};
use polly_render::{Overlays, RenderOptions, View};

fn scene() -> Scene {
    let mut s = Scene::default();
    s.add_object("Ground", polly_core::primitives::plane(polly_core::glam::Vec2::splat(6.0), 1, 1));
    let c = s.add_object("Box", Primitive::Cube { size: [1.0, 1.0, 1.0] }.build().unwrap());
    s.object_mut(c).unwrap().transform.translation = Vec3::new(-0.8, 0.5, 0.0);
    let metal = s.add_material(Material { metallic: 1.0, roughness: 0.25, ..Material::new("metal", [0.9, 0.9, 0.9, 1.0]) });
    let glow = s.add_material(Material { emissive: [1.0, 0.4, 0.1], emissive_strength: 8.0, ..Material::new("glow", [0.1, 0.1, 0.1, 1.0]) });
    let b = s.add_object("Ball", Primitive::UvSphere { radius: 0.5, segments: 24, rings: 12 }.build().unwrap());
    s.object_mut(b).unwrap().transform.translation = Vec3::new(0.8, 0.5, 0.0);
    let g = s.add_object("Glow", Primitive::Cube { size: [0.3, 0.3, 0.3] }.build().unwrap());
    s.object_mut(g).unwrap().transform.translation = Vec3::new(0.0, 0.15, 1.0);
    for (id, m) in [(b, metal), (g, glow)] {
        for f in std::sync::Arc::make_mut(&mut s.object_mut(id).unwrap().mesh).faces.values_mut() {
            f.material = m;
            f.smooth = id == b;
        }
    }
    s
}

fn opts(l: Lighting) -> RenderOptions {
    RenderOptions { view: View::Persp, width: 320, height: 240, overlays: Overlays::none(), legend: false, lighting: Some(l), ..Default::default() }
}

fn luma(img: &image::RgbaImage) -> f64 {
    img.pixels().map(|p| p[0] as f64 * 0.3 + p[1] as f64 * 0.59 + p[2] as f64 * 0.11).sum::<f64>() / (img.width() * img.height()) as f64
}

fn diff(a: &image::RgbaImage, b: &image::RgbaImage) -> f64 {
    a.pixels().zip(b.pixels()).map(|(p, q)| (0..3).map(|i| (p[i] as f64 - q[i] as f64).abs()).sum::<f64>()).sum::<f64>() / (a.width() * a.height()) as f64
}

#[test]
fn lighting_setups() {
    let Ok(r) = BevyRenderer::global() else { return };
    let s = scene();
    let out = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let mut shot = |name: &str, l: Lighting| {
        let img = r.render(&s, &opts(l)).unwrap().image;
        img.save(out.join(format!("fx_{name}.png"))).unwrap();
        img
    };
    let classic = shot("classic", Lighting::default());
    let studio = shot("studio", Lighting { environment: Some("studio".into()), tonemapping: Tonemap::Agx, ..Default::default() });
    let sky = shot(
        "sky_sun",
        Lighting {
            environment: Some("sky".into()),
            show_environment: true,
            sun: Some(Sun { direction: [0.4, 0.6, 0.3], intensity: 2.0, color: [1.0, 0.95, 0.85] }),
            shadows: true,
            tonemapping: Tonemap::Agx,
            ..Default::default()
        },
    );
    let no_shadow = shot(
        "sky_sun_noshadow",
        Lighting {
            environment: Some("sky".into()),
            show_environment: true,
            sun: Some(Sun { direction: [0.4, 0.6, 0.3], intensity: 2.0, color: [1.0, 0.95, 0.85] }),
            tonemapping: Tonemap::Agx,
            ..Default::default()
        },
    );
    let sun_only = |shadows| Lighting { sun: Some(Sun { direction: [0.4, 0.6, 0.3], intensity: 1.5, color: [1.0; 3] }), shadows, ..Default::default() };
    let hard = shot("sun_shadow", sun_only(true));
    let soft = shot("sun_noshadow", sun_only(false));
    eprintln!("shadow diffs: sky {} sun {}", diff(&sky, &no_shadow), diff(&hard, &soft));
    let ssao = shot("ssao", Lighting { environment: Some("studio".into()), ambient_occlusion: true, ..Default::default() });
    let studio_plain = shot("studio_plain", Lighting { environment: Some("studio".into()), ..Default::default() });
    let bloom = shot("bloom", Lighting { bloom: 0.3, tonemapping: Tonemap::Agx, ..Default::default() });
    let agx = shot("agx", Lighting { tonemapping: Tonemap::Agx, ..Default::default() });
    let bright = shot("exposure", Lighting { exposure: 1.0, ..Default::default() });
    // The background shows the sky only when asked: looking up, it is blue.
    let up = r
        .render(&s, &RenderOptions { view: View::Orbit { yaw_deg: 30.0, pitch_deg: -70.0, distance: None }, ..opts(Lighting { environment: Some("sky".into()), show_environment: true, ..Default::default() }) })
        .unwrap()
        .image;
    up.save(out.join("fx_sky_up.png")).unwrap();
    // Each setting visibly changes the image.
    assert!(diff(&classic, &studio) > 5.0, "environment lighting changes the image");
    assert!(diff(&hard, &soft) > 0.5 && luma(&hard) < luma(&soft), "shadows darken the ground: {}", diff(&hard, &soft));
    assert!(luma(&sky) < luma(&no_shadow));
    assert!(diff(&ssao, &studio_plain) > 0.2 && luma(&ssao) < luma(&studio_plain), "SSAO darkens creases");
    assert!(diff(&bloom, &agx) > 0.3, "bloom spreads glow");
    assert!(luma(&bright) > luma(&classic) + 5.0, "exposure +1 brightens: {} vs {}", luma(&bright), luma(&classic));
    let p = up.get_pixel(160, 5).0;
    assert!(p[2] > p[0] + 20, "sky background should be blue: {p:?}");
}
