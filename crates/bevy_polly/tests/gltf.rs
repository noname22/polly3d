//! A Polly GLB with a custom-shader material loads in Bevy as PollyMaterial.

use std::collections::BTreeMap;

use bevy::app::{PanicHandlerPlugin, TerminalCtrlCHandlerPlugin};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::ExitCondition;
use bevy::world_serialization::WorldAssetRoot;
use bevy_polly::{PollyMaterial, PollyMaterialsPlugin, shader_id};
use polly_core::scene::{CustomShader, Material, Scene};

#[test]
fn shader_materials_are_swapped_in() {
    let mut s = Scene::default();
    let params = BTreeMap::from([("tint".to_string(), vec![1.0, 0.5, 0.25, 1.0])]);
    let m = s.add_material(Material { shader: Some(CustomShader { name: "toon".into(), params }), ..Material::new("toonish", [1.0; 4]) });
    let mut mesh = polly_core::primitives::cube(polly_core::glam::Vec3::ONE);
    mesh.faces.values_mut().for_each(|f| f.material = m);
    s.add_object("Box", mesh);
    let dir = std::env::temp_dir().join(format!("bevy-polly-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("shaders")).unwrap();
    std::fs::write(dir.join("box.glb"), polly_export::export_glb(&s, &polly_export::ExportOptions::default()).unwrap()).unwrap();
    std::fs::write(dir.join("shaders/toon.wesl"), bevy_polly::shader_template("toon", &["tint".into()])).unwrap();

    let software = std::env::var("POLLY_ADAPTER").is_ok_and(|v| v == "software");
    let built = std::panic::catch_unwind(|| {
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin { primary_window: None, exit_condition: ExitCondition::DontExit, ..default() })
                .set(AssetPlugin { file_path: dir.to_string_lossy().into_owned(), ..default() })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings { force_fallback_adapter: software, ..default() })),
                    ..default()
                })
                .disable::<TerminalCtrlCHandlerPlugin>()
                .disable::<PanicHandlerPlugin>(),
        );
        app.add_plugins(PollyMaterialsPlugin::default());
        app.finish();
        app.cleanup();
        app
    });
    let Ok(mut app) = built else {
        eprintln!("skipping: no renderer");
        return;
    };
    let scene = app.world().resource::<AssetServer>().load(bevy::gltf::GltfAssetLabel::Scene(0).from_asset("box.glb"));
    app.world_mut().spawn(WorldAssetRoot(scene));
    let mut found = None;
    for _ in 0..300 {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let w = app.world_mut();
        let mut q = w.query::<&MeshMaterial3d<PollyMaterial>>();
        if let Some(h) = q.iter(w).next() {
            found = w.resource::<Assets<PollyMaterial>>().get(&h.0).cloned();
            break;
        }
    }
    let mat = found.expect("the glTF mesh should get a PollyMaterial");
    assert_eq!(mat.extension.shader, shader_id("toon"));
    assert_eq!(mat.extension.params.values[0], Vec4::new(1.0, 0.5, 0.25, 1.0));
    assert!(bevy_polly::registered_shader("toon").is_some());
    std::fs::remove_dir_all(dir).ok();
}
