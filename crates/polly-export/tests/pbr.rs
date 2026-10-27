//! PBR material export: factors, texture slots, KHR extensions and tangents.

use polly_core::glam::{Vec2, Vec3};
use polly_core::primitives::cube;
use polly_core::scene::{Anisotropy, Clearcoat, Material, Scene, Specular, Texture, Transmission};
use polly_export::{ExportOptions, export_glb};
use serde_json::Value;

fn json_chunk(glb: &[u8]) -> Value {
    let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&glb[20..20 + len]).unwrap()
}

/// A cube with per-face UVs (0,0)-(1,1); faces 0..3 use `a`, the rest `b`.
fn scene() -> Scene {
    let mut s = Scene::default();
    let base = s.add_texture(Texture::new("base", 4, 4, [200, 50, 50, 255]));
    let mut nt = Texture::new("normal", 4, 4, [128, 128, 255, 255]);
    nt.linear = true;
    let normal = s.add_texture(nt);
    let mut mr = Texture::new("orm", 4, 4, [255, 128, 0, 255]);
    mr.linear = true;
    let orm = s.add_texture(mr);
    let glow = s.add_texture(Texture::new("glow", 4, 4, [255, 255, 0, 255]));
    let a = s.add_material(Material {
        texture: Some(base),
        metallic: 0.25,
        roughness: 0.5,
        metallic_roughness_texture: Some(orm),
        occlusion_texture: Some(orm), // shared with metallic-roughness: one image
        occlusion_strength: 0.75,
        normal_texture: Some(normal),
        normal_scale: 0.5,
        emissive: [1.0, 0.5, 0.0],
        emissive_strength: 4.0,
        emissive_texture: Some(glow),
        blend: true,
        clearcoat: Some(Clearcoat { factor: 1.0, roughness: 0.1 }),
        anisotropy: Some(Anisotropy { strength: 0.5, rotation: 1.0 }),
        transmission: Some(Transmission { factor: 0.8, ior: 1.4, thickness: 0.05 }),
        specular: Some(Specular { factor: 0.9, color: [1.0, 0.8, 0.6] }),
        ..Material::new("full", [1.0; 4])
    });
    // A second material sharing the base texture, without a normal map.
    let b = s.add_material(Material { texture: Some(base), ..Material::new("plain", [1.0; 4]) });
    let mut mesh = cube(Vec3::ONE);
    let uvs = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0)];
    for (i, f) in mesh.faces.values_mut().enumerate() {
        f.material = if i < 3 { a } else { b };
        for (c, uv) in f.corners.iter_mut().zip(uvs) {
            c.uv = uv;
        }
    }
    s.add_object("Box", mesh);
    s
}

#[test]
fn pbr_material_fields_and_extensions() {
    let glb = export_glb(&scene(), &ExportOptions { objects: None }).unwrap();
    let j = json_chunk(&glb);
    let mats = j["materials"].as_array().unwrap();
    let full = mats.iter().find(|m| m["name"] == "full").unwrap();
    let pbr = &full["pbrMetallicRoughness"];
    assert_eq!(pbr["metallicFactor"], 0.25);
    assert_eq!(pbr["roughnessFactor"], 0.5);
    assert_eq!(full["normalTexture"]["scale"], 0.5);
    assert_eq!(full["occlusionTexture"]["strength"], 0.75);
    assert_eq!(full["emissiveFactor"], serde_json::json!([1.0, 0.5, 0.0]));
    assert_eq!(full["alphaMode"], "BLEND");
    let e = &full["extensions"];
    assert_eq!(e["KHR_materials_emissive_strength"]["emissiveStrength"], 4.0);
    assert_eq!(e["KHR_materials_clearcoat"]["clearcoatFactor"], 1.0);
    assert!((e["KHR_materials_clearcoat"]["clearcoatRoughnessFactor"].as_f64().unwrap() - 0.1).abs() < 1e-6);
    assert_eq!(e["KHR_materials_anisotropy"]["anisotropyStrength"], 0.5);
    assert!((e["KHR_materials_transmission"]["transmissionFactor"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    assert!((e["KHR_materials_ior"]["ior"].as_f64().unwrap() - 1.4).abs() < 1e-6);
    assert!((e["KHR_materials_volume"]["thicknessFactor"].as_f64().unwrap() - 0.05).abs() < 1e-6);
    assert!((e["KHR_materials_specular"]["specularFactor"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    let used: Vec<&str> = j["extensionsUsed"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    for x in [
        "KHR_materials_emissive_strength",
        "KHR_materials_clearcoat",
        "KHR_materials_anisotropy",
        "KHR_materials_transmission",
        "KHR_materials_ior",
        "KHR_materials_volume",
        "KHR_materials_specular",
    ] {
        assert!(used.contains(&x), "{x} missing from extensionsUsed");
    }
    assert!(j.get("extensionsRequired").is_none());

    // Texture indices: each slot points at its own image; shared textures appear once.
    let tex_name = |idx: &Value| {
        let t = &j["textures"][idx.as_u64().unwrap() as usize];
        j["images"][t["source"].as_u64().unwrap() as usize]["name"].as_str().unwrap().to_string()
    };
    assert_eq!(tex_name(&pbr["baseColorTexture"]["index"]), "base");
    assert_eq!(tex_name(&pbr["metallicRoughnessTexture"]["index"]), "orm");
    assert_eq!(tex_name(&full["occlusionTexture"]["index"]), "orm");
    assert_eq!(tex_name(&full["normalTexture"]["index"]), "normal");
    assert_eq!(tex_name(&full["emissiveTexture"]["index"]), "glow");
    assert_eq!(j["images"].as_array().unwrap().len(), 4, "base, normal, orm, glow once each");
    let plain = mats.iter().find(|m| m["name"] == "plain").unwrap();
    assert_eq!(plain["pbrMetallicRoughness"]["baseColorTexture"]["index"], pbr["baseColorTexture"]["index"]);
    assert!(plain.get("extensions").is_none());
    assert!(plain.get("emissiveFactor").is_none());
    assert_eq!(plain["pbrMetallicRoughness"]["roughnessFactor"], 1.0);
    // The gltf crate accepts the file.
    gltf::import_slice(&glb).unwrap();
}

#[test]
fn tangents_only_on_normal_mapped_primitives() {
    let glb = export_glb(&scene(), &ExportOptions { objects: None }).unwrap();
    let (doc, buffers, _) = gltf::import_slice(&glb).unwrap();
    let mut seen = (0, 0);
    for p in doc.meshes().flat_map(|m| m.primitives().collect::<Vec<_>>()) {
        let r = p.reader(|b| Some(&buffers[b.index()]));
        let normal_mapped = p.material().normal_texture().is_some();
        match r.read_tangents() {
            Some(t) => {
                assert!(normal_mapped);
                seen.0 += 1;
                let t: Vec<[f32; 4]> = t.collect();
                let n: Vec<[f32; 3]> = r.read_normals().unwrap().collect();
                assert_eq!(t.len(), n.len());
                for (t, n) in t.iter().zip(&n) {
                    let v = Vec3::new(t[0], t[1], t[2]);
                    assert!((v.length() - 1.0).abs() < 1e-3, "tangent not unit: {t:?}");
                    assert!(t[3] == 1.0 || t[3] == -1.0);
                    assert!(v.dot(Vec3::from_array(*n)).abs() < 1e-3, "tangent not perpendicular to normal");
                }
            }
            None => {
                assert!(!normal_mapped);
                seen.1 += 1;
            }
        }
    }
    assert_eq!(seen, (1, 1));
}
