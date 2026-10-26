pub mod glb;

use anyhow::{Context, Result, bail};
use polly_core::scene::{Scene, Texture};
use serde_json::json;

pub use glb::{ExportOptions, export_glb};

const FORMAT: &str = "polly3d";
const VERSION: u32 = 1;

/// Serialize a scene as a `.polly` project (JSON).
pub fn save_project(scene: &Scene) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&json!({"format": FORMAT, "version": VERSION, "scene": scene}))?)
}

pub fn load_project(bytes: &[u8]) -> Result<Scene> {
    let v: serde_json::Value = serde_json::from_slice(bytes).context("not a JSON file")?;
    if v["format"] != FORMAT {
        bail!("not a polly3d project file");
    }
    if v["version"].as_u64().unwrap_or(0) > VERSION as u64 {
        bail!("project was written by a newer polly3d (version {})", v["version"]);
    }
    serde_json::from_value(v["scene"].clone()).context("corrupt project file")
}

pub fn png(t: &Texture) -> Result<Vec<u8>> {
    let img = image::RgbaImage::from_raw(t.width, t.height, t.pixels.to_vec()).context("bad texture size")?;
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polly_core::glam::Vec3;
    use polly_core::primitives::cube;
    use polly_core::scene::Material;

    fn sample() -> Scene {
        let mut s = Scene::default();
        let tid = s.add_texture(Texture::new("tex", 8, 8, [200, 50, 50, 255]));
        let mut m = Material::new("red", [1.0; 4]);
        m.texture = Some(tid);
        let mid = s.add_material(m);
        let mut mesh = cube(Vec3::ONE);
        mesh.faces.values_mut().next().unwrap().material = mid;
        let a = s.add_object("Box", mesh);
        let b = s.add_object("Child", cube(Vec3::splat(0.5)));
        s.set_parent(b, Some(a)).unwrap();
        s
    }

    #[test]
    fn project_roundtrip() {
        let s = sample();
        let back = load_project(&save_project(&s).unwrap()).unwrap();
        assert_eq!(back.objects.len(), 2);
        assert_eq!(back.textures[&1].pixels, s.textures[&1].pixels);
        assert_eq!(back.objects[&1].mesh.faces.len(), 6);
    }

    #[test]
    fn glb_loads_with_gltf_crate() {
        let s = sample();
        let bytes = export_glb(&s, &ExportOptions { objects: None }).unwrap();
        let (doc, buffers, images) = gltf::import_slice(&bytes).unwrap();
        assert_eq!(doc.nodes().count(), 2);
        assert_eq!(doc.scenes().next().unwrap().nodes().count(), 1);
        assert_eq!(images.len(), 1);
        let sampler = doc.textures().next().unwrap().sampler();
        assert_eq!(sampler.mag_filter(), Some(gltf::texture::MagFilter::Nearest));
        let mesh = doc.meshes().next().unwrap();
        assert_eq!(mesh.primitives().count(), 2);
        let tris: usize = mesh
            .primitives()
            .map(|p| p.reader(|b| Some(&buffers[b.index()])).read_indices().unwrap().into_u32().count() / 3)
            .sum();
        assert_eq!(tris, 12);
    }
}
