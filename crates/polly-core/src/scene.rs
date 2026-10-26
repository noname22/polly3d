use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result, invalid};
use crate::ids::Selection;
use crate::mesh::Mesh;

pub type ObjectId = u32;
pub type MaterialId = u32;
pub type TextureId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Transform { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
}

impl Transform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
    pub fn euler_degrees(&self) -> [f32; 3] {
        let (x, y, z) = self.rotation.to_euler(glam::EulerRot::XYZ);
        [x.to_degrees(), y.to_degrees(), z.to_degrees()]
    }
}

pub fn quat_from_euler_degrees(e: [f32; 3]) -> Quat {
    Quat::from_euler(glam::EulerRot::XYZ, e[0].to_radians(), e[1].to_radians(), e[2].to_radians())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    pub name: String,
    pub mesh: Arc<Mesh>,
    #[serde(default)]
    pub transform: Transform,
    #[serde(default)]
    pub parent: Option<ObjectId>,
    #[serde(default = "yes")]
    pub visible: bool,
    /// The agent's current element selection on this object.
    #[serde(default)]
    pub selection: Selection,
    #[serde(default)]
    pub skeleton: Option<crate::anim::Skeleton>,
    /// Static bone pose (what posing tools set; clips blend on top).
    #[serde(default)]
    pub pose: BTreeMap<crate::anim::BoneId, crate::anim::BonePose>,
    /// Static morph weights by morph name.
    #[serde(default)]
    pub morph_weights: BTreeMap<String, f32>,
    /// A morph target being edited, if any.
    #[serde(default)]
    pub morph_edit: Option<crate::anim::MorphEdit>,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    #[default]
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Texture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, top row first.
    #[serde(with = "b64")]
    pub pixels: Arc<Vec<u8>>,
    #[serde(default)]
    pub filter: Filter,
    /// Name of the palette painting is locked to, if any.
    #[serde(default)]
    pub palette: Option<String>,
}

impl Texture {
    pub fn new(name: impl Into<String>, width: u32, height: u32, fill: [u8; 4]) -> Self {
        let pixels = fill.iter().copied().cycle().take((width * height * 4) as usize).collect();
        Texture { name: name.into(), width, height, pixels: Arc::new(pixels), filter: Filter::Nearest, palette: None }
    }
    pub fn get(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        self.pixels[i..i + 4].try_into().unwrap()
    }
    pub fn pixels_mut(&mut self) -> &mut Vec<u8> {
        Arc::make_mut(&mut self.pixels)
    }
}

mod b64 {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::sync::Arc;
    pub fn serialize<S: Serializer>(v: &Arc<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(v.as_slice()))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Arc<Vec<u8>>, D::Error> {
        let s = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD.decode(s).map(Arc::new).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    #[serde(default)]
    pub texture: Option<TextureId>,
    /// Ignore lighting (flat, fullbright).
    #[serde(default)]
    pub unlit: bool,
    /// Pixels with alpha below this are discarded (cutout transparency).
    #[serde(default)]
    pub alpha_cutoff: Option<f32>,
    #[serde(default)]
    pub double_sided: bool,
}

impl Material {
    pub fn new(name: impl Into<String>, base_color: [f32; 4]) -> Self {
        Material { name: name.into(), base_color, texture: None, unlit: false, alpha_cutoff: None, double_sided: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub objects: BTreeMap<ObjectId, Object>,
    pub materials: BTreeMap<MaterialId, Material>,
    pub textures: BTreeMap<TextureId, Texture>,
    #[serde(default)]
    pub palettes: BTreeMap<String, Vec<[u8; 4]>>,
    #[serde(default)]
    pub active: Option<ObjectId>,
    #[serde(default = "default_bg")]
    pub background: [f32; 4],
    #[serde(default)]
    pub clips: BTreeMap<crate::anim::ClipId, crate::anim::Clip>,
    #[serde(default)]
    next_clip: crate::anim::ClipId,
    next_object: ObjectId,
    next_material: MaterialId,
    next_texture: TextureId,
}

fn default_bg() -> [f32; 4] {
    [0.16, 0.17, 0.2, 1.0]
}

impl Default for Scene {
    fn default() -> Self {
        let mut materials = BTreeMap::new();
        materials.insert(0, Material::new("default", [0.8, 0.8, 0.8, 1.0]));
        Scene {
            objects: BTreeMap::new(),
            materials,
            textures: BTreeMap::new(),
            palettes: BTreeMap::new(),
            active: None,
            background: default_bg(),
            clips: BTreeMap::new(),
            next_clip: 1,
            next_object: 1,
            next_material: 1,
            next_texture: 1,
        }
    }
}

impl Scene {
    fn unique_name(&self, base: &str) -> String {
        if !self.objects.values().any(|o| o.name == base) {
            return base.to_string();
        }
        (1..).map(|i| format!("{base}.{i:03}")).find(|n| !self.objects.values().any(|o| &o.name == n)).unwrap()
    }

    pub fn add_object(&mut self, name: &str, mesh: Mesh) -> ObjectId {
        let id = self.next_object;
        self.next_object += 1;
        let name = self.unique_name(name);
        self.objects.insert(
            id,
            Object {
                name,
                mesh: Arc::new(mesh),
                transform: Transform::default(),
                parent: None,
                visible: true,
                selection: Selection::default(),
                skeleton: None,
                pose: BTreeMap::new(),
                morph_weights: BTreeMap::new(),
                morph_edit: None,
            },
        );
        self.active = Some(id);
        id
    }

    pub fn rename_object(&mut self, id: ObjectId, name: &str) -> Result<String> {
        if self.objects.iter().any(|(i, o)| *i != id && o.name == name) {
            return invalid(format!("an object named '{name}' already exists"));
        }
        self.object_mut(id)?.name = name.to_string();
        Ok(name.to_string())
    }

    /// Resolve an object by name or `o<id>`; `None` / empty means the active object.
    pub fn resolve(&self, name: Option<&str>) -> Result<ObjectId> {
        match name.map(str::trim).filter(|s| !s.is_empty()) {
            None => self.active.filter(|a| self.objects.contains_key(a)).ok_or_else(|| {
                Error::Invalid("no active object; pass an object name (see list_objects)".into())
            }),
            Some(n) => {
                if let Some((id, _)) = self.objects.iter().find(|(_, o)| o.name == n) {
                    return Ok(*id);
                }
                if let Some(id) = n.strip_prefix('o').and_then(|s| s.parse::<u32>().ok()) {
                    if self.objects.contains_key(&id) {
                        return Ok(id);
                    }
                }
                Err(Error::NotFound { kind: "object", id: n.to_string() })
            }
        }
    }

    pub fn object(&self, id: ObjectId) -> Result<&Object> {
        self.objects.get(&id).ok_or_else(|| Error::NotFound { kind: "object", id: format!("o{id}") })
    }

    pub fn object_mut(&mut self, id: ObjectId) -> Result<&mut Object> {
        self.objects.get_mut(&id).ok_or_else(|| Error::NotFound { kind: "object", id: format!("o{id}") })
    }

    pub fn remove_object(&mut self, id: ObjectId) -> Result<Object> {
        let o = self.objects.remove(&id).ok_or_else(|| Error::NotFound { kind: "object", id: format!("o{id}") })?;
        for other in self.objects.values_mut() {
            if other.parent == Some(id) {
                other.parent = o.parent;
            }
        }
        if self.active == Some(id) {
            self.active = self.objects.keys().next().copied();
        }
        Ok(o)
    }

    pub fn world_matrix(&self, id: ObjectId) -> Mat4 {
        let mut m = Mat4::IDENTITY;
        let mut cur = Some(id);
        let mut guard = 0;
        while let Some(c) = cur {
            let Some(o) = self.objects.get(&c) else { break };
            m = o.transform.matrix() * m;
            cur = o.parent;
            guard += 1;
            if guard > 64 {
                break;
            }
        }
        m
    }

    pub fn set_parent(&mut self, id: ObjectId, parent: Option<ObjectId>) -> Result<()> {
        if let Some(p) = parent {
            self.object(p)?;
            let mut cur = Some(p);
            while let Some(c) = cur {
                if c == id {
                    return invalid("parenting would create a cycle");
                }
                cur = self.objects.get(&c).and_then(|o| o.parent);
            }
        }
        self.object_mut(id)?.parent = parent;
        Ok(())
    }

    pub fn add_material(&mut self, m: Material) -> MaterialId {
        let id = self.next_material;
        self.next_material += 1;
        self.materials.insert(id, m);
        id
    }

    pub fn resolve_material(&self, name: &str) -> Result<MaterialId> {
        if let Some((id, _)) = self.materials.iter().find(|(_, m)| m.name == name) {
            return Ok(*id);
        }
        if let Some(id) = name.strip_prefix('m').and_then(|s| s.parse::<u32>().ok()).or_else(|| name.parse().ok()) {
            if self.materials.contains_key(&id) {
                return Ok(id);
            }
        }
        Err(Error::NotFound { kind: "material", id: name.to_string() })
    }

    pub fn add_clip(&mut self, c: crate::anim::Clip) -> crate::anim::ClipId {
        let id = self.next_clip.max(1);
        self.next_clip = id + 1;
        self.clips.insert(id, c);
        id
    }

    pub fn resolve_clip(&self, name: &str) -> Result<crate::anim::ClipId> {
        self.clips
            .iter()
            .find(|(_, c)| c.name == name)
            .map(|(id, _)| *id)
            .ok_or_else(|| Error::NotFound { kind: "clip", id: name.to_string() })
    }

    pub fn add_texture(&mut self, t: Texture) -> TextureId {
        let id = self.next_texture;
        self.next_texture += 1;
        self.textures.insert(id, t);
        id
    }

    pub fn resolve_texture(&self, name: &str) -> Result<TextureId> {
        if let Some((id, _)) = self.textures.iter().find(|(_, t)| t.name == name) {
            return Ok(*id);
        }
        if let Some(id) = name.strip_prefix('t').and_then(|s| s.parse::<u32>().ok()).or_else(|| name.parse().ok()) {
            if self.textures.contains_key(&id) {
                return Ok(id);
            }
        }
        Err(Error::NotFound { kind: "texture", id: name.to_string() })
    }

    pub fn texture_mut(&mut self, id: TextureId) -> Result<&mut Texture> {
        self.textures.get_mut(&id).ok_or_else(|| Error::NotFound { kind: "texture", id: format!("t{id}") })
    }

    /// World-space bounds over visible objects.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut out: Option<(Vec3, Vec3)> = None;
        for (&id, o) in &self.objects {
            if !o.visible {
                continue;
            }
            let m = self.world_matrix(id);
            for v in o.mesh.verts.values() {
                let p = m.transform_point3(v.pos);
                out = Some(match out {
                    None => (p, p),
                    Some((lo, hi)) => (lo.min(p), hi.max(p)),
                });
            }
        }
        out
    }
}
