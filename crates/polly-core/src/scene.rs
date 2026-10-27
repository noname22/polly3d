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

/// Largest texture side length (painting a big texture copies it into undo).
pub const MAX_TEXTURE_SIZE: u32 = 4096;

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
    /// Skinned to another object's skeleton (an armature) instead of its own:
    /// skin weights refer to the armature's bones, and the armature's pose and
    /// clips deform this mesh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skin: Option<SkinBinding>,
    /// Attached to this bone of the parent's skeleton: the transform is
    /// relative to the bone's frame and follows it when posed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_bone: Option<crate::anim::BoneId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkinBinding {
    pub armature: ObjectId,
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
    /// Holds linear data (normal, metallic-roughness, occlusion maps), not sRGB colour.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub linear: bool,
}

impl Texture {
    pub fn new(name: impl Into<String>, width: u32, height: u32, fill: [u8; 4]) -> Self {
        let pixels = fill.iter().copied().cycle().take((width * height * 4) as usize).collect();
        Texture { name: name.into(), width, height, pixels: Arc::new(pixels), filter: Filter::Nearest, palette: None, linear: false }
    }
    pub fn get(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        self.pixels[i..i + 4].try_into().unwrap()
    }
    pub fn pixels_mut(&mut self) -> &mut Vec<u8> {
        Arc::make_mut(&mut self.pixels)
    }
}

pub(crate) mod b64 {
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

/// Which material slot a texture is used in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Slot {
    /// Base colour (sRGB).
    Base,
    /// glTF metallic-roughness packing: G = roughness, B = metallic (linear).
    MetallicRoughness,
    /// Tangent-space normal map (linear).
    Normal,
    /// Ambient occlusion in R (linear).
    Occlusion,
    /// Emissive colour (sRGB).
    Emissive,
}

impl Slot {
    pub const ALL: [Slot; 5] = [Slot::Base, Slot::MetallicRoughness, Slot::Normal, Slot::Occlusion, Slot::Emissive];
    /// Data slots hold linear values, not colours.
    pub fn is_linear(self) -> bool {
        matches!(self, Slot::MetallicRoughness | Slot::Normal | Slot::Occlusion)
    }
    pub fn name(self) -> &'static str {
        match self {
            Slot::Base => "base",
            Slot::MetallicRoughness => "metallic_roughness",
            Slot::Normal => "normal",
            Slot::Occlusion => "occlusion",
            Slot::Emissive => "emissive",
        }
    }
    pub fn parse(s: &str) -> Option<Slot> {
        Slot::ALL.into_iter().find(|x| x.name() == s).or(match s {
            "base_color" | "color" | "albedo" => Some(Slot::Base),
            "metallic" | "roughness" | "mr" => Some(Slot::MetallicRoughness),
            "ao" => Some(Slot::Occlusion),
            _ => None,
        })
    }
}

/// `KHR_materials_clearcoat`: a thin clear layer (car paint, lacquer, wet skin).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Clearcoat {
    #[serde(default = "one")]
    pub factor: f32,
    #[serde(default)]
    pub roughness: f32,
}

/// `KHR_materials_anisotropy`: stretched highlights (brushed metal, hair).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Anisotropy {
    #[serde(default = "one")]
    pub strength: f32,
    /// Radians, counter-clockwise from the tangent direction.
    #[serde(default)]
    pub rotation: f32,
}

/// `KHR_materials_transmission` (+ `_ior`, `_volume` thickness): glass, thin plastic.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Transmission {
    #[serde(default = "one")]
    pub factor: f32,
    #[serde(default = "default_ior")]
    pub ior: f32,
    #[serde(default)]
    pub thickness: f32,
}

/// `KHR_materials_specular`: specular strength and tint for dielectrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Specular {
    #[serde(default = "one")]
    pub factor: f32,
    /// Linear RGB tint.
    #[serde(default = "white3")]
    pub color: [f32; 3],
}

fn one() -> f32 {
    1.0
}
fn default_ior() -> f32 {
    1.5
}
fn white3() -> [f32; 3] {
    [1.0; 3]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    /// Base colour map (sRGB).
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
    #[serde(default)]
    pub metallic: f32,
    #[serde(default = "one")]
    pub roughness: f32,
    /// glTF packing: G = roughness, B = metallic (multiplied with the factors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic_roughness_texture: Option<TextureId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_texture: Option<TextureId>,
    #[serde(default = "one")]
    pub normal_scale: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occlusion_texture: Option<TextureId>,
    #[serde(default = "one")]
    pub occlusion_strength: f32,
    /// Linear RGB.
    #[serde(default)]
    pub emissive: [f32; 3],
    #[serde(default = "one")]
    pub emissive_strength: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive_texture: Option<TextureId>,
    /// Alpha blending (exclusive with `alpha_cutoff`).
    #[serde(default)]
    pub blend: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clearcoat: Option<Clearcoat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anisotropy: Option<Anisotropy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transmission: Option<Transmission>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub specular: Option<Specular>,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            name: String::new(),
            base_color: [1.0; 4],
            texture: None,
            unlit: false,
            alpha_cutoff: None,
            double_sided: false,
            metallic: 0.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_strength: 1.0,
            emissive_texture: None,
            blend: false,
            clearcoat: None,
            anisotropy: None,
            transmission: None,
            specular: None,
        }
    }
}

impl Material {
    pub fn new(name: impl Into<String>, base_color: [f32; 4]) -> Self {
        Material { name: name.into(), base_color, ..Default::default() }
    }

    /// Every texture the material uses, with its slot.
    pub fn textures(&self) -> impl Iterator<Item = (Slot, TextureId)> + '_ {
        Slot::ALL.into_iter().filter_map(|s| self.slot(s).map(|t| (s, t)))
    }

    pub fn slot(&self, s: Slot) -> Option<TextureId> {
        match s {
            Slot::Base => self.texture,
            Slot::MetallicRoughness => self.metallic_roughness_texture,
            Slot::Normal => self.normal_texture,
            Slot::Occlusion => self.occlusion_texture,
            Slot::Emissive => self.emissive_texture,
        }
    }

    pub fn slot_mut(&mut self, s: Slot) -> &mut Option<TextureId> {
        match s {
            Slot::Base => &mut self.texture,
            Slot::MetallicRoughness => &mut self.metallic_roughness_texture,
            Slot::Normal => &mut self.normal_texture,
            Slot::Occlusion => &mut self.occlusion_texture,
            Slot::Emissive => &mut self.emissive_texture,
        }
    }

    /// Emissive colour times strength is non-zero.
    pub fn is_emissive(&self) -> bool {
        self.emissive_strength > 0.0 && self.emissive.iter().any(|c| *c > 0.0)
    }

    /// Uses channels beyond base colour, emissive and alpha (PBR shading data).
    pub fn has_pbr_channels(&self) -> bool {
        let d = Material::default();
        self.metallic != d.metallic
            || self.roughness != d.roughness
            || self.metallic_roughness_texture.is_some()
            || self.normal_texture.is_some()
            || self.occlusion_texture.is_some()
            || self.blend
            || self.clearcoat.is_some()
            || self.anisotropy.is_some()
            || self.transmission.is_some()
            || self.specular.is_some()
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
    /// Swappable gear: slot name -> variants (see `variants`).
    #[serde(default)]
    pub slots: BTreeMap<String, crate::variants::Slot>,
    /// Reference images shown in the axis views.
    #[serde(default)]
    pub references: Vec<crate::reference::Reference>,
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
            slots: BTreeMap::new(),
            references: Vec::new(),
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
                skin: None,
                parent_bone: None,
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
                other.parent_bone = None;
            }
            if other.skin.is_some_and(|s| s.armature == id) {
                other.skin = None;
            }
        }
        if self.active == Some(id) {
            self.active = self.objects.keys().next().copied();
        }
        self.drop_from_slots(id);
        Ok(o)
    }

    /// The object whose skeleton deforms `id`: its armature if bound, itself
    /// if it has its own skeleton.
    pub fn skeleton_source(&self, id: ObjectId) -> Option<ObjectId> {
        let o = self.objects.get(&id)?;
        match o.skin {
            Some(b) => self.objects.get(&b.armature).filter(|a| a.skeleton.is_some()).map(|_| b.armature),
            None => o.skeleton.as_ref().map(|_| id),
        }
    }

    /// Transform from `id`'s local space into the space of its skeleton source.
    pub fn to_skeleton_space(&self, id: ObjectId) -> Mat4 {
        match self.skeleton_source(id) {
            Some(src) if src != id => self.world_matrix(src).inverse() * self.world_matrix(id),
            _ => Mat4::IDENTITY,
        }
    }

    /// Rest frame of the bone `id` is attached to, in its parent's space
    /// (identity when not attached to a bone).
    pub fn bone_frame(&self, id: ObjectId) -> Mat4 {
        let Some(o) = self.objects.get(&id) else { return Mat4::IDENTITY };
        o.parent_bone
            .and_then(|b| self.objects.get(&o.parent?)?.skeleton.as_ref()?.bones.get(&b))
            .map_or(Mat4::IDENTITY, |b| b.rest_matrix())
    }

    pub fn world_matrix(&self, id: ObjectId) -> Mat4 {
        let mut m = Mat4::IDENTITY;
        let mut cur = Some(id);
        let mut guard = 0;
        while let Some(c) = cur {
            let Some(o) = self.objects.get(&c) else { break };
            m = self.bone_frame(c) * o.transform.matrix() * m;
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
        let o = self.object_mut(id)?;
        if o.parent != parent {
            o.parent_bone = None;
        }
        o.parent = parent;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_material_round_trips() {
        let m = Material {
            name: "paint".into(),
            base_color: [0.5, 0.2, 0.1, 1.0],
            texture: Some(1),
            metallic: 0.3,
            roughness: 0.4,
            metallic_roughness_texture: Some(2),
            normal_texture: Some(3),
            normal_scale: 0.7,
            occlusion_texture: Some(2),
            occlusion_strength: 0.5,
            emissive: [1.0, 0.5, 0.0],
            emissive_strength: 3.0,
            emissive_texture: Some(4),
            blend: true,
            clearcoat: Some(Clearcoat { factor: 1.0, roughness: 0.1 }),
            anisotropy: Some(Anisotropy { strength: 0.6, rotation: 0.5 }),
            transmission: Some(Transmission { factor: 0.9, ior: 1.45, thickness: 0.02 }),
            specular: Some(Specular { factor: 0.8, color: [1.0, 0.9, 0.8] }),
            ..Material::new("x", [1.0; 4])
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: Material = serde_json::from_str(&j).unwrap();
        assert_eq!(m, back);
        let slots: Vec<_> = back.textures().collect();
        assert_eq!(slots.len(), 5);
        assert!(back.has_pbr_channels());
    }

    #[test]
    fn old_material_json_gets_defaults() {
        let old = r#"{"name":"m","base_color":[1,0,0,1],"texture":2,"unlit":false,"alpha_cutoff":null,"double_sided":true}"#;
        let m: Material = serde_json::from_str(old).unwrap();
        assert_eq!(m, Material { texture: Some(2), double_sided: true, ..Material::new("m", [1.0, 0.0, 0.0, 1.0]) });
        assert_eq!((m.metallic, m.roughness, m.normal_scale, m.occlusion_strength, m.emissive_strength), (0.0, 1.0, 1.0, 1.0, 1.0));
        assert!(!m.has_pbr_channels());
        let t: Texture = serde_json::from_str(r#"{"name":"t","width":1,"height":1,"pixels":"AAAAAA=="}"#).unwrap();
        assert!(!t.linear);
    }
}
