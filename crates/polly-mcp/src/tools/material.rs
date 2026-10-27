//! Materials: glTF PBR fields (base colour, metallic/roughness, normal,
//! occlusion and emissive maps, alpha modes) plus optional KHR extensions.

use anyhow::{Result, bail};
use polly_core::scene::{Anisotropy, Clearcoat, CustomShader, Material, Scene, Slot, Specular, TextureId, Transmission};

pub fn valid_shader_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 64 && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use super::common::*;
use super::scene::{polly_hex, to_f32};
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "create_material",
            "Create a material. Faces use material 0 (\"default\") until assigned. Retro: color + texture (+ unlit). Modern PBR: metallic/roughness, normal/occlusion/emissive maps, blend, clearcoat, anisotropy, transmission, specular. All are exported to glTF.",
            false,
            create_material,
        ),
        def(
            "update_material",
            "Change properties of an existing material. Only the given fields change; null or \"\" removes a texture, colour or extension.",
            false,
            update_material,
        ),
        def("assign_material", "Assign a material to faces of an object.", false, assign_material),
    ]
}

/// PBR fields shared by create_material and update_material.
#[derive(Deserialize, JsonSchema, Default)]
pub struct PbrArgs {
    /// 0 = dielectric (default), 1 = metal.
    metallic: Option<f32>,
    /// 0 = mirror-smooth, 1 = fully rough (default).
    roughness: Option<f32>,
    /// Texture with roughness in G and metallic in B (glTF packing; multiplied by the factors). Treated as linear data. "" removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<String>")]
    metallic_roughness_texture: Option<Option<String>>,
    /// Tangent-space normal map (create_texture kind "normal"). Treated as linear data; export adds MikkTSpace tangents. "" removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<String>")]
    normal_texture: Option<Option<String>>,
    /// Normal map strength (default 1).
    normal_scale: Option<f32>,
    /// Ambient occlusion map (R channel, linear data). May be the same texture as metallic_roughness_texture (ORM packing). "" removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<String>")]
    occlusion_texture: Option<Option<String>>,
    /// Occlusion strength (default 1).
    occlusion_strength: Option<f32>,
    /// Emissive (glow) colour; alpha is ignored. "" or black removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<[u8; 4]>")]
    emissive: Option<Option<[u8; 4]>>,
    /// Emissive multiplier (default 1; >1 for bright lights, exported as KHR_materials_emissive_strength).
    emissive_strength: Option<f32>,
    /// Emissive colour map (sRGB), multiplied by `emissive` (white when unset). "" removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<String>")]
    emissive_texture: Option<Option<String>>,
    /// Alpha blending (glass, smoke, soft hair tips). Can't be combined with alpha_cutoff.
    blend: Option<bool>,
    /// Clear lacquer layer {factor, roughness} (car paint, varnish). null removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<Clearcoat>")]
    clearcoat: Option<Option<Clearcoat>>,
    /// Stretched highlights {strength, rotation (radians)} (brushed metal, hair). null removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<Anisotropy>")]
    anisotropy: Option<Option<Anisotropy>>,
    /// See-through surface {factor, ior (default 1.5), thickness} (glass, thin plastic). null removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<Transmission>")]
    transmission: Option<Option<Transmission>>,
    /// Specular strength/tint for non-metals {factor, color [r,g,b] linear 0..1}. null removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<Specular>")]
    specular: Option<Option<Specular>>,
    /// Custom shader {name, params: {name: number or [up to 4 numbers]}} from <asset_dir>/shaders/<name>.wesl (see write_shader). null removes.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<ShaderArg>")]
    shader: Option<Option<ShaderArg>>,
}

#[derive(Deserialize, JsonSchema, Clone)]
pub struct ShaderArg {
    name: String,
    /// Parameter values: a number or an array of up to 4 numbers each (packed into vec4 slots alphabetically).
    #[serde(default)]
    params: std::collections::BTreeMap<String, ParamValue>,
}

#[derive(Deserialize, JsonSchema, Clone)]
#[serde(untagged)]
pub enum ParamValue {
    One(f32),
    Many(Vec<f32>),
}

/// Resolve a texture argument; `Some(None)` = remove.
fn tex_arg(s: &Scene, t: &Option<Option<String>>) -> Result<Option<Option<TextureId>>> {
    Ok(match t {
        None => None,
        Some(None) => Some(None),
        Some(Some(n)) => Some(Some(s.resolve_texture(n)?)),
    })
}

impl PbrArgs {
    /// Apply to `m`; returns notes. Validates first so nothing changes on error.
    fn apply(&self, s: &mut Scene, mid: Option<u32>, m: &mut Material) -> Result<Vec<String>> {
        let slots = [
            (Slot::MetallicRoughness, tex_arg(s, &self.metallic_roughness_texture)?),
            (Slot::Normal, tex_arg(s, &self.normal_texture)?),
            (Slot::Occlusion, tex_arg(s, &self.occlusion_texture)?),
            (Slot::Emissive, tex_arg(s, &self.emissive_texture)?),
        ];
        for (name, v) in [("metallic", self.metallic), ("roughness", self.roughness), ("occlusion_strength", self.occlusion_strength)] {
            if v.is_some_and(|v| !(0.0..=1.0).contains(&v)) {
                bail!("{name} must be in 0..1");
            }
        }
        if self.emissive_strength.is_some_and(|v| v < 0.0) {
            bail!("emissive_strength must be >= 0");
        }
        let mut notes = Vec::new();
        if let Some(v) = self.metallic {
            m.metallic = v;
        }
        if let Some(v) = self.roughness {
            m.roughness = v;
        }
        if let Some(v) = self.normal_scale {
            m.normal_scale = v;
        }
        if let Some(v) = self.occlusion_strength {
            m.occlusion_strength = v;
        }
        if let Some(e) = self.emissive {
            m.emissive = e.map_or([0.0; 3], |c| {
                let l = to_f32(c);
                [l[0], l[1], l[2]]
            });
        }
        if let Some(v) = self.emissive_strength {
            m.emissive_strength = v;
        }
        if let Some(b) = self.blend {
            m.blend = b;
        }
        if let Some(c) = self.clearcoat {
            m.clearcoat = c;
        }
        if let Some(a) = self.anisotropy {
            m.anisotropy = a;
        }
        if let Some(t) = self.transmission {
            m.transmission = t;
        }
        if let Some(sp) = self.specular {
            m.specular = sp;
        }
        if let Some(sh) = &self.shader {
            m.shader = match sh {
                None => None,
                Some(a) => {
                    if !valid_shader_name(&a.name) {
                        bail!("shader name '{}' must be letters, digits, '_' or '-'", a.name);
                    }
                    if a.params.len() > bevy_polly::MAX_PARAMS {
                        bail!("at most {} shader params", bevy_polly::MAX_PARAMS);
                    }
                    let mut params = std::collections::BTreeMap::new();
                    for (k, v) in &a.params {
                        let v = match v {
                            ParamValue::One(x) => vec![*x],
                            ParamValue::Many(xs) if (1..=4).contains(&xs.len()) => xs.clone(),
                            ParamValue::Many(_) => bail!("shader param '{k}' needs 1 to 4 numbers"),
                        };
                        params.insert(k.clone(), v);
                    }
                    if s.asset_dir.is_none() {
                        notes.push("the scene has no asset directory yet: write_shader sets it (renders show the plain material until the shader file exists)".into());
                    }
                    Some(CustomShader { name: a.name.clone(), params })
                }
            };
        }
        for (slot, v) in slots {
            let Some(v) = v else { continue };
            *m.slot_mut(slot) = v;
            if let Some(tid) = v {
                notes.extend(mark_texture_kind(s, mid, tid, slot));
            }
        }
        // An emissive map with a black factor would never show.
        if self.emissive_texture.as_ref().is_some_and(|t| t.is_some()) && self.emissive.is_none() && m.emissive == [0.0; 3] {
            m.emissive = [1.0; 3];
        }
        Ok(notes)
    }
}

/// Using a texture in a data slot makes it linear (and a colour slot sRGB);
/// returns a note when that changes how it was treated before.
pub fn mark_texture_kind(s: &mut Scene, mid: Option<u32>, tid: TextureId, slot: Slot) -> Option<String> {
    let linear = slot.is_linear();
    // Other materials (or other slots) using it the other way.
    let conflict = s.materials.iter().any(|(id, m)| {
        m.textures().any(|(sl, t)| t == tid && sl.is_linear() != linear && !(Some(*id) == mid && sl == slot))
    });
    let t = s.textures.get_mut(&tid)?;
    if conflict {
        return Some(format!(
            "texture '{}' is also used as {} elsewhere; one texture can't be both colour (sRGB) and data (linear). It stays {}",
            t.name,
            if linear { "colour" } else { "data" },
            if t.linear { "linear" } else { "sRGB" }
        ));
    }
    if t.linear == linear {
        return None;
    }
    t.linear = linear;
    Some(if linear {
        format!("texture '{}' is now linear data (used as {}); if it was painted as colour, its values now mean data, not colours", t.name, slot.name())
    } else {
        format!("texture '{}' is now an sRGB colour texture (used as {})", t.name, slot.name())
    })
}

fn check_alpha(m: &Material) -> Result<()> {
    if m.blend && m.alpha_cutoff.is_some() {
        bail!("blend and alpha_cutoff are exclusive: pick blending (soft transparency) or a cutoff (hard cutout); set alpha_cutoff to -1 or blend to false");
    }
    Ok(())
}

#[derive(Deserialize, JsonSchema)]
struct CreateMaterial {
    name: String,
    /// Base color (default white, so textures show unmodified).
    color: Option<[u8; 4]>,
    /// Texture name or id to use as base color map.
    texture: Option<String>,
    /// Ignore lighting (flat, fullbright).
    #[serde(default)]
    unlit: bool,
    /// Discard pixels with alpha below this (cutout transparency, e.g. 0.5 for foliage).
    alpha_cutoff: Option<f32>,
    #[serde(default)]
    double_sided: bool,
    #[serde(flatten)]
    pbr: PbrArgs,
}

fn create_material(ctx: &mut Ctx, a: CreateMaterial) -> Result<Out> {
    let color = a.color.map(to_f32).unwrap_or([1.0; 4]);
    let tex = a.texture.as_deref().filter(|t| !t.is_empty()).map(|t| ctx.doc.scene.resolve_texture(t)).transpose()?;
    if ctx.doc.scene.materials.values().any(|m| m.name == a.name) {
        bail!("material '{}' already exists (use update_material)", a.name);
    }
    let (id, notes) = edit_scene(ctx, "create material", |s| {
        let mut m = Material::new(&a.name, color);
        m.unlit = a.unlit;
        m.alpha_cutoff = a.alpha_cutoff.filter(|c| *c >= 0.0);
        m.double_sided = a.double_sided;
        let mut notes = a.pbr.apply(s, None, &mut m).map_err(|e| polly_core::Error::Invalid(e.to_string()))?;
        check_alpha(&m).map_err(|e| polly_core::Error::Invalid(e.to_string()))?;
        m.texture = tex;
        let id = s.add_material(m);
        if let Some(t) = tex {
            notes.extend(mark_texture_kind(s, Some(id), t, Slot::Base));
        }
        Ok((id, notes))
    })?;
    let mut j = json!({"material": a.name, "index": id});
    if !notes.is_empty() {
        j["notes"] = json!(notes);
    }
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct UpdateMaterial {
    /// Material name or index.
    material: String,
    name: Option<String>,
    color: Option<[u8; 4]>,
    /// Base colour texture name/id; empty string removes the texture.
    texture: Option<String>,
    unlit: Option<bool>,
    /// Negative value removes cutout.
    alpha_cutoff: Option<f32>,
    double_sided: Option<bool>,
    #[serde(flatten)]
    pbr: PbrArgs,
}

fn update_material(ctx: &mut Ctx, a: UpdateMaterial) -> Result<Out> {
    let id = ctx.doc.scene.resolve_material(&a.material)?;
    let color = a.color.map(to_f32);
    let tex = match a.texture.as_deref() {
        None => None,
        Some("") => Some(None),
        Some(t) => Some(Some(ctx.doc.scene.resolve_texture(t)?)),
    };
    if let Some(n) = &a.name {
        if ctx.doc.scene.materials.iter().any(|(i, m)| *i != id && m.name == *n) {
            bail!("a material named '{n}' already exists");
        }
    }
    let notes = edit_scene(ctx, "update material", |s| {
        let mut m = s.materials[&id].clone();
        if let Some(n) = &a.name {
            m.name = n.clone();
        }
        if let Some(c) = color {
            m.base_color = c;
        }
        if let Some(u) = a.unlit {
            m.unlit = u;
        }
        if let Some(c) = a.alpha_cutoff {
            m.alpha_cutoff = (c >= 0.0).then_some(c);
        }
        if let Some(d) = a.double_sided {
            m.double_sided = d;
        }
        let mut notes = a.pbr.apply(s, Some(id), &mut m).map_err(|e| polly_core::Error::Invalid(e.to_string()))?;
        check_alpha(&m).map_err(|e| polly_core::Error::Invalid(e.to_string()))?;
        if let Some(t) = tex {
            m.texture = t;
            if let Some(t) = t {
                s.materials.insert(id, m.clone());
                notes.extend(mark_texture_kind(s, Some(id), t, Slot::Base));
            }
        }
        s.materials.insert(id, m);
        Ok(notes)
    })?;
    let mut j = json!({"material": ctx.doc.scene.materials[&id].name, "updated": material_json(&ctx.doc.scene, id)});
    if !notes.is_empty() {
        j["notes"] = json!(notes);
    }
    Ok(Out::json(j))
}

/// A material as JSON for describe_scene, with defaults omitted.
pub fn material_json(s: &Scene, id: u32) -> Value {
    let m = &s.materials[&id];
    let d = Material::default();
    let tex = |t: Option<TextureId>| t.and_then(|t| s.textures.get(&t)).map(|t| t.name.clone());
    let mut j = json!({"index": id, "name": m.name, "color": polly_hex(m.base_color)});
    if let Some(t) = tex(m.texture) {
        j["texture"] = json!(t);
    }
    if m.unlit {
        j["unlit"] = json!(true);
    }
    if let Some(c) = m.alpha_cutoff {
        j["alpha_cutoff"] = json!(c);
    }
    if m.blend {
        j["blend"] = json!(true);
    }
    if m.double_sided {
        j["double_sided"] = json!(true);
    }
    if m.metallic != d.metallic {
        j["metallic"] = json!(m.metallic);
    }
    if m.roughness != d.roughness {
        j["roughness"] = json!(m.roughness);
    }
    for slot in [Slot::MetallicRoughness, Slot::Normal, Slot::Occlusion, Slot::Emissive] {
        if let Some(t) = tex(m.slot(slot)) {
            j[format!("{}_texture", slot.name())] = json!(t);
        }
    }
    if m.normal_texture.is_some() && m.normal_scale != 1.0 {
        j["normal_scale"] = json!(m.normal_scale);
    }
    if m.occlusion_texture.is_some() && m.occlusion_strength != 1.0 {
        j["occlusion_strength"] = json!(m.occlusion_strength);
    }
    if m.emissive != [0.0; 3] {
        j["emissive"] = json!(polly_hex([m.emissive[0], m.emissive[1], m.emissive[2], 1.0]));
        if m.emissive_strength != 1.0 {
            j["emissive_strength"] = json!(m.emissive_strength);
        }
    }
    if let Some(c) = m.clearcoat {
        j["clearcoat"] = json!(c);
    }
    if let Some(c) = m.anisotropy {
        j["anisotropy"] = json!(c);
    }
    if let Some(c) = m.transmission {
        j["transmission"] = json!(c);
    }
    if let Some(c) = m.specular {
        j["specular"] = json!(c);
    }
    if let Some(c) = &m.shader {
        j["shader"] = json!(c);
    }
    j
}

#[derive(Deserialize, JsonSchema)]
struct AssignMaterial {
    object: Option<String>,
    /// Faces (default: current selection; "all" for every face).
    elements: Option<Vec<String>>,
    /// Material name or index.
    material: String,
}

fn assign_material(ctx: &mut Ctx, a: AssignMaterial) -> Result<Out> {
    let mid = ctx.doc.scene.resolve_material(&a.material)?;
    edit_mesh(ctx, &a.object, Some(&a.elements), "assign material", |m, sel| {
        for f in sel.implied_faces(m) {
            m.faces.get_mut(&f).unwrap().material = mid;
        }
        Ok(Default::default())
    })
}
