//! Bevy runtime support for models made in Polly3D.
//!
//! **Custom shaders.** A Polly material can name a shader and give it
//! parameters; the GLB export writes them to the material's extras:
//!
//! ```json
//! "extras": { "polly": { "shader": { "name": "toon", "params": { "bands": [3], "tint": [1, 0.8, 0.6, 1] } } } }
//! ```
//!
//! [`PollyMaterial`] is `StandardMaterial` extended with those parameters; the
//! fragment shader is chosen per material by name (`<shader dir>/<name>.wesl`),
//! so one material type serves every custom shader. Polly's own renderer uses
//! exactly this type, so a shader looks the same in Polly and in the game.
//!
//! Parameters are packed into `polly.values[i]` (`array<vec4<f32>, 16>`,
//! binding 100 of the material group) in **alphabetical order of their
//! names**; a 1-, 2- or 3-number parameter fills the first components of its
//! slot and the rest are 0.
//!
//! With the `gltf` feature, [`PollyMaterialsPlugin`] replaces the
//! `StandardMaterial` of spawned glTF meshes whose material extras name a
//! shader, loading it from the asset server (so Bevy's file watcher hot-reloads it).

use std::collections::{BTreeMap, HashMap};
use std::sync::RwLock;

use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError};

/// Number of vec4 parameter slots.
pub const MAX_PARAMS: usize = 16;

/// `StandardMaterial` plus a named custom fragment shader and its parameters.
pub type PollyMaterial = ExtendedMaterial<StandardMaterial, PollyShader>;

#[derive(Clone, Copy, Debug, Default, ShaderType, Reflect)]
pub struct PollyParams {
    pub values: [Vec4; MAX_PARAMS],
}

/// The extension part of [`PollyMaterial`].
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
#[bind_group_data(PollyShaderKey)]
pub struct PollyShader {
    #[uniform(100)]
    pub params: PollyParams,
    /// [`shader_id`] of the shader's name.
    pub shader: u32,
}

/// Pipeline key: which shader the material uses.
#[repr(C)]
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub struct PollyShaderKey {
    pub shader: u32,
}

impl From<&PollyShader> for PollyShaderKey {
    fn from(s: &PollyShader) -> Self {
        PollyShaderKey { shader: s.shader }
    }
}

/// Shader handles by id: specialization looks the fragment shader up here.
static SHADERS: RwLock<Option<HashMap<u32, Handle<Shader>>>> = RwLock::new(None);

/// Stable id of a shader name (FNV-1a).
pub fn shader_id(name: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Make `handle` the shader for materials naming `name`.
pub fn register_shader(name: &str, handle: Handle<Shader>) {
    SHADERS.write().unwrap().get_or_insert_with(HashMap::new).insert(shader_id(name), handle);
}

/// The registered shader of a name, if any.
pub fn registered_shader(name: &str) -> Option<Handle<Shader>> {
    SHADERS.read().unwrap().as_ref()?.get(&shader_id(name)).cloned()
}

/// Every registered shader handle and its id.
pub fn registered_shaders() -> Vec<(u32, Handle<Shader>)> {
    SHADERS.read().unwrap().as_ref().map(|m| m.iter().map(|(k, v)| (*k, v.clone())).collect()).unwrap_or_default()
}

impl PollyShader {
    /// From a shader name and its parameters (name -> 1..4 numbers).
    pub fn new(name: &str, params: &BTreeMap<String, Vec<f32>>) -> PollyShader {
        let mut p = PollyParams::default();
        for (i, v) in params.values().take(MAX_PARAMS).enumerate() {
            let mut a = [0.0; 4];
            for (k, x) in v.iter().take(4).enumerate() {
                a[k] = *x;
            }
            p.values[i] = Vec4::from_array(a);
        }
        PollyShader { params: p, shader: shader_id(name) }
    }

    /// From `extras.polly.shader` JSON: `{"name": ..., "params": {...}}`.
    /// Parameters may be numbers or arrays of up to four numbers.
    pub fn from_json(v: &serde_json::Value) -> Option<(String, PollyShader)> {
        let name = v.get("name")?.as_str()?.to_string();
        let mut params = BTreeMap::new();
        if let Some(obj) = v.get("params").and_then(|p| p.as_object()) {
            for (k, x) in obj {
                let nums: Vec<f32> = match x {
                    serde_json::Value::Number(n) => vec![n.as_f64()? as f32],
                    serde_json::Value::Array(a) => a.iter().filter_map(|e| e.as_f64()).map(|f| f as f32).collect(),
                    _ => continue,
                };
                params.insert(k.clone(), nums);
            }
        }
        let s = PollyShader::new(&name, &params);
        Some((name, s))
    }
}

impl MaterialExtension for PollyShader {
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Only the main (forward) pass uses the custom fragment shader; depth,
        // normal and shadow prepasses keep the standard ones.
        let Some(fragment) = descriptor.fragment.as_mut() else { return Ok(()) };
        let prepass = fragment.shader_defs.iter().any(|d| matches!(d, bevy::shader::ShaderDefVal::Bool(n, true) if n == "PREPASS_PIPELINE"));
        if prepass {
            return Ok(());
        }
        let shaders = SHADERS.read().unwrap();
        if let Some(h) = shaders.as_ref().and_then(|m| m.get(&key.bind_group_data.shader)) {
            fragment.shader = h.clone();
        }
        Ok(())
    }
}

/// A WESL template for a new Polly shader: the standard PBR result with the
/// parameters at hand. `params` names the slots in the comment.
pub fn shader_template(name: &str, params: &[String]) -> String {
    let mut slots = String::new();
    for (i, p) in params.iter().enumerate().take(MAX_PARAMS) {
        slots.push_str(&format!("//   polly.values[{i}] = {p}\n"));
    }
    if slots.is_empty() {
        slots.push_str("//   (no parameters yet)\n");
    }
    format!(
        r#"// Polly shader "{name}": a fragment shader on top of Bevy's StandardMaterial.
// Parameters (alphabetical by name), each a vec4:
{slots}import bevy_pbr::render::{{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing}},
    forward_io::{{VertexOutput, FragmentOutput}},
}};

struct PollyParams {{
    values: array<vec4<f32>, 16>,
}}

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var<uniform> polly: PollyParams;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {{
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    // Change the surface before lighting here (base colour, roughness, normal...).
    pbr_input.material.base_color = alpha_discard(pbr_input.material.flags, pbr_input.material.alpha_cutoff, pbr_input.material.base_color);
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    // ...or change the lit colour here.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}}
"#
    )
}

/// Game-side plugin: registers [`PollyMaterial`] and (with the `gltf`
/// feature) swaps materials of glTF meshes whose extras name a shader.
pub struct PollyMaterialsPlugin {
    /// Asset directory holding `<name>.wesl` shaders (default "shaders").
    pub shader_dir: String,
}

impl Default for PollyMaterialsPlugin {
    fn default() -> Self {
        PollyMaterialsPlugin { shader_dir: "shaders".into() }
    }
}

#[derive(Resource)]
#[cfg_attr(not(feature = "gltf"), allow(dead_code))]
struct ShaderDir(String);

impl Plugin for PollyMaterialsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<PollyMaterial>::default());
        app.insert_resource(ShaderDir(self.shader_dir.clone()));
        #[cfg(feature = "gltf")]
        app.add_systems(Update, swap_gltf_materials);
    }
}

#[cfg(feature = "gltf")]
#[allow(clippy::type_complexity)]
fn swap_gltf_materials(
    mut commands: Commands,
    dir: Res<ShaderDir>,
    assets: Res<AssetServer>,
    standard: Res<Assets<StandardMaterial>>,
    mut polly: ResMut<Assets<PollyMaterial>>,
    mut made: Local<HashMap<AssetId<StandardMaterial>, Handle<PollyMaterial>>>,
    q: Query<(Entity, &MeshMaterial3d<StandardMaterial>, &bevy::gltf::GltfMaterialExtras), Added<bevy::gltf::GltfMaterialExtras>>,
) {
    for (e, mat, extras) in &q {
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&extras.value) else { continue };
        let Some((name, ext)) = PollyShader::from_json(&json["polly"]["shader"]) else { continue };
        if registered_shader(&name).is_none() {
            register_shader(&name, assets.load(format!("{}/{name}.wesl", dir.0)));
        }
        let handle = match made.get(&mat.0.id()) {
            Some(h) => h.clone(),
            None => {
                let Some(base) = standard.get(&mat.0) else { continue };
                let h = polly.add(ExtendedMaterial { base: base.clone(), extension: ext });
                made.insert(mat.0.id(), h.clone());
                h
            }
        };
        commands.entity(e).remove::<MeshMaterial3d<StandardMaterial>>().insert(MeshMaterial3d(handle));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_pack_alphabetically() {
        let j = serde_json::json!({"name": "toon", "params": {"tint": [1, 0.5, 0.25], "bands": 3}});
        let (name, s) = PollyShader::from_json(&j).unwrap();
        assert_eq!(name, "toon");
        assert_eq!(s.shader, shader_id("toon"));
        assert_eq!(s.params.values[0], Vec4::new(3.0, 0.0, 0.0, 0.0));
        assert_eq!(s.params.values[1], Vec4::new(1.0, 0.5, 0.25, 0.0));
        assert_ne!(shader_id("toon"), shader_id("toon2"));
    }
}
