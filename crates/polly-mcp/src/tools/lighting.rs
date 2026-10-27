//! Scene lighting and camera effects (environment, sun, shadows, ambient
//! occlusion, tonemapping, exposure, bloom). Lighting settings, not styles.

use anyhow::{Result, bail};
use polly_core::lighting::{Lighting, Sun, Tonemap};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![def(
        "set_lighting",
        "Change the scene's lighting and camera effects (saved with the project; render `lighting` overrides them for one image). Only the given fields change. Defaults are polly's classic camera-relative studio lights with no effects. For a game-like PBR look: {\"environment\":\"studio\",\"tonemapping\":\"agx\",\"shadows\":true,\"ambient_occlusion\":true}; add bloom for glowing emissives.",
        false,
        set_lighting,
    )]
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum OnOffAuto {
    Bool(bool),
    /// "auto"
    Auto(String),
}

/// Partial lighting: absent fields keep their value.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct LightingPatch {
    /// Image-based lighting: "studio", "sky", or a path to an equirectangular .hdr file; null or "" removes it.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<String>")]
    pub environment: Option<Option<String>>,
    /// Environment brightness (default 1).
    pub environment_intensity: Option<f32>,
    /// Turn the environment around the vertical axis, degrees.
    pub environment_rotation: Option<f32>,
    /// Show the environment as the background.
    pub show_environment: Option<bool>,
    /// polly's camera-relative key/fill/ambient lights: true, false or "auto" (on unless an environment or sun is set).
    pub view_lights: Option<OnOffAuto>,
    /// A world-fixed directional light {direction: towards the sun, intensity (default 1), color [r,g,b] linear}; null removes it.
    #[serde(default, deserialize_with = "nullable")]
    #[schemars(with = "Option<Sun>")]
    pub sun: Option<Option<Sun>>,
    /// Shadows from the key light and the sun.
    pub shadows: Option<bool>,
    /// Screen-space ambient occlusion (contact darkening).
    pub ambient_occlusion: Option<bool>,
    /// none (clip, classic), agx, tony_mc_mapface, aces_fitted, reinhard, blender_filmic.
    pub tonemapping: Option<Tonemap>,
    /// Exposure in stops (+1 = twice as bright).
    pub exposure: Option<f32>,
    /// Bloom strength around bright pixels (0 = off, ~0.15 typical).
    pub bloom: Option<f32>,
    /// Start from the defaults before applying the other fields.
    #[serde(default)]
    pub reset: bool,
}

impl LightingPatch {
    pub fn apply(&self, base: &Lighting) -> Result<Lighting> {
        let mut l = if self.reset { Lighting::default() } else { base.clone() };
        if let Some(e) = &self.environment {
            if let Some(e) = e {
                check_environment(e)?;
            }
            l.environment = e.clone();
        }
        if let Some(v) = self.environment_intensity {
            if v < 0.0 {
                bail!("environment_intensity must be >= 0");
            }
            l.environment_intensity = v;
        }
        if let Some(v) = self.environment_rotation {
            l.environment_rotation = v;
        }
        if let Some(v) = self.show_environment {
            l.show_environment = v;
        }
        if let Some(v) = &self.view_lights {
            l.view_lights = match v {
                OnOffAuto::Bool(b) => Some(*b),
                OnOffAuto::Auto(s) if s == "auto" => None,
                OnOffAuto::Auto(s) => bail!("view_lights must be true, false or \"auto\", got '{s}'"),
            };
        }
        if let Some(s) = self.sun {
            if let Some(s) = s {
                if s.direction.iter().all(|c| *c == 0.0) {
                    bail!("sun direction must not be zero");
                }
            }
            l.sun = s;
        }
        if let Some(v) = self.shadows {
            l.shadows = v;
        }
        if let Some(v) = self.ambient_occlusion {
            l.ambient_occlusion = v;
        }
        if let Some(v) = self.tonemapping {
            l.tonemapping = v;
        }
        if let Some(v) = self.exposure {
            l.exposure = v.clamp(-10.0, 10.0);
        }
        if let Some(v) = self.bloom {
            l.bloom = v.clamp(0.0, 1.0);
        }
        if l.show_environment && l.environment.is_none() {
            bail!("show_environment needs an environment (\"studio\", \"sky\" or an .hdr path)");
        }
        Ok(l)
    }
}

fn check_environment(e: &str) -> Result<()> {
    match e {
        "studio" | "sky" => Ok(()),
        path if path.to_ascii_lowercase().ends_with(".hdr") => {
            if !std::path::Path::new(path).is_file() {
                bail!("environment file '{path}' not found");
            }
            Ok(())
        }
        other => bail!("unknown environment '{other}' (\"studio\", \"sky\" or a path to an equirectangular .hdr)"),
    }
}

fn set_lighting(ctx: &mut Ctx, a: LightingPatch) -> Result<Out> {
    let l = a.apply(&ctx.doc.scene.lighting)?;
    let l2 = l.clone();
    edit_scene(ctx, "set lighting", move |s| {
        s.lighting = l2;
        Ok(())
    })?;
    let mut j = json!({"lighting": l});
    j["view_lights_on"] = json!(l.view_lights_on());
    Ok(Out::json(j))
}
