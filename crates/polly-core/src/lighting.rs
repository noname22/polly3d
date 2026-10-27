//! Scene lighting and camera effects. These are lighting settings, not render
//! styles: there is one shading model, and these choose the light it sees
//! (camera-relative studio lights, an environment, a sun) and how the camera
//! develops the image (tonemapping, exposure, bloom, ambient occlusion).
//! The defaults reproduce polly's classic look.

use serde::{Deserialize, Serialize};

/// How the camera maps scene light to display colours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Tonemap {
    /// Clip at white (the classic look; colours shown as authored).
    #[default]
    None,
    /// AgX: soft highlight roll-off, natural saturation (good default for PBR).
    Agx,
    TonyMcMapface,
    AcesFitted,
    Reinhard,
    BlenderFilmic,
}

/// A directional light fixed in the world (not following the camera).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Sun {
    /// Direction towards the sun (world space; need not be unit length).
    pub direction: [f32; 3],
    /// Brightness (1 = the classic key light's 0.8 x 1.25; typical 1..3).
    #[serde(default = "one")]
    pub intensity: f32,
    /// Linear RGB colour.
    #[serde(default = "white")]
    pub color: [f32; 3],
}

fn one() -> f32 {
    1.0
}
fn white() -> [f32; 3] {
    [1.0; 3]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct Lighting {
    /// Image-based lighting: "studio" (soft neutral studio), "sky" (outdoor sky and ground),
    /// or a path to an equirectangular .hdr image. None = no environment.
    pub environment: Option<String>,
    /// Environment brightness multiplier (default 1).
    pub environment_intensity: f32,
    /// Turn the environment around the vertical axis (degrees).
    pub environment_rotation: f32,
    /// Show the environment as the background instead of the flat background colour.
    pub show_environment: bool,
    /// polly's camera-relative key + fill + ambient lights. Default: on unless
    /// an environment or a sun is set.
    pub view_lights: Option<bool>,
    pub sun: Option<Sun>,
    /// Shadows from the key light and the sun.
    pub shadows: bool,
    /// Screen-space ambient occlusion (contact darkening in creases).
    pub ambient_occlusion: bool,
    pub tonemapping: Tonemap,
    /// Exposure adjustment in stops (+1 = twice as bright).
    pub exposure: f32,
    /// Glow around bright (emissive, > 1) pixels; 0 = off, 0.15 = typical.
    pub bloom: f32,
}

impl Default for Lighting {
    fn default() -> Self {
        Lighting {
            environment: None,
            environment_intensity: 1.0,
            environment_rotation: 0.0,
            show_environment: false,
            view_lights: None,
            sun: None,
            shadows: false,
            ambient_occlusion: false,
            tonemapping: Tonemap::None,
            exposure: 0.0,
            bloom: 0.0,
        }
    }
}

impl Lighting {
    pub fn is_default(&self) -> bool {
        *self == Lighting::default()
    }
    /// Whether the camera-relative studio lights are on.
    pub fn view_lights_on(&self) -> bool {
        self.view_lights.unwrap_or(self.environment.is_none() && self.sun.is_none())
    }
    /// Effects that need a high-dynamic-range camera.
    pub fn needs_hdr(&self) -> bool {
        self.bloom > 0.0 || self.tonemapping != Tonemap::None
    }
}
