//! Polly3D renderer: the agent's eyes.
//!
//! One shading model (base colour x texture x corner colour, flat or smooth,
//! key + fill + rim + hemisphere ambient lighting, MSAA 4x, sRGB output) plus
//! overlays annotated with element ids (`f12`, `v3`, `e3-9`) so the agent can
//! reference geometry it sees.
//!
//! Colour conventions: material `base_color` and corner colours are linear;
//! textures are sRGB; the scene background and overlay colours are sRGB.

mod annotate;
mod camera;
pub mod canvas;
mod gpu;
mod lines;
mod options;
mod prep;
mod uv_layout;

pub use camera::{CameraInfo, orbit_dir};
pub use gpu::{Rendered, Renderer, Tile, resolve_camera};
pub use options::{Overlays, RenderOptions, View};
pub use prep::{Hit, pick};
pub use uv_layout::render_uv_layout;

pub use annotate::MAX_LABELS;
pub use wgpu;
