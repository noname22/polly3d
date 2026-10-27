//! Polly3D render helpers (CPU): camera resolution and framing, prepared
//! geometry and picking, overlay line lists, element-id labels and legend
//! drawn after readback, contact sheets and the UV layout view. The GPU side
//! is the Bevy render service in `polly-bevy`.
//!
//! Colour conventions: material `base_color` and corner colours are linear;
//! textures are sRGB; the scene background and overlay colours are sRGB.

mod annotate;
mod camera;
pub mod canvas;
pub mod frame;
pub mod lines;
mod options;
pub mod prep;
mod uv_layout;

pub use camera::{CameraInfo, orbit_dir};
pub use frame::{Frame, Rendered, Tile, reference_bounds, resolve_camera};
pub use options::{Overlays, RenderOptions, View};
pub use prep::{Hit, pick};
pub use uv_layout::render_uv_layout;

pub use annotate::MAX_LABELS;
