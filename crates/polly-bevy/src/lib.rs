//! Polly's renderer: a headless Bevy app on a worker thread.
//!
//! [`BevyRenderer`] is a cheap handle to one process-wide render service.
//! The calling thread prepares the frame (camera, geometry, overlay lines;
//! `polly_render::frame`) and annotates the result (labels, legend); the
//! worker thread syncs the scene into Bevy entities, renders with
//! `StandardMaterial` into an image target and reads it back.

mod conv;
mod gpu;
mod lines;
mod projection;
mod service;

pub use projection::PollyProjection;
pub use service::{BevyRenderer, Stats};
