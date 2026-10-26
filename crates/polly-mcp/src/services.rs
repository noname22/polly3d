//! Process-wide services shared by tool handlers (renderer, settings).

use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result};
use polly_render::{Renderer, Tile};

#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// If set, mutating mesh tools append a preview render of this size.
    pub auto_preview: Option<u32>,
    /// Last project path used by save/load.
    pub project_path: Option<std::path::PathBuf>,
}

/// The camera(s) of the most recent render, for `pick` / `paint_stroke`.
#[derive(Debug, Clone)]
pub struct LastView {
    pub camera: polly_render::CameraInfo,
    pub tiles: Vec<Tile>,
}

#[derive(Default)]
pub struct Services {
    pub settings: Mutex<Settings>,
    renderer: Mutex<Option<Renderer>>,
    pub last_view: Mutex<Option<LastView>>,
}

impl Services {
    /// Use a renderer provided by the host (e.g. sharing the viewer's GPU device).
    pub fn with_renderer(r: Renderer) -> Self {
        Services { renderer: Mutex::new(Some(r)), ..Default::default() }
    }

    /// The renderer, created on first use.
    pub fn renderer(&self) -> Result<RendererGuard<'_>> {
        let mut g = self.renderer.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            *g = Some(Renderer::headless().context("no GPU adapter available for rendering (try POLLY_ADAPTER=software)")?);
        }
        Ok(RendererGuard(g))
    }
}

pub struct RendererGuard<'a>(MutexGuard<'a, Option<Renderer>>);

impl std::ops::Deref for RendererGuard<'_> {
    type Target = Renderer;
    fn deref(&self) -> &Renderer {
        self.0.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for RendererGuard<'_> {
    fn deref_mut(&mut self) -> &mut Renderer {
        self.0.as_mut().unwrap()
    }
}
