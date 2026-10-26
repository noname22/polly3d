//! Process-wide services shared by tool handlers (renderer, settings).

use std::sync::{Mutex, MutexGuard, mpsc};
use std::time::Duration;

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

/// Requests to the live viewer window. Each carries a channel for the outcome.
#[derive(Debug)]
pub enum ViewerCommand {
    Show(mpsc::Sender<Result<(), String>>),
    Hide(mpsc::Sender<Result<(), String>>),
}

/// Sends a command to the viewer's event loop (provided by the host binary).
pub type ViewerControl = Box<dyn Fn(ViewerCommand) -> Result<()> + Send + Sync>;

#[derive(Default)]
pub struct Services {
    pub settings: Mutex<Settings>,
    renderer: Mutex<Option<Renderer>>,
    pub last_view: Mutex<Option<LastView>>,
    viewer: Mutex<Option<ViewerControl>>,
}

impl Services {
    /// Use a renderer provided by the host (e.g. sharing the viewer's GPU device).
    pub fn with_renderer(r: Renderer) -> Self {
        Services { renderer: Mutex::new(Some(r)), ..Default::default() }
    }

    pub fn set_viewer(&self, control: ViewerControl) {
        *self.viewer.lock().unwrap() = Some(control);
    }

    /// Open (`true`) or close the live viewer window and wait for the outcome.
    pub fn viewer(&self, show: bool) -> Result<()> {
        let g = self.viewer.lock().unwrap();
        let control = g.as_ref().context("the live viewer is unavailable (no display, or started with --no-viewer)")?;
        let (tx, rx) = mpsc::channel();
        control(if show { ViewerCommand::Show(tx) } else { ViewerCommand::Hide(tx) })?;
        rx.recv_timeout(Duration::from_secs(10))
            .context("the viewer did not respond")?
            .map_err(|e| anyhow::anyhow!(e))
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
