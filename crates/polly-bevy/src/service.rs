//! The process-wide render service: a worker thread owning the Bevy app,
//! fed through a channel.

use std::sync::{Arc, Mutex, OnceLock, mpsc};

use anyhow::{Context, Result, anyhow};
use polly_core::scene::Scene;
use polly_render::{Rendered, RenderOptions, frame};

use crate::gpu::{Gpu, Job};

type Reply = mpsc::Sender<Result<image::RgbaImage>>;

enum Request {
    Render(Job, Reply),
    Stats(mpsc::Sender<Stats>),
}

/// What the render service keeps cached between requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Requests served.
    pub jobs: u64,
    /// Objects with cached mesh entities.
    pub objects: usize,
    /// Cached mesh entities (one per object and material).
    pub meshes: usize,
    pub materials: usize,
    pub images: usize,
}

/// Handle to the render service. Clones share one Bevy app; requests from
/// any thread are served one at a time.
#[derive(Clone)]
pub struct BevyRenderer {
    tx: Arc<Mutex<mpsc::Sender<Request>>>,
    adapter: Arc<String>,
}

static SERVICE: OnceLock<Result<BevyRenderer, String>> = OnceLock::new();

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into())
}

impl BevyRenderer {
    /// The process-wide renderer, started on first use. Set
    /// `POLLY_ADAPTER=software` to force the software (llvmpipe) adapter.
    pub fn global() -> Result<BevyRenderer> {
        SERVICE.get_or_init(start).clone().map_err(|e| anyhow!(e))
    }

    /// Name of the GPU adapter Bevy uses.
    pub fn adapter_name(&self) -> &str {
        &self.adapter
    }

    /// Render offscreen at the options' size, with labels and legend.
    pub fn render(&self, scene: &Scene, o: &RenderOptions) -> Result<Rendered> {
        self.render_sized(scene, o, o.clamped_size())
    }

    /// Render at an explicit size (the viewer may exceed the agent's 2048 px limit).
    pub fn render_sized(&self, scene: &Scene, o: &RenderOptions, size: [u32; 2]) -> Result<Rendered> {
        let f = Arc::new(frame::plan_sized(scene, o, size));
        let (tx, rx) = mpsc::channel();
        let job = Job { scene: scene.clone(), opts: o.clone(), frame: f.clone() };
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(Request::Render(job, tx)).map_err(|_| anyhow!("the render service stopped"))?;
        let image = rx.recv().context("the render service stopped")??;
        Ok(frame::finish(image, scene, o, &f))
    }

    /// Cache statistics (for tests and diagnostics).
    pub fn stats(&self) -> Result<Stats> {
        let (tx, rx) = mpsc::channel();
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(Request::Stats(tx)).map_err(|_| anyhow!("the render service stopped"))?;
        rx.recv().context("the render service stopped")
    }

    /// 2x2 sheet: front, right, top, perspective.
    pub fn contact_sheet(&self, scene: &Scene, o: &RenderOptions) -> Result<Rendered> {
        frame::contact_sheet(o, |to| self.render(scene, to))
    }
}

fn start() -> Result<BevyRenderer, String> {
    let (tx, rx) = mpsc::channel::<Request>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<String, String>>();
    std::thread::Builder::new()
        .name("polly-bevy".into())
        .spawn(move || worker(rx, ready_tx))
        .map_err(|e| format!("cannot start the render thread: {e}"))?;
    let adapter = ready_rx.recv().map_err(|_| "the render thread died during startup".to_string())??;
    tracing::info!("bevy renderer on {adapter}");
    Ok(BevyRenderer { tx: Arc::new(Mutex::new(tx)), adapter: Arc::new(adapter) })
}

fn build() -> Result<Gpu, String> {
    match std::panic::catch_unwind(Gpu::new) {
        Ok(Ok(g)) => Ok(g),
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(p) => Err(format!("no GPU adapter available for rendering (try POLLY_ADAPTER=software): {}", panic_message(p))),
    }
}

fn worker(rx: mpsc::Receiver<Request>, ready: mpsc::Sender<Result<String, String>>) {
    let mut gpu = match build() {
        Ok(g) => {
            let _ = ready.send(Ok(g.adapter.clone()));
            Some(g)
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    for req in rx {
        let (job, reply) = match req {
            Request::Render(job, reply) => (job, reply),
            Request::Stats(tx) => {
                let _ = tx.send(gpu.as_ref().map(Gpu::stats).unwrap_or_default());
                continue;
            }
        };
        let Some(g) = gpu.as_mut() else {
            let _ = reply.send(Err(anyhow!("the renderer could not be restarted")));
            continue;
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.render(&job))) {
            Ok(r) => {
                let _ = reply.send(r);
            }
            Err(p) => {
                let msg = panic_message(p);
                tracing::error!("renderer panicked: {msg}; restarting it");
                let _ = reply.send(Err(anyhow!("internal renderer error: {msg}")));
                gpu = build().map_err(|e| tracing::error!("renderer restart failed: {e}")).ok();
            }
        }
    }
}
