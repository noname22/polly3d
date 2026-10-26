//! Live viewer: a read-only window mirroring the scenes while the agent works.
//! The event loop runs for the whole process, but the window only exists
//! while the agent has opened it (`show_viewer`) and the user hasn't closed it.
//!
//! With several scenes the window starts on an overview grid; clicking a tile
//! opens that scene alone, and "< Back" (or Backspace/Esc) returns to the grid.
//! Frames are rendered by the shared offscreen renderer and presented with
//! softbuffer, so buttons and captions are plain CPU drawing.
//!
//! Single-scene view: left-drag orbits, wheel zooms. Keys: 1 front, 2 right,
//! 3 top, 4 persp, W wireframe, G grid, S selection, F cycle texture filter
//! (texture / nearest / linear), R reset camera.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use image::RgbaImage;
use polly_core::scene::Filter;
use polly_mcp::overview::{TileRect, layout, render_grid};
use polly_mcp::services::{Services, ViewerCommand};
use polly_mcp::workspace::{MAIN, Workspace};
use polly_render::canvas::{self, Rect};
use polly_render::{Overlays, RenderOptions, View};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// How long a frame may wait for a scene that a tool call is busy with.
const SCENE_WAIT: Duration = Duration::from_millis(20);

struct Win {
    window: Arc<Window>,
    surface: softbuffer::Surface<OwnedDisplayHandle, Arc<Window>>,
}

#[derive(Clone, Copy)]
enum Cam {
    Fixed(&'static str),
    Orbit { yaw: f32, pitch: f32 },
}

/// Something clickable in the window.
#[derive(Clone, PartialEq)]
enum Target {
    Back,
    Tile(String),
}

#[derive(Clone, PartialEq)]
enum Mode {
    Overview,
    Single(String),
}

struct App {
    workspace: Arc<Workspace>,
    services: Arc<Services>,
    context: softbuffer::Context<OwnedDisplayHandle>,
    win: Option<Win>,
    /// Open at startup (from `--viewer`).
    start_open: bool,
    mode: Mode,
    cam: Cam,
    zoom: f32,
    overlays: Overlays,
    filter: Option<Filter>,
    /// What the last frame showed: (list version, per-scene versions).
    seen: Option<(u64, Vec<(String, u64)>)>,
    tiles: Vec<TileRect>,
    back: Option<Rect>,
    /// Target under the pointer when the button went down.
    armed: Option<Target>,
    drag: Option<(f64, f64)>,
    cursor: (f64, f64),
    error: Option<String>,
}

impl App {
    fn view(&self) -> View {
        match self.cam {
            Cam::Fixed("front") => View::Front,
            Cam::Fixed("right") => View::Right,
            Cam::Fixed("top") => View::Top,
            Cam::Fixed(_) => View::Persp,
            Cam::Orbit { yaw, pitch } => View::Orbit { yaw_deg: yaw, pitch_deg: pitch, distance: None },
        }
    }

    /// The mode actually shown: a lone "main" scene skips the overview.
    fn effective_mode(&self) -> Mode {
        match &self.mode {
            Mode::Overview if self.workspace.names().len() == 1 => Mode::Single(MAIN.into()),
            m => m.clone(),
        }
    }

    fn open(&mut self, el: &ActiveEventLoop) -> Result<()> {
        if self.win.is_some() {
            return Ok(());
        }
        let attrs = Window::default_attributes().with_title("Polly3D viewer").with_inner_size(PhysicalSize::new(900, 700));
        let window = Arc::new(el.create_window(attrs)?);
        let surface = softbuffer::Surface::new(&self.context, window.clone()).map_err(|e| anyhow::anyhow!("{e}"))?;
        self.win = Some(Win { window, surface });
        self.seen = None;
        Ok(())
    }

    fn redraw(&self) {
        if let Some(w) = &self.win {
            w.window.request_redraw();
        }
    }

    /// Scene versions relevant to the current mode, read without blocking.
    fn signature(&self) -> Option<(u64, Vec<(String, u64)>)> {
        let names = match self.effective_mode() {
            Mode::Overview => self.workspace.names(),
            Mode::Single(n) => vec![n],
        };
        let mut v = Vec::new();
        for n in names {
            let slot = self.workspace.get(&n).ok()?;
            let g = slot.try_lock().ok()?;
            v.push((n, g.doc.version));
        }
        Some((self.workspace.list_version(), v))
    }

    fn compose(&mut self, size: [u32; 2]) -> Result<Option<RgbaImage>> {
        let mode = self.effective_mode();
        let names = match &mode {
            Mode::Overview => None,
            Mode::Single(n) => Some(vec![n.clone()]),
        };
        let Ok((snaps, busy)) = self.workspace.snapshots(names.as_deref(), None, SCENE_WAIT) else {
            // The scene being viewed was deleted.
            self.mode = Mode::Overview;
            return Ok(None);
        };
        if !busy.is_empty() && snaps.is_empty() {
            return Ok(None); // try again next frame
        }
        let mut renderer = self.services.renderer()?;
        self.tiles.clear();
        self.back = None;
        match mode {
            Mode::Overview => {
                let (cols, tile) = layout(snaps.len(), size[0], size[1]);
                let (img, rects) = render_grid(&mut renderer, &self.services.tiles, &snaps, cols, tile.max(64), Some(size))?;
                self.tiles = rects;
                Ok(Some(img))
            }
            Mode::Single(name) => {
                let opts = RenderOptions {
                    view: self.view(),
                    width: size[0],
                    height: size[1],
                    zoom: self.zoom,
                    overlays: self.overlays.clone(),
                    filter: self.filter,
                    ..RenderOptions::default()
                };
                let mut img = renderer.render(&snaps[0].scene, &opts)?.image;
                if self.workspace.names().len() > 1 {
                    // Top-right: the legend occupies the top-left corner.
                    let scale = if size[0] >= 600 { 2 } else { 1 };
                    let b = canvas::label_rect("< Back", scale, 0.0, 0.0);
                    let b = Rect { x: img.width() as i32 - b.w - 8, y: 8, ..b };
                    canvas::boxed_label(&mut img, b, "< Back", scale, [40, 90, 160, 255], canvas::WHITE);
                    let n = canvas::label_rect(&name, scale, 0.0, 0.0);
                    let n = Rect { x: b.x - n.w - 6, y: 8, ..n };
                    canvas::boxed_label(&mut img, n, &name, scale, [20, 20, 26, 220], canvas::WHITE);
                    self.back = Some(b);
                }
                Ok(Some(img))
            }
        }
    }

    fn draw(&mut self) {
        let Some(size) = self.win.as_ref().map(|w| w.window.inner_size()) else { return };
        let size = [size.width.max(1), size.height.max(1)];
        let sig = self.signature();
        let img = match self.compose(size) {
            Ok(Some(img)) => img,
            Ok(None) => {
                self.redraw();
                return;
            }
            Err(e) => {
                if self.error.as_deref() != Some(&e.to_string()) {
                    tracing::error!("viewer render failed: {e:#}");
                    self.error = Some(e.to_string());
                }
                return;
            }
        };
        self.seen = sig;
        // The renderer caps image size; centre smaller images on the background.
        let (iw, ih) = (img.width().min(size[0]), img.height().min(size[1]));
        let (ox, oy) = ((size[0] - iw) / 2, (size[1] - ih) / 2);
        // Clicks are in window coordinates; shift hit rects by the centring offset.
        for t in &mut self.tiles {
            t.x += ox;
            t.y += oy;
        }
        if let Some(b) = &mut self.back {
            b.x += ox as i32;
            b.y += oy as i32;
        }
        let title = match self.effective_mode() {
            Mode::Overview => format!("Polly3D — {} scenes — click one to open it", self.workspace.names().len()),
            Mode::Single(n) => {
                let cam = match self.cam {
                    Cam::Fixed(c) => c.to_string(),
                    Cam::Orbit { yaw, pitch } => format!("orbit {yaw:.0}°/{pitch:.0}°"),
                };
                format!("Polly3D — {n} — {cam} — [1-4 views, drag orbit, wheel zoom, W/G/S/F/R]")
            }
        };
        let Some(win) = self.win.as_mut() else { return };
        let (Some(w), Some(h)) = (NonZeroU32::new(size[0]), NonZeroU32::new(size[1])) else { return };
        if win.surface.resize(w, h).is_err() {
            return;
        }
        let Ok(mut buf) = win.surface.buffer_mut() else { return };
        buf.fill(0x0012_1317);
        for y in 0..ih {
            let row = ((y + oy) * size[0] + ox) as usize;
            for x in 0..iw {
                let p = img.get_pixel(x, y).0;
                buf[row + x as usize] = (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32;
            }
        }
        let _ = buf.present();
        win.window.set_title(&title);
    }

    fn orbit_from_current(&self) -> (f32, f32) {
        match self.cam {
            Cam::Orbit { yaw, pitch } => (yaw, pitch),
            Cam::Fixed("front") => (0.0, 0.0),
            Cam::Fixed("right") => (90.0, 0.0),
            Cam::Fixed("top") => (0.0, 89.0),
            Cam::Fixed(_) => (35.0, 25.0),
        }
    }

    fn reset_camera(&mut self) {
        self.cam = Cam::Fixed("persp");
        self.zoom = 1.0;
    }

    /// The clickable element under the cursor, if any.
    fn target_at(&self, (x, y): (f64, f64)) -> Option<Target> {
        if let Some(b) = self.back {
            if x >= b.x as f64 && y >= b.y as f64 && x < (b.x + b.w) as f64 && y < (b.y + b.h) as f64 {
                return Some(Target::Back);
            }
        }
        self.tiles.iter().find(|t| t.contains(x, y)).map(|t| Target::Tile(t.scene.clone()))
    }

    /// Button down only arms a target (or starts an orbit drag); the target
    /// fires on release, and only if the pointer is still over it.
    fn press(&mut self) {
        self.armed = self.target_at(self.cursor);
        if self.armed.is_none() && matches!(self.effective_mode(), Mode::Single(_)) {
            self.drag = Some(self.cursor);
        }
    }

    fn release(&mut self) {
        self.drag = None;
        let Some(armed) = self.armed.take() else { return };
        if self.target_at(self.cursor) != Some(armed.clone()) {
            return;
        }
        match armed {
            Target::Back => self.mode = Mode::Overview,
            Target::Tile(scene) => {
                self.mode = Mode::Single(scene);
                self.reset_camera();
            }
        }
        self.redraw();
    }
}

impl ApplicationHandler<ViewerCommand> for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.start_open {
            self.start_open = false;
            if let Err(e) = self.open(el) {
                tracing::error!("viewer unavailable: {e:#}");
            }
        }
    }

    fn user_event(&mut self, el: &ActiveEventLoop, cmd: ViewerCommand) {
        match cmd {
            ViewerCommand::Show { scene, reply } => {
                self.mode = scene.map_or(Mode::Overview, Mode::Single);
                self.reset_camera();
                let r = self.open(el).map_err(|e| format!("{e:#}"));
                self.seen = None;
                self.redraw();
                let _ = reply.send(r);
            }
            ViewerCommand::Hide { reply } => {
                self.win = None;
                let _ = reply.send(Ok(()));
            }
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.win = None,
            WindowEvent::Resized(_) => self.redraw(),
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => match state {
                ElementState::Pressed => self.press(),
                ElementState::Released => self.release(),
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x, position.y);
                if let Some((x0, y0)) = self.drag {
                    let (yaw, pitch) = self.orbit_from_current();
                    let (dx, dy) = ((position.x - x0) as f32, (position.y - y0) as f32);
                    self.cam = Cam::Orbit { yaw: yaw - dx * 0.4, pitch: (pitch + dy * 0.4).clamp(-89.0, 89.0) };
                    self.drag = Some(self.cursor);
                    self.redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } if matches!(self.effective_mode(), Mode::Single(_)) => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.zoom = (self.zoom * 1.1f32.powf(d)).clamp(0.1, 50.0);
                self.redraw();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let single = matches!(self.effective_mode(), Mode::Single(_));
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Backspace | NamedKey::Escape) if single && self.back.is_some() => {
                        self.mode = Mode::Overview
                    }
                    Key::Named(NamedKey::Escape) => self.win = None,
                    Key::Character("1") if single => self.cam = Cam::Fixed("front"),
                    Key::Character("2") if single => self.cam = Cam::Fixed("right"),
                    Key::Character("3") if single => self.cam = Cam::Fixed("top"),
                    Key::Character("4") if single => self.cam = Cam::Fixed("persp"),
                    Key::Character("w") => self.overlays.wireframe = !self.overlays.wireframe,
                    Key::Character("g") => self.overlays.grid = !self.overlays.grid,
                    Key::Character("s") => self.overlays.selection = !self.overlays.selection,
                    Key::Character("f") => {
                        self.filter = match self.filter {
                            None => Some(Filter::Nearest),
                            Some(Filter::Nearest) => Some(Filter::Linear),
                            Some(Filter::Linear) => None,
                        }
                    }
                    Key::Character("r") | Key::Named(NamedKey::Home) => self.reset_camera(),
                    _ => return,
                }
                self.redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if self.win.is_none() {
            el.set_control_flow(ControlFlow::Wait);
            return;
        }
        if let Some(sig) = self.signature() {
            if self.seen.as_ref() != Some(&sig) {
                self.redraw();
            }
        }
        el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(100)));
    }
}

/// Create the event loop (needs a display). Must be called on the main thread.
pub fn event_loop() -> Result<EventLoop<ViewerCommand>> {
    EventLoop::with_user_event().build().context("no display for the live viewer")
}

/// Run the viewer event loop on the main thread. Opens the window right away
/// only if `start_open`; otherwise it waits for `ViewerCommand::Show`.
pub fn run(el: EventLoop<ViewerCommand>, workspace: Arc<Workspace>, services: Arc<Services>, start_open: bool) -> Result<()> {
    let context = softbuffer::Context::new(el.owned_display_handle()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut app = App {
        workspace,
        services,
        context,
        win: None,
        start_open,
        mode: Mode::Overview,
        cam: Cam::Fixed("persp"),
        zoom: 1.0,
        overlays: Overlays::default(),
        filter: None,
        seen: None,
        tiles: vec![],
        back: None,
        armed: None,
        drag: None,
        cursor: (0.0, 0.0),
        error: None,
    };
    el.run_app(&mut app)?;
    Ok(())
}
