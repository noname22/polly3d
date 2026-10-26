//! Live viewer: a read-only window mirroring the document while the agent works.
//! The event loop runs for the whole process, but the window only exists
//! while the agent has opened it (`show_viewer`) and the user hasn't closed it.
//!
//! Mouse: left-drag orbits, wheel zooms. Keys: 1 front, 2 right, 3 top, 4 persp,
//! W wireframe, G grid, S selection, F cycle texture filter (texture / nearest /
//! linear), R reset camera.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use polly_core::document::Document;
use polly_core::scene::Filter;
use polly_mcp::services::ViewerCommand;
use polly_render::{Overlays, RenderOptions, Renderer, View, wgpu};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::{Key, NamedKey};
use winit::raw_window_handle::{DisplayHandle, HandleError, HasDisplayHandle};
use winit::window::{Window, WindowId};

/// `OwnedDisplayHandle` with the `Debug` impl wgpu wants.
struct Display(OwnedDisplayHandle);

impl std::fmt::Debug for Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Display")
    }
}

impl HasDisplayHandle for Display {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.0.display_handle()
    }
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
}

#[derive(Clone, Copy)]
enum Cam {
    Fixed(&'static str),
    Orbit { yaw: f32, pitch: f32 },
}

struct App {
    doc: Arc<Mutex<Document>>,
    instance: wgpu::Instance,
    gfx: Option<Gfx>,
    start_open: bool,
    cam: Cam,
    zoom: f32,
    overlays: Overlays,
    filter: Option<Filter>,
    seen_version: Option<u64>,
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

    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let attrs = Window::default_attributes().with_title("Polly3D viewer").with_inner_size(PhysicalSize::new(900, 700));
        let window = Arc::new(el.create_window(attrs)?);
        let surface = self.instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(self.instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .context("no GPU adapter compatible with the window")?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("polly-viewer"),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            ..Default::default()
        }))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("surface not supported by adapter")?;
        // Prefer an sRGB format; the renderer encodes manually otherwise.
        let caps = surface.get_capabilities(&adapter);
        if let Some(f) = caps.formats.iter().find(|f| f.is_srgb()) {
            config.format = *f;
        }
        surface.configure(&device, &config);
        let renderer = Renderer::new(&self.instance, &adapter, device, queue);
        self.gfx = Some(Gfx { window, surface, config, renderer });
        Ok(())
    }

    fn draw(&mut self) {
        let opts = RenderOptions {
            view: self.view(),
            zoom: self.zoom,
            overlays: self.overlays.clone(),
            filter: self.filter,
            ..RenderOptions::default()
        };
        // Never block on the document: a tool call may hold it while waiting for us.
        let Ok(d) = self.doc.try_lock() else {
            self.redraw();
            return;
        };
        let (scene, version) = (d.scene.clone(), d.version);
        drop(d);
        self.seen_version = Some(version);
        let Some(g) = self.gfx.as_mut() else { return };
        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            _ => {
                g.surface.configure(g.renderer.device(), &g.config);
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let size = [g.config.width, g.config.height];
        if let Err(e) = g.renderer.render_to_view(&scene, &opts, &view, g.config.format, size) {
            if self.error.as_deref() != Some(&e.to_string()) {
                tracing::error!("viewer render failed: {e:#}");
                self.error = Some(e.to_string());
            }
        }
        g.renderer.queue().present(frame);
        let name = match self.cam {
            Cam::Fixed(n) => n.to_string(),
            Cam::Orbit { yaw, pitch } => format!("orbit {yaw:.0}°/{pitch:.0}°"),
        };
        let filt = match self.filter {
            None => "texture filters",
            Some(Filter::Nearest) => "nearest",
            Some(Filter::Linear) => "linear",
        };
        g.window.set_title(&format!(
            "Polly3D — {} objects — {name} — {filt} — [1-4 views, drag orbit, wheel zoom, W/G/S/F/R]",
            scene.objects.len()
        ));
    }

    fn redraw(&self) {
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
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
}

impl ApplicationHandler<ViewerCommand> for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.start_open && self.gfx.is_none() {
            self.start_open = false;
            if let Err(e) = self.init(el) {
                tracing::error!("viewer unavailable: {e:#}");
            }
        }
    }

    fn user_event(&mut self, el: &ActiveEventLoop, cmd: ViewerCommand) {
        match cmd {
            ViewerCommand::Show(reply) => {
                let r = if self.gfx.is_some() { Ok(()) } else { self.init(el).map_err(|e| format!("{e:#}")) };
                self.seen_version = None;
                self.redraw();
                let _ = reply.send(r);
            }
            ViewerCommand::Hide(reply) => {
                self.gfx = None;
                let _ = reply.send(Ok(()));
            }
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.gfx = None,
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.config.width = size.width.max(1);
                    g.config.height = size.height.max(1);
                    g.surface.configure(g.renderer.device(), &g.config);
                }
                self.redraw();
            }
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.drag = (state == ElementState::Pressed).then_some(self.cursor);
            }
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
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.zoom = (self.zoom * 1.1f32.powf(d)).clamp(0.1, 50.0);
                self.redraw();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key.as_ref() {
                    Key::Character("1") => self.cam = Cam::Fixed("front"),
                    Key::Character("2") => self.cam = Cam::Fixed("right"),
                    Key::Character("3") => self.cam = Cam::Fixed("top"),
                    Key::Character("4") => self.cam = Cam::Fixed("persp"),
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
                    Key::Character("r") | Key::Named(NamedKey::Home) => {
                        self.cam = Cam::Fixed("persp");
                        self.zoom = 1.0;
                    }
                    Key::Named(NamedKey::Escape) => self.gfx = None,
                    _ => return,
                }
                self.redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_none() {
            el.set_control_flow(ControlFlow::Wait);
            return;
        }
        if let Ok(d) = self.doc.try_lock() {
            if self.seen_version != Some(d.version) {
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
pub fn run(el: EventLoop<ViewerCommand>, doc: Arc<Mutex<Document>>, start_open: bool) -> Result<()> {
    let display = Box::new(Display(el.owned_display_handle()));
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(display));
    let mut app = App {
        doc,
        instance,
        gfx: None,
        start_open,
        cam: Cam::Fixed("persp"),
        zoom: 1.0,
        overlays: Overlays::default(),
        filter: None,
        seen_version: None,
        drag: None,
        cursor: (0.0, 0.0),
        error: None,
    };
    el.run_app(&mut app)?;
    Ok(())
}
