//! Live viewer: an egui window mirroring the scenes while the agent works.
//! The event loop runs for the whole process, but the window only exists
//! while the agent has opened it (`show_viewer`) and the user hasn't closed it.
//!
//! With several scenes the window starts on an overview of thumbnails;
//! clicking one opens that scene alone, and "< Back" (or Backspace/Esc)
//! returns. The single-scene view renders straight into a GPU texture that
//! egui displays, so orbiting never reads pixels back to the CPU.
//!
//! Single-scene view: left-drag orbits, wheel zooms. Keys: 1 front, 2 right,
//! 3 top, 4 persp, W wireframe, G grid, S selection, F cycle texture filter
//! (texture / nearest / linear), R reset camera.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use egui::load::SizedTexture;
use polly_core::scene::Filter;
use polly_mcp::overview::{layout, tile_options};
use polly_mcp::services::ViewerCommand;
use polly_mcp::workspace::{MAIN, Snapshot, Workspace};
use polly_render::{Overlays, RenderOptions, View, wgpu};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::raw_window_handle::{DisplayHandle, HandleError, HasDisplayHandle};
use winit::window::{Window, WindowId};

/// How long a frame may wait for a scene that a tool call is busy with.
const SCENE_WAIT: Duration = Duration::from_millis(20);
/// Format of the 3-D viewport texture egui samples. egui treats sampled
/// texels as gamma-encoded, so this is a non-sRGB format: the renderer
/// encodes sRGB in its shader for such targets.
const VIEWPORT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

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

#[derive(Clone, Copy, PartialEq)]
enum Cam {
    Front,
    Right,
    Top,
    Persp,
    Orbit { yaw: f32, pitch: f32 },
}

impl Cam {
    fn view(self) -> View {
        match self {
            Cam::Front => View::Front,
            Cam::Right => View::Right,
            Cam::Top => View::Top,
            // Same angle as View::Persp but orbit framing, so the first drag doesn't jump.
            Cam::Persp => View::Orbit { yaw_deg: 35.0, pitch_deg: 25.0, distance: None },
            Cam::Orbit { yaw, pitch } => View::Orbit { yaw_deg: yaw, pitch_deg: pitch, distance: None },
        }
    }

    fn angles(self) -> (f32, f32) {
        match self {
            Cam::Orbit { yaw, pitch } => (yaw, pitch),
            Cam::Front => (0.0, 0.0),
            Cam::Right => (90.0, 0.0),
            Cam::Top => (0.0, 89.0),
            Cam::Persp => (35.0, 25.0),
        }
    }

    fn label(self) -> String {
        match self {
            Cam::Front => "front".into(),
            Cam::Right => "right".into(),
            Cam::Top => "top".into(),
            Cam::Persp => "persp".into(),
            Cam::Orbit { yaw, pitch } => format!("orbit {yaw:.0}°/{pitch:.0}°"),
        }
    }
}

#[derive(Clone, PartialEq)]
enum Mode {
    Overview,
    Single(String),
}

/// What the user is looking at; survives closing and reopening the window.
struct ViewState {
    mode: Mode,
    cam: Cam,
    zoom: f32,
    overlays: Overlays,
    filter: Option<Filter>,
}

impl ViewState {
    fn reset_camera(&mut self) {
        self.cam = Cam::Persp;
        self.zoom = 1.0;
    }
}

struct Viewport {
    size: [u32; 2],
    view: wgpu::TextureView,
    id: egui::TextureId,
}

/// GPU resources the UI code needs while egui builds a frame.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    scene: polly_render::Renderer,
    painter: egui_wgpu::Renderer,
    viewport: Option<Viewport>,
    /// Scene thumbnails: name -> (scene version, size, texture).
    thumbs: HashMap<String, (u64, u32, egui::TextureHandle)>,
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    egui: egui::Context,
    state: egui_winit::State,
    gpu: Gpu,
}

struct App {
    workspace: Arc<Workspace>,
    instance: wgpu::Instance,
    gfx: Option<Gfx>,
    /// Open at startup (from `--viewer`).
    start_open: bool,
    view: ViewState,
    /// Scene versions shown by the last frame, to redraw when the agent edits.
    seen: Option<(u64, Vec<(String, u64)>)>,
    error: Option<String>,
}

impl App {
    fn open(&mut self, el: &ActiveEventLoop) -> Result<()> {
        if self.gfx.is_some() {
            return Ok(());
        }
        let attrs = Window::default_attributes().with_title("Polly3D viewer").with_inner_size(PhysicalSize::new(1000, 760));
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
            .context("window surface not supported by the adapter")?;
        config.format = egui_wgpu::preferred_framebuffer_format(&surface.get_capabilities(&adapter).formats)?;
        surface.configure(&device, &config);
        let painter = egui_wgpu::Renderer::new(&device, config.format, egui_wgpu::RendererOptions::default());
        let scene = polly_render::Renderer::new(&self.instance, &adapter, device.clone(), queue.clone());
        let egui = egui::Context::default();
        let state = egui_winit::State::new(egui.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        self.gfx = Some(Gfx {
            window,
            surface,
            config,
            egui,
            state,
            gpu: Gpu { device, queue, scene, painter, viewport: None, thumbs: HashMap::new() },
        });
        self.seen = None;
        Ok(())
    }

    fn redraw(&self) {
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }

    /// The mode actually shown: a lone "main" scene skips the overview.
    fn effective_mode(&self) -> Mode {
        effective_mode(&self.view.mode, &self.workspace)
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

    fn frame(&mut self) {
        let sig = self.signature();
        let Some(gfx) = self.gfx.as_mut() else { return };
        let input = gfx.state.take_egui_input(&gfx.window);
        let ctx = gfx.egui.clone();
        let mut result = Ok(true);
        let out = ctx.run_ui(input, |ui| result = build_ui(ui, &mut self.view, &mut gfx.gpu, &self.workspace));
        gfx.state.handle_platform_output(&gfx.window, out.platform_output);
        match result {
            Ok(complete) => {
                // An incomplete frame (a scene was busy) is retried shortly.
                self.seen = if complete { sig } else { None };
                self.error = None;
            }
            Err(e) => {
                if self.error.as_deref() != Some(&e.to_string()) {
                    tracing::error!("viewer: {e:#}");
                    self.error = Some(e.to_string());
                }
            }
        }
        let gpu = &mut gfx.gpu;
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                gpu.painter.update_texture(&gpu.device, &gpu.queue, *id, delta);
            }
        }
        let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            _ => {
                gfx.surface.configure(&gpu.device, &gfx.config);
                gfx.window.request_redraw();
                return;
            }
        };
        let target = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gfx.config.width, gfx.config.height],
            pixels_per_point: out.pixels_per_point,
        };
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("egui") });
        let extra = gpu.painter.update_buffers(&gpu.device, &gpu.queue, &mut enc, &jobs, &screen);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            gpu.painter.render(&mut pass, &jobs, &screen);
        }
        gpu.queue.submit(extra.into_iter().chain([enc.finish()]));
        gpu.queue.present(frame);
        for id in &out.textures_delta.free {
            gpu.painter.free_texture(id);
        }
        if out.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|v| v.repaint_delay.is_zero()) {
            gfx.window.request_redraw();
        }
        let title = match effective_mode(&self.view.mode, &self.workspace) {
            Mode::Overview => format!("Polly3D — {} scenes", self.workspace.names().len()),
            Mode::Single(n) => format!("Polly3D — {n}"),
        };
        gfx.window.set_title(&title);
    }
}

fn effective_mode(mode: &Mode, workspace: &Workspace) -> Mode {
    match mode {
        Mode::Overview if workspace.names().len() == 1 => Mode::Single(MAIN.into()),
        m => m.clone(),
    }
}

/// Build one frame of UI. Returns `Ok(false)` if a scene was busy and the
/// frame should be redrawn soon.
fn build_ui(ui: &mut egui::Ui, view: &mut ViewState, gpu: &mut Gpu, workspace: &Workspace) -> Result<bool> {
    let mode = effective_mode(&view.mode, workspace);
    let several = workspace.names().len() > 1;
    shortcuts(ui, view, &mode, several);
    egui::Panel::top("toolbar").show(ui, |ui| {
        ui.horizontal(|ui| match &mode {
            Mode::Overview => {
                ui.strong(format!("{} scenes", workspace.names().len()));
                ui.label("— click one to open it");
            }
            Mode::Single(name) => toolbar(ui, view, name, several),
        });
    });
    let mut complete = true;
    egui::CentralPanel::default().show(ui, |ui| -> Result<()> {
        match &mode {
            Mode::Overview => complete = overview(ui, view, gpu, workspace)?,
            Mode::Single(name) => complete = single(ui, view, gpu, workspace, name)?,
        }
        Ok(())
    })
    .inner?;
    Ok(complete)
}

fn shortcuts(ui: &mut egui::Ui, view: &mut ViewState, mode: &Mode, several: bool) {
    use egui::Key;
    let single = matches!(mode, Mode::Single(_));
    ui.input(|i| {
        if single && several && (i.key_pressed(Key::Backspace) || i.key_pressed(Key::Escape)) {
            view.mode = Mode::Overview;
        }
        if single {
            for (key, cam) in [(Key::Num1, Cam::Front), (Key::Num2, Cam::Right), (Key::Num3, Cam::Top), (Key::Num4, Cam::Persp)] {
                if i.key_pressed(key) {
                    view.cam = cam;
                }
            }
        }
        if i.key_pressed(Key::W) {
            view.overlays.wireframe ^= true;
        }
        if i.key_pressed(Key::G) {
            view.overlays.grid ^= true;
        }
        if i.key_pressed(Key::S) {
            view.overlays.selection ^= true;
        }
        if i.key_pressed(Key::F) {
            view.filter = match view.filter {
                None => Some(Filter::Nearest),
                Some(Filter::Nearest) => Some(Filter::Linear),
                Some(Filter::Linear) => None,
            };
        }
        if i.key_pressed(Key::R) {
            view.reset_camera();
        }
    });
}

fn toolbar(ui: &mut egui::Ui, view: &mut ViewState, name: &str, several: bool) {
    if several && ui.button("< Back").on_hover_text("Backspace / Esc").clicked() {
        view.mode = Mode::Overview;
    }
    ui.strong(name);
    ui.separator();
    for (cam, label, key) in [(Cam::Front, "Front", "1"), (Cam::Right, "Right", "2"), (Cam::Top, "Top", "3"), (Cam::Persp, "Persp", "4")] {
        if ui.selectable_label(view.cam == cam, label).on_hover_text(key).clicked() {
            view.cam = cam;
        }
    }
    ui.separator();
    ui.checkbox(&mut view.overlays.wireframe, "Wireframe").on_hover_text("W");
    ui.checkbox(&mut view.overlays.grid, "Grid").on_hover_text("G");
    ui.checkbox(&mut view.overlays.selection, "Selection").on_hover_text("S");
    egui::ComboBox::from_id_salt("filter")
        .selected_text(match view.filter {
            None => "Texture filter",
            Some(Filter::Nearest) => "Nearest",
            Some(Filter::Linear) => "Linear",
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut view.filter, None, "Texture filter");
            ui.selectable_value(&mut view.filter, Some(Filter::Nearest), "Nearest");
            ui.selectable_value(&mut view.filter, Some(Filter::Linear), "Linear");
        })
        .response
        .on_hover_text("F");
    if ui.button("Reset view").on_hover_text("R").clicked() {
        view.reset_camera();
    }
}

fn overview(ui: &mut egui::Ui, view: &mut ViewState, gpu: &mut Gpu, workspace: &Workspace) -> Result<bool> {
    let (snaps, busy) = workspace.snapshots(None, None, SCENE_WAIT)?;
    gpu.thumbs.retain(|name, _| workspace.names().contains(name));
    let names = workspace.names();
    let caption = 2.0 * ui.text_style_height(&egui::TextStyle::Body) + 12.0;
    let avail = ui.available_size();
    let ppp = ui.ctx().pixels_per_point();
    let (cols, tile_px) = layout(names.len(), (avail.x * ppp) as u32, ((avail.y - caption * 2.0) * ppp).max(64.0) as u32);
    // Leave room for each tile's button frame and the grid spacing.
    let per_tile_extra = 8.0 + 2.0 * ui.spacing().button_padding.x + 2.0;
    let fit_pts = (avail.x - cols as f32 * per_tile_extra) / cols as f32;
    let tile_px = tile_px.min((fit_pts * ppp).max(64.0) as u32).clamp(64, 1024);
    let tile_pts = tile_px as f32 / ppp;
    egui::ScrollArea::vertical().show(ui, |ui| -> Result<()> {
        egui::Grid::new("scenes").num_columns(cols as usize).spacing([8.0, 8.0]).show(ui, |ui| -> Result<()> {
            for (i, name) in names.iter().enumerate() {
                let snap = snaps.iter().find(|s| &s.name == name);
                if let Some(s) = snap {
                    refresh_thumb(ui.ctx(), gpu, s, tile_px)?;
                }
                ui.vertical(|ui| {
                    if let Some((_, _, tex)) = gpu.thumbs.get(name) {
                        let img = egui::Image::from_texture(SizedTexture::new(tex.id(), [tile_pts, tile_pts]));
                        if ui.add(egui::Button::image(img)).on_hover_text("open this scene").clicked() {
                            view.mode = Mode::Single(name.clone());
                            view.reset_camera();
                        }
                    } else {
                        ui.add_sized([tile_pts, tile_pts], egui::Spinner::new());
                    }
                    ui.strong(name);
                    ui.weak(match snap {
                        Some(s) if s.scene.objects.is_empty() => "(empty)".to_string(),
                        Some(s) => {
                            let tris: usize = s.scene.objects.values().map(|o| o.mesh.triangle_count()).sum();
                            format!("{} obj, {tris} tris", s.scene.objects.len())
                        }
                        None => "(busy)".to_string(),
                    });
                });
                if (i as u32 + 1) % cols == 0 {
                    ui.end_row();
                }
            }
            Ok(())
        })
        .inner
    })
    .inner?;
    Ok(busy.is_empty())
}

fn refresh_thumb(ctx: &egui::Context, gpu: &mut Gpu, s: &Snapshot, size: u32) -> Result<()> {
    if matches!(gpu.thumbs.get(&s.name), Some((v, sz, _)) if *v == s.version && *sz == size) {
        return Ok(());
    }
    let img = gpu.scene.render(&polly_core::anim::pose_scene(&s.scene, &[]), &tile_options(size))?.image;
    let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
    let tex = ctx.load_texture(format!("thumb:{}", s.name), color, egui::TextureOptions::LINEAR);
    gpu.thumbs.insert(s.name.clone(), (s.version, size, tex));
    Ok(())
}

fn single(ui: &mut egui::Ui, view: &mut ViewState, gpu: &mut Gpu, workspace: &Workspace, name: &str) -> Result<bool> {
    let Ok((snaps, _)) = workspace.snapshots(Some(&[name.to_string()]), None, SCENE_WAIT) else {
        // The scene was deleted.
        view.mode = Mode::Overview;
        return Ok(true);
    };
    let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
    if response.dragged() {
        let d = response.drag_delta();
        let (yaw, pitch) = view.cam.angles();
        view.cam = Cam::Orbit { yaw: yaw - d.x * 0.4, pitch: (pitch + d.y * 0.4).clamp(-89.0, 89.0) };
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            view.zoom = (view.zoom * 1.1f32.powf(scroll / 40.0)).clamp(0.1, 50.0);
        }
    }
    let ppp = ui.ctx().pixels_per_point();
    let size = [(rect.width() * ppp).round().max(1.0) as u32, (rect.height() * ppp).round().max(1.0) as u32];
    let Some(snap) = snaps.first() else {
        // Busy: keep showing the previous frame.
        if let Some(vp) = &gpu.viewport {
            ui.painter().image(vp.id, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
        }
        return Ok(false);
    };
    let (target, id) = viewport(gpu, size);
    let opts = RenderOptions {
        view: view.cam.view(),
        width: size[0],
        height: size[1],
        zoom: view.zoom,
        overlays: view.overlays.clone(),
        filter: view.filter,
        legend: false,
        ..RenderOptions::default()
    };
    let shown = polly_core::anim::pose_scene(&snap.scene, &[]);
    gpu.scene.render_to_view(&shown, &opts, &target, VIEWPORT_FORMAT, size)?;
    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    ui.painter().image(id, rect, uv, egui::Color32::WHITE);
    ui.painter().text(
        rect.left_top() + egui::vec2(8.0, 6.0),
        egui::Align2::LEFT_TOP,
        format!("{} ({})", view.cam.label(), if matches!(view.cam, Cam::Front | Cam::Right | Cam::Top) { "ortho" } else { "persp" }),
        egui::FontId::monospace(13.0),
        egui::Color32::from_gray(220),
    );
    ui.painter().text(
        rect.left_bottom() + egui::vec2(8.0, -6.0),
        egui::Align2::LEFT_BOTTOM,
        "drag: orbit   wheel: zoom",
        egui::FontId::proportional(12.0),
        egui::Color32::from_gray(150),
    );
    Ok(true)
}

/// The viewport texture, (re)created at `size` and registered with egui.
fn viewport(gpu: &mut Gpu, size: [u32; 2]) -> (wgpu::TextureView, egui::TextureId) {
    if gpu.viewport.as_ref().is_none_or(|v| v.size != size) {
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VIEWPORT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let id = match gpu.viewport.take() {
            Some(old) => {
                gpu.painter.update_egui_texture_from_wgpu_texture(&gpu.device, &view, wgpu::FilterMode::Nearest, old.id);
                old.id
            }
            None => gpu.painter.register_native_texture(&gpu.device, &view, wgpu::FilterMode::Nearest),
        };
        gpu.viewport = Some(Viewport { size, view, id });
    }
    let vp = gpu.viewport.as_ref().unwrap();
    (vp.view.clone(), vp.id)
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
                self.view.mode = scene.map_or(Mode::Overview, Mode::Single);
                self.view.reset_camera();
                let r = self.open(el).map_err(|e| format!("{e:#}"));
                self.seen = None;
                self.redraw();
                let _ = reply.send(r);
            }
            ViewerCommand::Hide { reply } => {
                self.gfx = None;
                let _ = reply.send(Ok(()));
            }
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        match &event {
            WindowEvent::CloseRequested => {
                self.gfx = None;
                return;
            }
            WindowEvent::Resized(size) => {
                gfx.config.width = size.width.max(1);
                gfx.config.height = size.height.max(1);
                gfx.surface.configure(&gfx.gpu.device, &gfx.config);
            }
            WindowEvent::RedrawRequested => {
                self.frame();
                return;
            }
            _ => {}
        }
        let r = gfx.state.on_window_event(&gfx.window, &event);
        if r.repaint {
            gfx.window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_none() {
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
pub fn run(el: EventLoop<ViewerCommand>, workspace: Arc<Workspace>, start_open: bool) -> Result<()> {
    let display = Box::new(Display(el.owned_display_handle()));
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(display));
    let mut app = App {
        workspace,
        instance,
        gfx: None,
        start_open,
        view: ViewState { mode: Mode::Overview, cam: Cam::Persp, zoom: 1.0, overlays: Overlays::default(), filter: None },
        seen: None,
        error: None,
    };
    el.run_app(&mut app)?;
    Ok(())
}
