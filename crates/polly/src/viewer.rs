//! Live viewer: an egui window mirroring the scenes while the agent works.
//! The event loop runs for the whole process, but the window only exists
//! while the agent has opened it (`show_viewer`) and the user hasn't closed it.
//!
//! With several scenes the window starts on an overview of thumbnails;
//! clicking one opens that scene alone, and "< Back" (or Backspace/Esc)
//! returns. Scenes are rendered by the shared Bevy render service (the same
//! one the agent's renders use) and shown as egui textures.
//!
//! Single-scene view: left-drag orbits, wheel zooms. Keys: 1 front, 2 right,
//! 3 top, 4 persp, W wireframe, G grid, S selection, F cycle texture filter
//! (texture / nearest / linear), R reset camera.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use egui::load::SizedTexture;
use polly_core::reference::{Layer, RefView, Reference};
use polly_core::scene::{Filter, Scene};

use crate::anim_panel::{self, AnimState};
use polly_mcp::overview::{layout, tile_options};
use polly_mcp::services::ViewerCommand;
use polly_mcp::workspace::{MAIN, Snapshot, Workspace};
use polly_render::{Overlays, RenderOptions, View};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::raw_window_handle::{DisplayHandle, HandleError, HasDisplayHandle};
use winit::window::{Window, WindowId};

/// How long a frame may wait for a scene that a tool call is busy with.
const SCENE_WAIT: Duration = Duration::from_millis(20);

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
    Back,
    Left,
    Right,
    Top,
    Bottom,
    Persp,
    Orbit { yaw: f32, pitch: f32 },
}

impl Cam {
    fn view(self) -> View {
        match self {
            Cam::Front => View::Front,
            Cam::Back => View::Back,
            Cam::Left => View::Left,
            Cam::Right => View::Right,
            Cam::Top => View::Top,
            Cam::Bottom => View::Bottom,
            // Same angle as View::Persp but orbit framing, so the first drag doesn't jump.
            Cam::Persp => View::Orbit { yaw_deg: 35.0, pitch_deg: 25.0, distance: None },
            Cam::Orbit { yaw, pitch } => View::Orbit { yaw_deg: yaw, pitch_deg: pitch, distance: None },
        }
    }

    fn angles(self) -> (f32, f32) {
        match self {
            Cam::Orbit { yaw, pitch } => (yaw, pitch),
            Cam::Front => (0.0, 0.0),
            Cam::Back => (180.0, 0.0),
            Cam::Left => (-90.0, 0.0),
            Cam::Right => (90.0, 0.0),
            Cam::Top => (0.0, 89.0),
            Cam::Bottom => (0.0, -89.0),
            Cam::Persp => (35.0, 25.0),
        }
    }

    fn label(self) -> String {
        match self {
            Cam::Front => "front".into(),
            Cam::Back => "back".into(),
            Cam::Left => "left".into(),
            Cam::Right => "right".into(),
            Cam::Top => "top".into(),
            Cam::Bottom => "bottom".into(),
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
    /// Animation playback per scene name.
    anim: HashMap<String, AnimState>,
    /// Variant previews per scene name (slot -> variant), overriding the scene's active ones.
    variants: HashMap<String, std::collections::BTreeMap<String, String>>,
    mode: Mode,
    cam: Cam,
    zoom: f32,
    overlays: Overlays,
    filter: Option<Filter>,
    /// Reference image settings and the user's own references, per scene name.
    refs: HashMap<String, RefState>,
    /// An open "Add reference" file dialog: (scene, view, picked file).
    ref_pick: Option<(String, polly_core::reference::RefView, std::sync::mpsc::Receiver<Option<std::path::PathBuf>>)>,
}

impl ViewState {
    fn reset_camera(&mut self) {
        self.cam = Cam::Persp;
        self.zoom = 1.0;
    }
}

/// GPU resources the UI code needs while egui builds a frame.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The process-wide Bevy render service.
    scene: polly_bevy::BevyRenderer,
    painter: egui_wgpu::Renderer,
    /// The last rendered single-scene view.
    viewport: Option<egui::TextureHandle>,
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
        let scene = polly_bevy::BevyRenderer::global()?;
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
        ui.horizontal_wrapped(|ui| match &mode {
            Mode::Overview => {
                ui.strong(format!("{} scenes", workspace.names().len()));
                ui.label("— click one to open it");
            }
            Mode::Single(name) => toolbar(ui, view, name, several),
        });
    });
    // Single-scene mode: snapshot the scene once for the panel and the view.
    let mut shown: Option<Scene> = None;
    let mut rest_bounds = None;
    if let Mode::Single(name) = &mode {
        let Ok((snaps, _)) = workspace.snapshots(Some(std::slice::from_ref(name)), None, SCENE_WAIT) else {
            // The scene was deleted.
            view.mode = Mode::Overview;
            return Ok(true);
        };
        if let Some(snap) = snaps.into_iter().next() {
            // The viewer never edits the scene: everything the user changes here
            // is a local override applied to this copy.
            let mut scene = snap.scene;
            let agent_refs = scene.references.clone();
            view.anim.entry(name.clone()).or_default().apply_loops(&mut scene);
            view.refs.entry(name.clone()).or_default().apply(&mut scene);
            // Gear variants: preview choices in a side panel.
            let choice = view.variants.entry(name.clone()).or_default();
            choice.retain(|slot, v| scene.slots.get(slot).is_some_and(|s| s.variants.iter().any(|x| &x.name == v)));
            if !scene.slots.is_empty() {
                egui::Panel::right("variants").show(ui, |ui| variants_panel(ui, &scene, choice));
            }
            let chosen = polly_core::variants::variant_view(&scene, choice);
            if !scene.references.is_empty() {
                egui::Panel::right("references").show(ui, |ui| references_panel(ui, view, name, &agent_refs));
            }
            incoming_references(ui, view, name);
            let st = view.anim.entry(name.clone()).or_default();
            let dt = ui.input(|i| i.stable_dt).min(0.1);
            st.advance(&scene, dt);
            if ui.input(|i| i.key_pressed(egui::Key::Space)) && !scene.clips.is_empty() {
                st.toggle_play(&scene);
            }
            if AnimState::has_content(&scene) {
                egui::Panel::bottom("animation").show(ui, |ui| {
                    ui.add_space(4.0);
                    anim_panel::panel(ui, st, &scene)
                });
            }
            if st.playing() {
                ui.ctx().request_repaint();
            }
            // Frame the rest pose so the camera holds still during playback,
            // plus the reference images of the current view.
            let refs = if view.overlays.references { polly_render::reference_bounds(&chosen, &view.cam.view()) } else { None };
            rest_bounds = match (chosen.bounds(), refs) {
                (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1))),
                (a, b) => a.or(b),
            }
            .map(|(lo, hi)| [lo.to_array(), hi.to_array()]);
            shown = Some(st.display(&chosen));
        }
    }
    let mut complete = true;
    egui::CentralPanel::default().show(ui, |ui| -> Result<()> {
        match &mode {
            Mode::Overview => complete = overview(ui, view, gpu, workspace)?,
            Mode::Single(_) => complete = single(ui, view, gpu, shown.as_ref(), rest_bounds)?,
        }
        Ok(())
    })
    .inner?;
    Ok(complete)
}

/// One dropdown per gear slot; choices are previews layered over the scene's active variants.
fn variants_panel(ui: &mut egui::Ui, scene: &Scene, choice: &mut std::collections::BTreeMap<String, String>) {
    ui.add_space(4.0);
    ui.strong("Variants");
    ui.add_space(4.0);
    for (slot, s) in &scene.slots {
        let current = choice.get(slot).cloned().unwrap_or_else(|| s.active.clone());
        ui.label(slot);
        egui::ComboBox::from_id_salt(("variant", slot)).selected_text(&current).show_ui(ui, |ui| {
            for v in &s.variants {
                let label = if v.name == s.default { format!("{} (default)", v.name) } else { v.name.clone() };
                if ui.selectable_label(current == v.name, label).clicked() {
                    choice.insert(slot.clone(), v.name.clone());
                }
            }
        });
        ui.add_space(4.0);
    }
    if !choice.is_empty() && ui.small_button("reset").on_hover_text("back to the scene's active variants").clicked() {
        choice.clear();
    }
}

/// Viewer-only reference settings for one scene.
#[derive(Default)]
struct RefState {
    /// Overrides of the agent's references, by name.
    overrides: std::collections::BTreeMap<String, RefOverride>,
    /// References the user added in the viewer (never seen by the agent).
    mine: Vec<Reference>,
}

#[derive(Default, Clone, Copy, PartialEq)]
struct RefOverride {
    visible: Option<bool>,
    flip: Option<bool>,
    layer: Option<Layer>,
    opacity: Option<f32>,
}

impl RefState {
    /// Lay the user's settings and references over the scene's (a display copy).
    fn apply(&mut self, scene: &mut Scene) {
        // Forget overrides of references the agent removed.
        self.overrides.retain(|n, _| scene.references.iter().any(|r| &r.name == n));
        for r in &mut scene.references {
            if let Some(o) = self.overrides.get(&r.name) {
                r.visible = o.visible.unwrap_or(r.visible);
                r.flip = o.flip.unwrap_or(r.flip);
                r.layer = o.layer.unwrap_or(r.layer);
                r.opacity = o.opacity.unwrap_or(r.opacity);
            }
        }
        for m in &self.mine {
            let mut m = m.clone();
            // Names key the GPU cache: keep them apart from the agent's.
            m.name = format!("{} (mine)", m.name);
            scene.references.push(m);
        }
    }
}

fn ref_cam(v: RefView) -> Cam {
    match v {
        RefView::Front => Cam::Front,
        RefView::Back => Cam::Back,
        RefView::Left => Cam::Left,
        RefView::Right => Cam::Right,
        RefView::Top => Cam::Top,
        RefView::Bottom => Cam::Bottom,
    }
}

/// Visible checkbox and view button, layer toggle and opacity slider for one
/// reference, editing `visible`, `layer` and `opacity` in place.
#[allow(clippy::too_many_arguments)]
fn ref_controls(ui: &mut egui::Ui, cam: &mut Cam, label: &str, view: RefView, visible: &mut bool, layer: &mut Layer, opacity: &mut f32, flip: &mut bool) {
    ui.horizontal(|ui| {
        ui.checkbox(visible, label);
        let c = ref_cam(view);
        if ui.selectable_label(*cam == c, c.label()).on_hover_text("show this view").clicked() {
            *cam = c;
        }
    });
    ui.horizontal(|ui| {
        for (l, text) in [(Layer::Behind, "Behind"), (Layer::Front, "On top")] {
            if ui.selectable_label(*layer == l, text).clicked() {
                *layer = l;
            }
        }
        ui.checkbox(flip, "Flip").on_hover_text("mirror the image left/right");
    });
    ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("opacity"));
}

/// Reference images: the agent's (with viewer-only overrides) and the user's own.
fn references_panel(ui: &mut egui::Ui, view: &mut ViewState, name: &str, agent: &[Reference]) {
    let st = view.refs.entry(name.to_string()).or_default();
    let cam = &mut view.cam;
    ui.add_space(4.0);
    ui.strong("References");
    ui.weak("Changes here affect only this window.");
    ui.add_space(4.0);
    for r in agent {
        let o = st.overrides.entry(r.name.clone()).or_default();
        let (mut visible, mut layer, mut opacity) = (o.visible.unwrap_or(r.visible), o.layer.unwrap_or(r.layer), o.opacity.unwrap_or(r.opacity));
        let mut flip = o.flip.unwrap_or(r.flip);
        ref_controls(ui, cam, &r.name, r.view, &mut visible, &mut layer, &mut opacity, &mut flip);
        o.flip = (flip != r.flip).then_some(flip);
        // Keep only what differs from the agent's settings.
        o.visible = (visible != r.visible).then_some(visible);
        o.layer = (layer != r.layer).then_some(layer);
        o.opacity = (opacity != r.opacity).then_some(opacity);
        if *o != RefOverride::default() && ui.small_button("Reset").on_hover_text("back to the agent's settings").clicked() {
            *o = RefOverride::default();
        }
        ui.separator();
    }
    if !st.mine.is_empty() {
        ui.strong("Mine");
        ui.weak("Not visible to the agent.");
        let mut remove = None;
        for (i, r) in st.mine.iter_mut().enumerate() {
            ref_controls(ui, cam, &r.name, r.view, &mut r.visible, &mut r.layer, &mut r.opacity, &mut r.flip);
            if ui.small_button("Remove").clicked() {
                remove = Some(i);
            }
            ui.separator();
        }
        if let Some(i) = remove {
            st.mine.remove(i);
        }
    }
}

/// The reference view for the current camera (Front when looking in perspective).
fn current_ref_view(view: &mut ViewState) -> RefView {
    match view.cam {
        Cam::Back => RefView::Back,
        Cam::Left => RefView::Left,
        Cam::Right => RefView::Right,
        Cam::Top => RefView::Top,
        Cam::Bottom => RefView::Bottom,
        _ => {
            view.cam = Cam::Front;
            RefView::Front
        }
    }
}

/// Open the system file chooser (xdg portal on Linux) without blocking the viewer.
fn pick_reference(ui: &egui::Ui, view: &mut ViewState, scene: &str) {
    if view.ref_pick.is_some() {
        return;
    }
    let rv = current_ref_view(view);
    let (tx, rx) = std::sync::mpsc::channel();
    let ctx = ui.ctx().clone();
    std::thread::spawn(move || {
        let file = rfd::FileDialog::new()
            .set_title("Add reference image")
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tga", "tif", "tiff"])
            .pick_file();
        tx.send(file).ok();
        ctx.request_repaint();
    });
    view.ref_pick = Some((scene.to_string(), rv, rx));
}

/// Load an image as one of the user's own references (viewer only).
fn add_my_reference(view: &mut ViewState, name: &str, path: &std::path::Path, rv: RefView) {
    let st = view.refs.entry(name.to_string()).or_default();
    let stem = path.file_stem().map_or("reference".into(), |s| s.to_string_lossy().into_owned());
    let mut unique = stem.clone();
    let mut n = 2;
    while st.mine.iter().any(|r| r.name == unique) {
        unique = format!("{stem}_{n}");
        n += 1;
    }
    match polly_mcp::tools::reference::load_reference(&path.to_string_lossy(), rv, Some(&unique)) {
        Ok(r) => st.mine.push(r),
        Err(e) => tracing::warn!("viewer: {e:#}"),
    }
}

/// New references from the file dialog, or image files dropped on the window
/// (X11/macOS/Windows; winit has no drag-and-drop on Wayland). They go to the
/// current axis view.
fn incoming_references(ui: &egui::Ui, view: &mut ViewState, name: &str) {
    if let Some((sc, rv, rx)) = &view.ref_pick {
        match rx.try_recv() {
            Ok(file) => {
                let (sc, rv) = (sc.clone(), *rv);
                view.ref_pick = None;
                if let Some(path) = file {
                    add_my_reference(view, &sc, &path, rv);
                }
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => view.ref_pick = None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }
    let dropped: Vec<std::path::PathBuf> = ui.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
    for path in dropped {
        let rv = current_ref_view(view);
        add_my_reference(view, name, &path, rv);
    }
}

fn shortcuts(ui: &mut egui::Ui, view: &mut ViewState, mode: &Mode, several: bool) {
    use egui::Key;
    let single = matches!(mode, Mode::Single(_));
    ui.input(|i| {
        if single && several && (i.key_pressed(Key::Backspace) || i.key_pressed(Key::Escape)) {
            view.mode = Mode::Overview;
        }
        if single {
            // Ctrl + 1/2/3 looks from the opposite side (as in Blender).
            for (key, cam, opposite) in [(Key::Num1, Cam::Front, Cam::Back), (Key::Num2, Cam::Right, Cam::Left), (Key::Num3, Cam::Top, Cam::Bottom)] {
                if i.key_pressed(key) {
                    view.cam = if i.modifiers.ctrl { opposite } else { cam };
                }
            }
            if i.key_pressed(Key::Num4) {
                view.cam = Cam::Persp;
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
    for (cam, label, key) in [
        (Cam::Front, "Front", "1"),
        (Cam::Back, "Back", "Ctrl+1"),
        (Cam::Right, "Right", "2"),
        (Cam::Left, "Left", "Ctrl+2"),
        (Cam::Top, "Top", "3"),
        (Cam::Bottom, "Bottom", "Ctrl+3"),
        (Cam::Persp, "Persp", "4"),
    ] {
        if ui.selectable_label(view.cam == cam, label).on_hover_text(key).clicked() {
            view.cam = cam;
        }
    }
    ui.separator();
    ui.checkbox(&mut view.overlays.wireframe, "Wireframe").on_hover_text("W");
    ui.checkbox(&mut view.overlays.grid, "Grid").on_hover_text("G");
    ui.checkbox(&mut view.overlays.selection, "Selection").on_hover_text("S");
    ui.checkbox(&mut view.overlays.references, "Refs").on_hover_text("reference images (in their axis views)");
    if ui.add_enabled(view.ref_pick.is_none(), egui::Button::new("Add reference…")).on_hover_text("your own image for the current axis view (Front when in perspective), shown only here (or drop a file on the window)").clicked() {
        pick_reference(ui, view, name);
    }
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
    let img = gpu.scene.render(&polly_core::variants::display(&s.scene, &[], &Default::default()), &tile_options(size))?.image;
    let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
    let tex = ctx.load_texture(format!("thumb:{}", s.name), color, egui::TextureOptions::LINEAR);
    gpu.thumbs.insert(s.name.clone(), (s.version, size, tex));
    Ok(())
}

fn single(ui: &mut egui::Ui, view: &mut ViewState, gpu: &mut Gpu, shown: Option<&Scene>, frame_bounds: Option<[[f32; 3]; 2]>) -> Result<bool> {
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
    let Some(scene) = shown else {
        // Busy: keep showing the previous frame.
        if let Some(vp) = &gpu.viewport {
            ui.painter().image(vp.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
        }
        return Ok(false);
    };
    let opts = RenderOptions {
        view: view.cam.view(),
        width: size[0],
        height: size[1],
        zoom: view.zoom,
        overlays: view.overlays.clone(),
        filter: view.filter,
        legend: false,
        frame_bounds,
        ..RenderOptions::default()
    };
    let img = gpu.scene.render_sized(scene, &opts, size)?.image;
    let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
    match &mut gpu.viewport {
        Some(t) => t.set(color, egui::TextureOptions::NEAREST),
        None => gpu.viewport = Some(ui.ctx().load_texture("viewport", color, egui::TextureOptions::NEAREST)),
    }
    let id = gpu.viewport.as_ref().unwrap().id();
    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    ui.painter().image(id, rect, uv, egui::Color32::WHITE);
    ui.painter().text(
        rect.left_top() + egui::vec2(8.0, 6.0),
        egui::Align2::LEFT_TOP,
        format!("{} ({})", view.cam.label(), if matches!(view.cam, Cam::Front | Cam::Back | Cam::Left | Cam::Right | Cam::Top | Cam::Bottom) { "ortho" } else { "persp" }),
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
        view: ViewState { anim: HashMap::new(), variants: HashMap::new(), mode: Mode::Overview, cam: Cam::Persp, zoom: 1.0, overlays: Overlays::default(), filter: None, refs: HashMap::new(), ref_pick: None },
        seen: None,
        error: None,
    };
    el.run_app(&mut app)?;
    Ok(())
}
