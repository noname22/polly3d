//! wgpu renderer: shaded meshes + GPU overlay lines, offscreen or into a view.

use std::collections::{BTreeMap, HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use polly_core::reference::{Layer, RefView};
use polly_core::scene::{Filter, Material, Scene, TextureId};
use wgpu::util::DeviceExt;

use crate::annotate;
use crate::camera::{self, CameraInfo};
use crate::lines::{self, LineInst, srgb_to_linear};
use crate::options::{RenderOptions, View};
use crate::prep::{Prepared, world_bounds};

const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshVertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
    sel: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    forward: [f32; 4],
    key: [f32; 4],
    fill: [f32; 4],
    viewport: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    base: [f32; 4],
    params: [f32; 4],
}

pub(crate) struct Setup {
    pub prep: Prepared,
    pub cam: CameraInfo,
    pub frame_radius: f32,
    pub notes: Vec<String>,
}

/// Prepare geometry and resolve the camera for a render of `size`.
pub(crate) fn setup(scene: &Scene, o: &RenderOptions, size: [u32; 2]) -> Setup {
    let mut notes = Vec::new();
    let focus: Option<Vec<_>> = o.focus.as_ref().filter(|f| !f.is_empty()).cloned();
    if let Some(f) = &focus {
        for id in f {
            match scene.objects.get(id) {
                None => notes.push(format!("focus object o{id} does not exist")),
                Some(ob) if !ob.visible => notes.push(format!("focus object '{}' (o{id}) is hidden", ob.name)),
                _ => {}
            }
        }
    }
    let prep = Prepared::new(scene, if o.isolate { focus.as_deref() } else { None });
    let all_refs: Vec<_> = prep.objects.iter().collect();
    let focus_refs: Vec<_> = match &focus {
        Some(f) => prep.objects.iter().filter(|p| f.contains(&p.id)).collect(),
        None => all_refs.clone(),
    };
    let frame = match o.frame_bounds.or(o.region) {
        Some([lo, hi]) => (Vec3::from(lo), Vec3::from(hi)),
        None => {
            let objects = world_bounds(&focus_refs).or_else(|| world_bounds(&all_refs));
            // Unfocused axis views also frame their reference images.
            let refs = if focus.is_none() && o.overlays.references { reference_bounds(scene, &o.view) } else { None };
            // Shown bones count too (an armature often has no mesh of its own).
            let bones = if o.overlays.bones || o.overlays.bone_axes { bone_bounds(scene, focus.as_deref()) } else { None };
            union(union(objects, refs), bones).unwrap_or((Vec3::splat(-1.0), Vec3::splat(1.0)))
        }
    };
    if prep.objects.is_empty() {
        notes.push("scene has no visible objects".into());
    }
    let mut depth = world_bounds(&all_refs).unwrap_or(frame);
    if o.overlays.grid {
        let (ext, _) = lines::grid_extent(o.frame_bounds.map(|_| frame).or_else(|| world_bounds(&all_refs)));
        depth = (depth.0.min(Vec3::new(-ext, 0.0, -ext)), depth.1.max(Vec3::new(ext, 0.0, ext)));
    }
    let cam = camera::resolve(&o.view, frame, depth, o.zoom, size);
    let frame_radius = ((frame.1 - frame.0).length() * 0.5).max(0.05);
    Setup { prep, cam, frame_radius, notes }
}

fn union(a: Option<(Vec3, Vec3)>, b: Option<(Vec3, Vec3)>) -> Option<(Vec3, Vec3)> {
    match (a, b) {
        (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1))),
        (a, b) => a.or(b),
    }
}

/// World bounds of the bones of visible objects (only `focus` ones, if given).
fn bone_bounds(scene: &Scene, focus: Option<&[polly_core::scene::ObjectId]>) -> Option<(Vec3, Vec3)> {
    scene
        .objects
        .iter()
        .filter(|(id, o)| o.visible && focus.is_none_or(|f| f.contains(id)))
        .filter_map(|(id, o)| Some((scene.world_matrix(*id), o.skeleton.as_ref()?)))
        .flat_map(|(m, sk)| sk.bones.values().flat_map(move |b| [m.transform_point3(b.head), m.transform_point3(b.tail)]))
        .fold(None, |acc, p| union(acc, Some((p, p))))
}

/// The reference image view an axis view shows, if any.
pub fn reference_view(view: &crate::View) -> Option<RefView> {
    match view {
        crate::View::Front => Some(RefView::Front),
        crate::View::Back => Some(RefView::Back),
        crate::View::Left => Some(RefView::Left),
        crate::View::Right => Some(RefView::Right),
        crate::View::Top => Some(RefView::Top),
        crate::View::Bottom => Some(RefView::Bottom),
        _ => None,
    }
}

/// World bounds of the visible reference images `view` shows.
pub fn reference_bounds(scene: &Scene, view: &crate::View) -> Option<(Vec3, Vec3)> {
    let v = reference_view(view)?;
    scene
        .references
        .iter()
        .filter(|r| r.visible && r.view == v)
        .flat_map(|r| r.quad().map(|(p, _)| p))
        .fold(None, |acc, p| union(acc, Some((p, p))))
}

/// The camera `render` would use for these options (without rendering), e.g.
/// to `pick` on an earlier screenshot.
pub fn resolve_camera(scene: &Scene, o: &RenderOptions) -> CameraInfo {
    setup(scene, o, o.clamped_size()).cam
}

/// Result of an offscreen render.
#[derive(Debug, Clone)]
pub struct Rendered {
    pub image: image::RgbaImage,
    /// The camera used (for a contact sheet: the perspective tile's camera).
    pub camera: CameraInfo,
    /// Number of element / name labels drawn.
    pub labels: usize,
    /// Human/agent-readable remarks (label caps, hidden objects, ...).
    pub notes: Vec<String>,
    /// Contact sheet tiles (empty for a single render): each tile's camera and
    /// pixel offset, so picks on a sheet can be mapped back.
    pub tiles: Vec<Tile>,
}

#[derive(Debug, Clone)]
pub struct Tile {
    pub name: String,
    pub x: u32,
    pub y: u32,
    pub camera: CameraInfo,
}

struct Pipelines {
    mesh_cull: wgpu::RenderPipeline,
    mesh_double: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
    reference: wgpu::RenderPipeline,
}

/// Vertex of a reference image quad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RefVertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

struct GpuTexture {
    w: u32,
    h: u32,
    hash: u64,
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

struct Targets {
    size: [u32; 2],
    format: wgpu::TextureFormat,
    samples: u32,
    msaa: Option<wgpu::TextureView>,
    depth: wgpu::TextureView,
}

struct Offscreen {
    size: [u32; 2],
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    buffer: wgpu::Buffer,
    bytes_per_row: u32,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    shader: wgpu::ShaderModule,
    globals_bgl: wgpu::BindGroupLayout,
    material_bgl: wgpu::BindGroupLayout,
    layout_mesh: wgpu::PipelineLayout,
    layout_lines: wgpu::PipelineLayout,
    pipelines: HashMap<(wgpu::TextureFormat, u32), Pipelines>,
    sampler_nearest: wgpu::Sampler,
    sampler_linear: wgpu::Sampler,
    white: wgpu::TextureView,
    textures: HashMap<TextureId, GpuTexture>,
    /// Reference images by name, keyed by their pixel buffer's address
    /// (reference pixels never change in place).
    references: HashMap<String, (usize, GpuTexture)>,
    sampler_clamp: wgpu::Sampler,
    targets: Option<Targets>,
    offscreen: Option<Offscreen>,
}

/// What one GPU frame drew; used for CPU annotation afterwards.
pub(crate) struct Frame {
    pub prep: Prepared,
    pub cam: CameraInfo,
    pub notes: Vec<String>,
    pub frame_radius: f32,
}

impl Renderer {
    /// Create a headless renderer. Prefers a hardware adapter; set
    /// `POLLY_ADAPTER=software` to force the fallback (llvmpipe / lavapipe) adapter.
    pub fn headless() -> Result<Self> {
        let software = std::env::var("POLLY_ADAPTER").is_ok_and(|v| v.eq_ignore_ascii_case("software"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = if software {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: true,
                compatible_surface: None,
                ..Default::default()
            }))
            .ok()
            .or_else(|| {
                // Some drivers don't flag llvmpipe as a fallback adapter; look for a CPU device.
                pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
                    .into_iter()
                    .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
            })
            .context("POLLY_ADAPTER=software but no software (CPU) adapter is available")?
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            }))
            .or_else(|_| {
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    force_fallback_adapter: true,
                    ..Default::default()
                }))
            })
            .map_err(|e| anyhow!("no GPU adapter available: {e}"))?
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("polly-render"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            ..Default::default()
        }))
        .context("failed to create wgpu device")?;
        Ok(Self::new(&instance, &adapter, device, queue))
    }

    /// Build a renderer on an existing device (e.g. one shared with a window surface).
    pub fn new(_instance: &wgpu::Instance, adapter: &wgpu::Adapter, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polly shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout_mesh = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh layout"),
            bind_group_layouts: &[Some(&globals_bgl), Some(&material_bgl)],
            immediate_size: 0,
        });
        let layout_lines = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lines layout"),
            bind_group_layouts: &[Some(&globals_bgl)],
            immediate_size: 0,
        });
        let sampler = |f: wgpu::FilterMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: f,
                min_filter: f,
                ..Default::default()
            })
        };
        let sampler_nearest = sampler(wgpu::FilterMode::Nearest);
        let sampler_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reference sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler_linear = sampler(wgpu::FilterMode::Linear);
        let white_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("white"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            white_tex.as_image_copy(),
            &[255; 4],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: None },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white = white_tex.create_view(&Default::default());
        Renderer {
            device,
            queue,
            adapter: adapter.clone(),
            shader,
            globals_bgl,
            material_bgl,
            layout_mesh,
            layout_lines,
            pipelines: HashMap::new(),
            sampler_nearest,
            sampler_linear,
            white,
            textures: HashMap::new(),
            references: HashMap::new(),
            sampler_clamp,
            targets: None,
            offscreen: None,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    fn sample_count(&self, format: wgpu::TextureFormat) -> u32 {
        let c = self.adapter.get_texture_format_features(format).flags;
        let d = self.adapter.get_texture_format_features(DEPTH_FORMAT).flags;
        if c.sample_count_supported(4) && d.sample_count_supported(4) { 4 } else { 1 }
    }

    fn ensure_pipelines(&mut self, format: wgpu::TextureFormat, samples: u32) {
        let key = (format, samples);
        if !self.pipelines.contains_key(&key) {
            let p = self.create_pipelines(format, samples);
            self.pipelines.insert(key, p);
        }
    }

    fn create_pipelines(&self, format: wgpu::TextureFormat, samples: u32) -> Pipelines {
        let mesh_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4, 4 => Float32];
        let line_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x4];
        let ms = wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false };
        let mesh = |cull: Option<wgpu::Face>, label: &str| {
            self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&self.layout_mesh),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vs_mesh"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &mesh_attrs,
                    })],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: ms,
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some("fs_mesh"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let lines = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&self.layout_lines),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs_line"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineInst>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &line_attrs,
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: ms,
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs_line"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let ref_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];
        let reference = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("reference"),
            layout: Some(&self.layout_mesh),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs_ref"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RefVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &ref_attrs,
                })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: ms,
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs_ref"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Pipelines { mesh_cull: mesh(Some(wgpu::Face::Back), "mesh"), mesh_double: mesh(None, "mesh double-sided"), lines, reference }
    }

    fn ensure_targets(&mut self, format: wgpu::TextureFormat, size: [u32; 2]) -> u32 {
        let samples = self.sample_count(format);
        if self.targets.as_ref().is_some_and(|t| t.size == size && t.format == format && t.samples == samples) {
            return samples;
        }
        let ext = wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 };
        let mk = |format, label| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: ext,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let msaa = (samples > 1).then(|| mk(format, "msaa color"));
        let depth = mk(DEPTH_FORMAT, "depth");
        self.targets = Some(Targets { size, format, samples, msaa, depth });
        samples
    }

    fn texture_view(&mut self, scene: &Scene, id: TextureId, notes: &mut Vec<String>) -> Option<wgpu::TextureView> {
        let t = scene.textures.get(&id)?;
        if t.width == 0 || t.height == 0 || t.pixels.len() != (t.width * t.height * 4) as usize {
            notes.push(format!("texture t{id} has invalid dimensions/data; rendered white"));
            return None;
        }
        let max = self.device.limits().max_texture_dimension_2d;
        if t.width > max || t.height > max {
            notes.push(format!("texture t{id} exceeds GPU limit {max}; rendered white"));
            return None;
        }
        let mut h = DefaultHasher::new();
        t.pixels.hash(&mut h);
        let hash = h.finish();
        let ext = wgpu::Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 };
        let needs_new = self.textures.get(&id).is_none_or(|g| g.w != t.width || g.h != t.height);
        if needs_new {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scene texture"),
                size: ext,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.textures.insert(id, GpuTexture { w: t.width, h: t.height, hash: hash.wrapping_add(1), tex: tex, view });
        }
        let g = self.textures.get_mut(&id).unwrap();
        if g.hash != hash {
            self.queue.write_texture(
                g.tex.as_image_copy(),
                &t.pixels,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(t.width * 4), rows_per_image: None },
                ext,
            );
            g.hash = hash;
        }
        Some(g.view.clone())
    }

    /// GPU texture of a reference image (uploaded once per pixel buffer).
    fn reference_view(&mut self, r: &polly_core::reference::Reference, notes: &mut Vec<String>) -> Option<wgpu::TextureView> {
        if r.width == 0 || r.height == 0 || r.pixels.len() != (r.width * r.height * 4) as usize {
            notes.push(format!("reference '{}' has invalid image data", r.name));
            return None;
        }
        let max = self.device.limits().max_texture_dimension_2d;
        if r.width > max || r.height > max {
            notes.push(format!("reference '{}' exceeds GPU limit {max}", r.name));
            return None;
        }
        let key = std::sync::Arc::as_ptr(&r.pixels) as usize;
        if self.references.get(&r.name).is_none_or(|(k, g)| *k != key || g.w != r.width || g.h != r.height) {
            let ext = wgpu::Extent3d { width: r.width, height: r.height, depth_or_array_layers: 1 };
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("reference image"),
                size: ext,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                tex.as_image_copy(),
                &r.pixels,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(r.width * 4), rows_per_image: None },
                ext,
            );
            let view = tex.create_view(&Default::default());
            self.references.insert(r.name.clone(), (key, GpuTexture { w: r.width, h: r.height, hash: 0, tex, view }));
        }
        Some(self.references[&r.name].1.view.clone())
    }

    /// Resolve camera + geometry and encode GPU passes into `view`.
    fn draw(
        &mut self,
        scene: &Scene,
        o: &RenderOptions,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: [u32; 2],
    ) -> Result<Frame> {
        let Setup { prep, cam, frame_radius, mut notes } = setup(scene, o, size);

        // ---- mesh batches ----
        let default_mat = Material::new("missing", [0.8, 0.8, 0.8, 1.0]);
        let mut batches: BTreeMap<(u32, bool), Vec<MeshVertex>> = BTreeMap::new();
        for (oi, po) in prep.objects.iter().enumerate() {
            let obj = &scene.objects[&po.id];
            let sel = if o.overlays.selection { obj.selection.implied_faces(&po.mesh) } else { Default::default() };
            let vert_faces = po.mesh.vert_faces();
            for (&fid, f) in &po.mesh.faces {
                if !po.face_normal.contains_key(&fid) {
                    continue;
                }
                let mat = scene.materials.get(&f.material).unwrap_or(&default_mat);
                let normals = prep.corner_normals(oi, fid, &vert_faces);
                let s = if sel.contains(&fid) { 1.0 } else { 0.0 };
                let batch = batches.entry((f.material, mat.double_sided)).or_default();
                for tri in po.face_tris(f) {
                    for i in tri {
                        let c = &f.corners[i];
                        batch.push(MeshVertex {
                            pos: po.pos[&f.verts[i]].to_array(),
                            normal: normals[i].to_array(),
                            uv: c.uv.to_array(),
                            color: c.color.to_array(),
                            sel: s,
                        });
                    }
                }
            }
        }

        let scale = (size[0].min(size[1]) as f32 / 512.0).clamp(0.75, 2.0);
        let fb = o.frame_bounds.map(|[lo, hi]| (Vec3::from(lo), Vec3::from(hi)));
        let line_insts = lines::build(scene, &prep, &o.overlays, scale, frame_radius, fb);

        // ---- uniforms ----
        let f = cam.forward();
        let r = cam.right();
        let u = cam.screen_up();
        // Key light from upper-left-front of the camera, fill from the lower right.
        let key = (-r * 0.55 + u * 0.75 - f * 0.45).normalize();
        let fill = (r * 0.7 - u * 0.25 - f * 0.4).normalize();
        // Overlay lines are pulled towards the eye so they win the depth test
        // against the faces they lie on (fraction of eye distance, or world units for ortho).
        let bias = if cam.ortho { 0.01 * frame_radius } else { 0.004 };
        let encode = if format.is_srgb() { 0.0 } else { 1.0 };
        let globals = Globals {
            view_proj: cam.view_proj().to_cols_array_2d(),
            eye: [cam.eye[0], cam.eye[1], cam.eye[2], if cam.ortho { 1.0 } else { 0.0 }],
            forward: [f.x, f.y, f.z, bias],
            key: [key.x, key.y, key.z, 0.8],
            fill: [fill.x, fill.y, fill.z, 0.12],
            viewport: [size[0] as f32, size[1] as f32, encode, 0.0],
        };
        let globals_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let globals_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &self.globals_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });

        struct Draw {
            vb: wgpu::Buffer,
            count: u32,
            bg: wgpu::BindGroup,
            double: bool,
        }
        let mut draws = Vec::new();
        for ((mid, double), verts) in &batches {
            if verts.is_empty() {
                continue;
            }
            let mat = scene.materials.get(mid).unwrap_or(&default_mat);
            let (tview, filt) = match mat.texture {
                Some(tid) => match self.texture_view(scene, tid, &mut notes) {
                    Some(v) => (v, scene.textures[&tid].filter),
                    None => {
                        if !scene.textures.contains_key(&tid) {
                            notes.push(format!("material '{}' references missing texture t{tid}", mat.name));
                        }
                        (self.white.clone(), Filter::Nearest)
                    }
                },
                None => (self.white.clone(), Filter::Nearest),
            };
            let sampler = match o.filter.unwrap_or(filt) {
                Filter::Nearest => &self.sampler_nearest,
                Filter::Linear => &self.sampler_linear,
            };
            let mu = MaterialUniform {
                base: mat.base_color,
                params: [if mat.unlit { 1.0 } else { 0.0 }, mat.alpha_cutoff.unwrap_or(-1.0), 0.0, 0.0],
            };
            let ub = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material"),
                contents: bytemuck::bytes_of(&mu),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material"),
                layout: &self.material_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tview) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            });
            let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertices"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            draws.push(Draw { vb, count: verts.len() as u32, bg, double: *double });
        }
        // ---- reference images (only in their own orthographic view) ----
        struct RefDraw {
            vb: wgpu::Buffer,
            bg: wgpu::BindGroup,
            front: bool,
        }
        let mut ref_draws = Vec::new();
        let shown_view = RefView::ALL.into_iter().find(|v| {
            let (u, up) = v.axes();
            cam.ortho && f.dot(-u.cross(up)) > 0.999
        });
        if o.overlays.references {
            for rf in scene.references.iter().filter(|r| r.visible && Some(r.view) == shown_view && o.reference_opacity.unwrap_or(r.opacity) > 0.0) {
                let Some(tview) = self.reference_view(rf, &mut notes) else { continue };
                let q = rf.quad();
                let verts: Vec<RefVertex> =
                    [0, 1, 2, 0, 2, 3].iter().map(|&i| RefVertex { pos: q[i].0.to_array(), uv: q[i].1 }).collect();
                let opacity = o.reference_opacity.unwrap_or(rf.opacity).clamp(0.0, 1.0);
                let mu = MaterialUniform { base: [1.0, 1.0, 1.0, opacity], params: [0.0; 4] };
                let ub = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("reference"),
                    contents: bytemuck::bytes_of(&mu),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("reference"),
                    layout: &self.material_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tview) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler_clamp) },
                    ],
                });
                let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("reference quad"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                ref_draws.push(RefDraw { vb, bg, front: o.reference_layer.unwrap_or(rf.layer) == Layer::Front });
            }
        }
        self.references.retain(|name, _| scene.references.iter().any(|r| &r.name == name));
        let line_buf = (!line_insts.is_empty()).then(|| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lines"),
                contents: bytemuck::cast_slice(&line_insts),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });

        // ---- passes ----
        let samples = self.ensure_targets(format, size);
        self.ensure_pipelines(format, samples);
        let pipes = &self.pipelines[&(format, samples)];
        let targets = self.targets.as_ref().unwrap();
        let bg = o.background.unwrap_or(scene.background);
        let clear = if format.is_srgb() {
            wgpu::Color {
                r: srgb_to_linear(bg[0]) as f64,
                g: srgb_to_linear(bg[1]) as f64,
                b: srgb_to_linear(bg[2]) as f64,
                a: bg[3] as f64,
            }
        } else {
            wgpu::Color { r: bg[0] as f64, g: bg[1] as f64, b: bg[2] as f64, a: bg[3] as f64 }
        };
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("polly frame") });
        {
            let (color_view, resolve) = match &targets.msaa {
                Some(m) => (m, Some(view)),
                None => (view, None),
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: if resolve.is_some() { wgpu::StoreOp::Discard } else { wgpu::StoreOp::Store },
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &globals_bg, &[]);
            let draw_refs = |pass: &mut wgpu::RenderPass, front: bool| {
                for d in ref_draws.iter().filter(|d| d.front == front) {
                    pass.set_pipeline(&pipes.reference);
                    pass.set_bind_group(1, &d.bg, &[]);
                    pass.set_vertex_buffer(0, d.vb.slice(..));
                    pass.draw(0..6, 0..1);
                }
            };
            draw_refs(&mut pass, false);
            for d in &draws {
                pass.set_pipeline(if d.double { &pipes.mesh_double } else { &pipes.mesh_cull });
                pass.set_bind_group(1, &d.bg, &[]);
                pass.set_vertex_buffer(0, d.vb.slice(..));
                pass.draw(0..d.count, 0..1);
            }
            if let Some(lb) = &line_buf {
                pass.set_pipeline(&pipes.lines);
                pass.set_bind_group(0, &globals_bg, &[]);
                pass.set_vertex_buffer(0, lb.slice(..));
                pass.draw(0..6, 0..line_insts.len() as u32);
            }
            draw_refs(&mut pass, true);
        }
        self.queue.submit([enc.finish()]);
        Ok(Frame { prep, cam, notes, frame_radius })
    }

    /// Render into a caller-provided texture view (live viewer path). GPU passes
    /// only: shading, wireframe, selection, grid, normals... but no text labels.
    pub fn render_to_view(
        &mut self,
        scene: &Scene,
        o: &RenderOptions,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: [u32; 2],
    ) -> Result<()> {
        let size = [size[0].max(1), size[1].max(1)];
        self.draw(scene, o, view, format, size)?;
        Ok(())
    }

    /// Render offscreen and read back to an image, with CPU-drawn labels and legend.
    pub fn render(&mut self, scene: &Scene, o: &RenderOptions) -> Result<Rendered> {
        let size = o.clamped_size();
        self.ensure_offscreen(size);
        let view = self.offscreen.as_ref().unwrap().view.clone();
        let frame = self.draw(scene, o, &view, OFFSCREEN_FORMAT, size)?;
        let mut image = self.read_back()?;
        let mut notes = frame.notes.clone();
        let labels = annotate::annotate(&mut image, scene, o, &frame, &mut notes);
        Ok(Rendered { image, camera: frame.cam, labels, notes, tiles: Vec::new() })
    }

    fn ensure_offscreen(&mut self, size: [u32; 2]) {
        if self.offscreen.as_ref().is_some_and(|o| o.size == size) {
            return;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let bytes_per_row = (size[0] * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * size[1]) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.offscreen = Some(Offscreen { size, tex, view, buffer, bytes_per_row });
    }

    fn read_back(&mut self) -> Result<image::RgbaImage> {
        let off = self.offscreen.as_ref().unwrap();
        let [w, h] = off.size;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_texture_to_buffer(
            off.tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &off.buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(off.bytes_per_row), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        off.buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("GPU poll failed: {e}"))?;
        rx.recv().context("readback callback dropped")?.map_err(|e| anyhow!("buffer map failed: {e}"))?;
        let mut img = image::RgbaImage::new(w, h);
        {
            let data = off.buffer.get_mapped_range(..).map_err(|e| anyhow!("map range failed: {e:?}"))?;
            let row = (w * 4) as usize;
            for y in 0..h as usize {
                let src = &data[y * off.bytes_per_row as usize..][..row];
                img.as_mut()[y * row..(y + 1) * row].copy_from_slice(src);
            }
        }
        off.buffer.unmap();
        // The image is opaque regardless of the background alpha used for the clear.
        Ok(img)
    }

    /// 2x2 sheet: front, right, top, perspective — each captioned. The total
    /// size is `o.width` x `o.height`.
    pub fn contact_sheet(&mut self, scene: &Scene, o: &RenderOptions) -> Result<Rendered> {
        let [w, h] = o.clamped_size();
        let (tw, th) = ((w / 2).max(64), (h / 2).max(64));
        let mut sheet = image::RgbaImage::new(tw * 2, th * 2);
        let mut labels = 0;
        let mut notes = Vec::new();
        let mut tiles = Vec::new();
        let mut persp_cam = None;
        for (i, v) in [View::Front, View::Right, View::Top, View::Persp].into_iter().enumerate() {
            let mut to = o.clone();
            to.view = v;
            to.width = tw;
            to.height = th;
            let r = self.render(scene, &to)?;
            let (x, y) = ((i as u32 % 2) * tw, (i as u32 / 2) * th);
            image::imageops::replace(&mut sheet, &r.image, x as i64, y as i64);
            labels += r.labels;
            for n in r.notes {
                if !notes.contains(&n) {
                    notes.push(n);
                }
            }
            if i == 3 {
                persp_cam = Some(r.camera.clone());
            }
            tiles.push(Tile { name: to.view.name().to_string(), x, y, camera: r.camera });
        }
        // Tile separators.
        let sep = [235, 235, 240, 255];
        for y in 0..sheet.height() {
            sheet.put_pixel(tw - 1, y, image::Rgba(sep));
            sheet.put_pixel(tw, y, image::Rgba(sep));
        }
        for x in 0..sheet.width() {
            sheet.put_pixel(x, th - 1, image::Rgba(sep));
            sheet.put_pixel(x, th, image::Rgba(sep));
        }
        notes.push(format!("contact sheet tiles ({tw}x{th}): front top-left, right top-right, top bottom-left, persp bottom-right"));
        Ok(Rendered { image: sheet, camera: persp_cam.unwrap(), labels, notes, tiles })
    }
}
