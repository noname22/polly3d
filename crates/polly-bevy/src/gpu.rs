//! The Bevy side of the render service: one headless app, scene -> entity
//! sync with caches, camera/lights, readback. Runs on the service's worker thread.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use bevy::app::{PanicHandlerPlugin, SubApps, TerminalCtrlCHandlerPlugin};
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ClearColorConfig, Exposure, RenderTarget};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::anti_alias::fxaa::Fxaa;
use bevy::camera::Hdr;
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::light::{CascadeShadowConfigBuilder, EnvironmentMapLight, Skybox};
use bevy::pbr::ScreenSpaceAmbientOcclusion;
use bevy::post_process::bloom::Bloom;
use polly_core::lighting::Tonemap;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::light::{AmbientLight, NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::RenderAdapterInfo;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::ExitCondition;
use polly_core::mesh::Mesh as PMesh;
use polly_core::reference::{Layer, RefView};
use polly_core::scene::{Filter, Material as PMaterial, ObjectId, Scene, Texture};
use polly_render::{Frame, RenderOptions};

use crate::conv::{m4, v3};
use crate::env;
use bevy::pbr::ExtendedMaterial;
use bevy_polly::{PollyMaterial, PollyShader};
use crate::lines::{self, LineMaterial};
use crate::projection::PollyProjection;

/// Cache entries unused for this many jobs are dropped.
const EVICT_AFTER: u64 = 64;

/// Exposure so that 1 lux of illuminance gives an output value of 1.0
/// (Bevy: exposure = 2^-ev100 / 1.2).
const EV100_UNIT: f32 = -0.263_034_4;

/// Today's lighting: intensities as in the old shader (key 0.8, fill 0.12,
/// hemisphere ambient 0.12 below .. 0.30 above).
const KEY: f32 = 0.8;
const FILL: f32 = 0.12;
const AMBIENT: f32 = 0.19;
const SKY: f32 = 0.13;

/// One render request handled on the worker thread.
pub struct Job {
    pub scene: Scene,
    pub opts: RenderOptions,
    pub frame: Arc<Frame>,
}

/// A batch of an object's faces sharing a material: one mesh entity.
struct Batch {
    material: u32,
    entity: Entity,
    mesh: Handle<Mesh>,
}

struct ObjectEntry {
    /// Keeps the mesh alive so its address can't be reused while cached.
    _mesh: Arc<PMesh>,
    batches: Vec<Batch>,
    last_used: u64,
}

struct Cached<T> {
    value: T,
    /// Holds pixel buffers alive (address reuse) for texture entries.
    _keep: Option<Arc<Vec<u8>>>,
    last_used: u64,
}

/// A material of either kind: plain PBR or PBR with a custom shader.
#[derive(Clone)]
enum MatHandle {
    Standard(Handle<StandardMaterial>),
    Custom(Handle<PollyMaterial>),
}

/// A custom shader loaded from the asset directory.
struct ShaderFile {
    name: String,
    handle: Handle<Shader>,
    stamp: Option<std::time::SystemTime>,
}

/// Prefiltered cubemaps of one environment.
#[derive(Clone)]
pub struct EnvMaps {
    diffuse: Handle<Image>,
    specular: Handle<Image>,
    skybox: Option<Handle<Image>>,
}

pub struct Gpu {
    apps: SubApps,
    pub adapter: String,
    job: u64,
    /// Key, fill, sky (classic view lights) and sun.
    lights: [Entity; 4],
    environments: HashMap<String, EnvMaps>,
    targets: HashMap<[u32; 2], Handle<Image>>,
    objects: HashMap<u64, ObjectEntry>,
    materials: HashMap<u64, Cached<MatHandle>>,
    /// Custom shaders by source path.
    shaders: HashMap<std::path::PathBuf, ShaderFile>,
    /// Remarks for the current job (missing shaders, compile errors).
    notes: Vec<String>,
    images: HashMap<u64, Cached<Handle<Image>>>,
    line_material: Handle<LineMaterial>,
    /// Entities and meshes that live for one job only (lines, selection, references).
    transient: Vec<(Entity, Option<Handle<Mesh>>)>,
}

fn hash_of(h: impl Hash) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    h.hash(&mut s);
    s.finish()
}

fn world_bits(m: polly_core::glam::Mat4) -> [u32; 16] {
    m.to_cols_array().map(f32::to_bits)
}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let software = std::env::var("POLLY_ADAPTER").is_ok_and(|v| v.eq_ignore_ascii_case("software"));
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin { primary_window: None, exit_condition: ExitCondition::DontExit, ..default() })
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        force_fallback_adapter: software,
                        ..default()
                    })),
                    ..default()
                })
                .disable::<TerminalCtrlCHandlerPlugin>()
                .disable::<PanicHandlerPlugin>()
                .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>(),
        );
        // The shader handle must exist before the material plugin asks for it.
        let shader = app.world_mut().resource_mut::<Assets<Shader>>().add(Shader::from_wesl(lines::LINE_SHADER, "polly://line.wesl"));
        lines::set_shader(shader);
        app.add_plugins(MaterialPlugin::<LineMaterial>::default());
        app.add_plugins(MaterialPlugin::<PollyMaterial>::default());
        app.finish();
        app.cleanup();
        let mut apps = std::mem::take(app.sub_apps_mut());
        let world = apps.main.world_mut();
        let adapter = world.get_resource::<RenderAdapterInfo>().map(|a| a.name.clone()).ok_or_else(|| anyhow::anyhow!("Bevy found no GPU adapter"))?;
        let light = |world: &mut World| {
            world.spawn((DirectionalLight { illuminance: 0.0, shadow_maps_enabled: false, ..default() }, Transform::default())).id()
        };
        let lights = [light(world), light(world), light(world), light(world)];
        let line_material = world.resource_mut::<Assets<LineMaterial>>().add(LineMaterial::default());
        let _ = &mut apps;
        Ok(Gpu {
            apps,
            adapter,
            job: 0,
            lights,
            environments: HashMap::new(),
            targets: HashMap::new(),
            objects: HashMap::new(),
            materials: HashMap::new(),
            shaders: HashMap::new(),
            notes: Vec::new(),
            images: HashMap::new(),
            line_material,
            transient: Vec::new(),
        })
    }

    pub fn stats(&self) -> crate::Stats {
        crate::Stats {
            jobs: self.job,
            objects: self.objects.len(),
            meshes: self.objects.values().map(|e| e.batches.len()).sum(),
            materials: self.materials.len(),
            images: self.images.len(),
        }
    }

    /// Cubemaps for an environment spec ("studio", "sky" or an .hdr path),
    /// built once per spec (and file modification time).
    fn environment(&mut self, spec: &str, skybox: bool) -> Result<EnvMaps> {
        let stamp = std::fs::metadata(spec).and_then(|m| m.modified()).ok();
        let key = format!("{spec}|{stamp:?}");
        let missing = match self.environments.get(&key) {
            None => true,
            Some(e) => skybox && e.skybox.is_none(),
        };
        if missing {
            let src = env::Source::load(spec)?;
            let sky_img = skybox.then(|| env::skybox(&src));
            let (d, sp) = (env::diffuse(&src), env::specular(&src));
            if self.environments.len() >= 4 {
                let old: Vec<EnvMaps> = self.environments.drain().map(|(_, v)| v).collect();
                let images = &mut self.world().resource_mut::<Assets<Image>>();
                for e in old {
                    images.remove(&e.diffuse);
                    images.remove(&e.specular);
                    if let Some(s) = e.skybox {
                        images.remove(&s);
                    }
                }
            }
            self.environments.remove(&key);
            let images = &mut self.world().resource_mut::<Assets<Image>>();
            let maps = EnvMaps { diffuse: images.add(d), specular: images.add(sp), skybox: sky_img.map(|i| images.add(i)) };
            self.environments.insert(key.clone(), maps);
        }
        Ok(self.environments[&key].clone())
    }

    fn world(&mut self) -> &mut World {
        self.apps.main.world_mut()
    }

    /// Render one job and read the image back (sRGB RGBA8, unannotated).
    /// Render one job; returns the image and remarks for the agent.
    pub fn render(&mut self, job: &Job) -> Result<(image::RgbaImage, Vec<String>)> {
        self.job += 1;
        self.notes.clear();
        let [w, h] = job.frame.size;
        let scene = &job.scene;
        let frame = &job.frame;
        let o = &job.opts;

        // ---- meshes ----
        let mut shown: Vec<Entity> = Vec::new();
        for (oi, po) in frame.prep.objects.iter().enumerate() {
            let level = polly_core::subdiv::display_level(scene, po.id, o.subdivision);
            let key = hash_of((self.object_key(scene, po.id, &po.mesh), level));
            if !self.objects.contains_key(&key) {
                let entry = if level > 0 { self.build_subdivided(scene, frame, oi, level)? } else { self.build_object(scene, frame, oi)? };
                self.objects.insert(key, entry);
            }
            let entry = self.objects.get_mut(&key).unwrap();
            entry.last_used = self.job;
            let batches: Vec<(u32, Entity)> = entry.batches.iter().map(|b| (b.material, b.entity)).collect();
            for (mid, e) in batches {
                let mat = self.material(scene, mid, o)?;
                let mut ent = self.world().entity_mut(e);
                match mat {
                    MatHandle::Standard(h) => {
                        ent.remove::<MeshMaterial3d<PollyMaterial>>().insert(MeshMaterial3d(h));
                    }
                    MatHandle::Custom(h) => {
                        ent.remove::<MeshMaterial3d<StandardMaterial>>().insert(MeshMaterial3d(h));
                    }
                }
                shown.push(e);
            }
        }
        let job_no = self.job;
        let mut evict = Vec::new();
        for (k, entry) in &self.objects {
            if job_no - entry.last_used > EVICT_AFTER {
                evict.push(*k);
            }
        }
        for k in evict {
            let entry = self.objects.remove(&k).unwrap();
            for b in entry.batches {
                self.world().despawn(b.entity);
                self.world().resource_mut::<Assets<Mesh>>().remove(&b.mesh);
            }
        }
        let all: Vec<Entity> = self.objects.values().flat_map(|e| e.batches.iter().map(|b| b.entity)).collect();
        for e in all {
            let v = if shown.contains(&e) { Visibility::Visible } else { Visibility::Hidden };
            self.world().entity_mut(e).insert(v);
        }
        self.materials.retain(|_, c| job_no - c.last_used <= EVICT_AFTER);
        self.images.retain(|_, c| job_no - c.last_used <= EVICT_AFTER);

        // ---- per-job overlays ----
        if o.overlays.selection {
            self.selection_overlay(scene, frame);
        }
        if !frame.lines.is_empty() {
            let mesh = lines::mesh(&frame.lines, frame.line_bias());
            let mh = self.world().resource_mut::<Assets<Mesh>>().add(mesh);
            let lm = self.line_material.clone();
            let e = self.world().spawn((Mesh3d(mh.clone()), MeshMaterial3d(lm), Transform::IDENTITY, NoFrustumCulling, NotShadowCaster, NotShadowReceiver)).id();
            self.transient.push((e, Some(mh)));
        }
        if o.overlays.references {
            self.references(scene, frame, o);
        }

        // ---- camera, lights and effects ----
        let (target, fresh) = self.target([w, h]);
        let lighting = o.lighting.clone().unwrap_or_else(|| scene.lighting.clone());
        let envs_before = self.environments.len();
        let env = match &lighting.environment {
            Some(spec) => Some(self.environment(spec, lighting.show_environment)?),
            None => None,
        };
        // New cubemaps only reach the GPU after an update, like new targets.
        let fresh = fresh || self.environments.len() != envs_before;
        let cam = frame.cam.clone();
        let (near, far) = (cam.near, cam.far);
        let bg = o.background.unwrap_or(scene.background);
        let view = m4(cam.view_matrix());
        let cam_transform = Transform::from_matrix(view.inverse());
        let (key, fill) = frame.light_dirs();
        let lights = self.lights;
        let world = self.world();
        let mut camera = world.spawn((
            Camera3d::default(),
            Camera {
                is_active: true,
                clear_color: ClearColorConfig::Custom(Color::srgba(bg[0], bg[1], bg[2], bg[3])),
                ..default()
            },
            RenderTarget::Image(target.clone().into()),
            Projection::custom(PollyProjection { cam }),
            cam_transform,
            DebandDither::Disabled,
            Exposure { ev100: EV100_UNIT - lighting.exposure },
        ));
        let view_lights = lighting.view_lights_on();
        camera.insert(AmbientLight {
            color: Color::WHITE,
            brightness: if view_lights { AMBIENT } else { 0.0 },
            affects_lightmapped_meshes: true,
        });
        camera.insert(match lighting.tonemapping {
            Tonemap::None => Tonemapping::None,
            Tonemap::Agx => Tonemapping::AgX,
            Tonemap::TonyMcMapface => Tonemapping::TonyMcMapface,
            Tonemap::AcesFitted => Tonemapping::AcesFitted,
            Tonemap::Reinhard => Tonemapping::Reinhard,
            Tonemap::BlenderFilmic => Tonemapping::BlenderFilmic,
        });
        if lighting.needs_hdr() {
            camera.insert(Hdr);
        }
        if lighting.bloom > 0.0 {
            camera.insert(Bloom { intensity: lighting.bloom, ..Bloom::NATURAL });
        }
        if lighting.ambient_occlusion {
            // SSAO needs the depth/normal prepass, which doesn't support MSAA: use FXAA instead.
            camera.insert((ScreenSpaceAmbientOcclusion::default(), Msaa::Off, Fxaa::default()));
        } else {
            camera.insert(Msaa::Sample4);
        }
        if let Some(env) = &env {
            let rotation = Quat::from_rotation_y(lighting.environment_rotation.to_radians());
            camera.insert(EnvironmentMapLight {
                diffuse_map: env.diffuse.clone(),
                specular_map: env.specular.clone(),
                intensity: lighting.environment_intensity,
                rotation,
                affects_lightmapped_mesh_diffuse: true,
            });
            if let Some(sky) = &env.skybox {
                camera.insert(Skybox { image: Some(sky.clone()), brightness: lighting.environment_intensity, rotation });
            }
        }
        let cam_entity = camera.id();
        let pi = std::f32::consts::PI;
        let shadows = |on: bool| {
            on.then(|| CascadeShadowConfigBuilder { num_cascades: 1, minimum_distance: near, maximum_distance: far, first_cascade_far_bound: far, overlap_proportion: 0.2 }.build())
        };
        let sun = lighting.sun.map(|s| (Vec3::from_array(s.direction).normalize_or(Vec3::Y), s.intensity * KEY, Color::linear_rgb(s.color[0], s.color[1], s.color[2])));
        let rig = [
            (lights[0], v3(key), if view_lights { KEY } else { 0.0 }, Color::WHITE, lighting.shadows && view_lights),
            (lights[1], v3(fill), if view_lights { FILL } else { 0.0 }, Color::WHITE, false),
            (lights[2], Vec3::Y, if view_lights { SKY } else { 0.0 }, Color::WHITE, false),
            (lights[3], sun.map_or(Vec3::Y, |s| s.0), sun.map_or(0.0, |s| s.1), sun.map_or(Color::WHITE, |s| s.2), lighting.shadows && sun.is_some()),
        ];
        for (e, dir, lux, color, cast) in rig {
            let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
            let mut ent = world.entity_mut(e);
            ent.insert((
                DirectionalLight { illuminance: lux * pi, color, shadow_maps_enabled: cast, ..default() },
                Transform::IDENTITY.looking_to(-dir, up),
            ));
            if let Some(c) = shadows(cast) {
                ent.insert(c);
            }
        }

        self.transient.push((cam_entity, None));

        // ---- render + readback ----
        // A new target only exists on the GPU after one update: render into it once first.
        if fresh {
            self.apps.update();
        }
        let world = self.world();
        let out: Arc<Mutex<Option<Image>>> = Default::default();
        let slot = out.clone();
        world.spawn(Screenshot::image(target)).observe(move |ev: On<ScreenshotCaptured>| {
            *slot.lock().unwrap() = Some(ev.image.clone());
        });
        let mut updates = 0;
        while out.lock().unwrap().is_none() {
            self.apps.update();
            updates += 1;
            if updates > 12 {
                bail!("the renderer produced no image");
            }
        }
        // Clean up per-job entities.
        for (e, m) in std::mem::take(&mut self.transient) {
            self.world().despawn(e);
            if let Some(m) = m {
                self.world().resource_mut::<Assets<Mesh>>().remove(&m);
            }
        }
        let img = out.lock().unwrap().take().unwrap();
        let data = img.data.ok_or_else(|| anyhow::anyhow!("screenshot without data"))?;
        let size = img.texture_descriptor.size;
        if size.width != w || size.height != h || data.len() != (w * h * 4) as usize {
            bail!("unexpected readback size {}x{} ({} bytes)", size.width, size.height, data.len());
        }
        let mut rgba = image::RgbaImage::from_raw(w, h, data).ok_or_else(|| anyhow::anyhow!("readback buffer"))?;
        // The image is opaque regardless of the background alpha.
        for p in rgba.pixels_mut() {
            p.0[3] = 255;
        }
        self.shader_errors();
        Ok((rgba, std::mem::take(&mut self.notes)))
    }

    /// Load (or reload, when the file changed) a custom shader; `None` with a
    /// note when it can't be used.
    fn custom_shader(&mut self, scene: &Scene, name: &str) -> Option<()> {
        let Some(path) = scene.shader_path(name) else {
            self.note(format!("material shader '{name}' needs the asset directory (write_shader sets it); shown without it"));
            return None;
        };
        let stamp = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        if stamp.is_none() {
            self.note(format!("shader file {} not found; '{name}' shown without its shader", path.display()));
            return None;
        }
        let current = self.shaders.get(&path).is_some_and(|f| f.stamp == stamp);
        if !current {
            let src = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    self.note(format!("cannot read {}: {e}", path.display()));
                    return None;
                }
            };
            let shader = Shader::from_wesl(src, format!("polly://shaders/{name}.wesl"));
            let existing = self.shaders.get(&path).map(|f| f.handle.clone());
            let handle = match existing {
                // Same handle, new source: Bevy recompiles the pipelines using it (hot reload).
                Some(h) => {
                    let _ = self.world().resource_mut::<Assets<Shader>>().insert(h.id(), shader);
                    h
                }
                None => self.world().resource_mut::<Assets<Shader>>().add(shader),
            };
            bevy_polly::register_shader(name, handle.clone());
            self.shaders.insert(path, ShaderFile { name: name.to_string(), handle, stamp });
        }
        Some(())
    }

    fn note(&mut self, n: String) {
        if !self.notes.contains(&n) {
            self.notes.push(n);
        }
    }

    /// Compile errors of custom shaders, as notes.
    fn shader_errors(&mut self) {
        use bevy::render::render_resource::{CachedPipelineState, PipelineCache, PipelineDescriptor};
        let names: HashMap<AssetId<Shader>, String> = self.shaders.values().map(|f| (f.handle.id(), f.name.clone())).collect();
        if names.is_empty() {
            return;
        }
        let mut errors = Vec::new();
        if let Some(render) = self.apps.sub_apps.values().find(|a| a.world().contains_resource::<PipelineCache>()) {
            let cache = render.world().resource::<PipelineCache>();
            for p in cache.pipelines() {
                let CachedPipelineState::Err(err) = &p.state else { continue };
                let PipelineDescriptor::RenderPipelineDescriptor(d) = &p.descriptor else { continue };
                let Some(f) = &d.fragment else { continue };
                if let Some(name) = names.get(&f.shader.id()) {
                    errors.push(format!("shader '{name}' failed to compile (faces using it are not drawn): {err}"));
                }
            }
        }
        for e in errors {
            self.note(e);
        }
    }

    /// The render target for `size`; `true` when it was just created.
    fn target(&mut self, size: [u32; 2]) -> (Handle<Image>, bool) {
        if let Some(h) = self.targets.get(&size) {
            return (h.clone(), false);
        }
        let mut img = Image::new_target_texture(size[0], size[1], TextureFormat::Rgba8UnormSrgb, None);
        img.asset_usage = RenderAssetUsages::RENDER_WORLD;
        img.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let h = self.world().resource_mut::<Assets<Image>>().add(img);
        if self.targets.len() > 8 {
            let old: Vec<_> = self.targets.drain().collect();
            for (_, t) in old {
                self.world().resource_mut::<Assets<Image>>().remove(&t);
            }
        }
        self.targets.insert(size, h.clone());
        (h, true)
    }

    /// Cache key of an object's mesh entities: object, mesh identity, world
    /// placement and how its materials split/shade it.
    fn object_key(&self, scene: &Scene, id: ObjectId, mesh: &Arc<PMesh>) -> u64 {
        let mats: std::collections::BTreeSet<u32> = mesh.faces.values().map(|f| f.material).collect();
        let sig: Vec<(u32, bool, bool)> = mats
            .iter()
            .map(|m| {
                let mat = scene.materials.get(m);
                (*m, mat.is_some_and(|m| m.double_sided), mat.is_some_and(|m| m.needs_tangents()))
            })
            .collect();
        hash_of((id, Arc::as_ptr(mesh) as usize, world_bits(scene.world_matrix(id)), sig))
    }

    /// Entities for the subdivided surface of an object (the cage stays in
    /// the frame for overlays, labels and picking).
    fn build_subdivided(&mut self, scene: &Scene, frame: &Frame, oi: usize, level: u8) -> Result<ObjectEntry> {
        let po = &frame.prep.objects[oi];
        let surface = polly_core::subdiv::subdivided(&po.mesh, level);
        let m = scene.world_matrix(po.id);
        let nm = m.inverse().transpose();
        let normals = surface.corner_normals();
        let mut batches: BTreeMap<u32, Vec<([f32; 3], [f32; 3], [f32; 2], [f32; 4])>> = BTreeMap::new();
        for (&fid, f) in &surface.faces {
            let b = batches.entry(f.material).or_default();
            let n = f.len();
            for k in 1..n - 1 {
                let tri = if po.flip { [0, k + 1, k] } else { [0, k, k + 1] };
                for i in tri {
                    let c = &f.corners[i];
                    let p = m.transform_point3(surface.verts[&f.verts[i]].pos);
                    let nn = nm.transform_vector3(normals[&(fid, i)]).normalize_or_zero();
                    b.push((p.to_array(), nn.to_array(), c.uv.to_array(), c.color.to_array()));
                }
            }
        }
        let mut out = Vec::new();
        for (mid, verts) in batches {
            if verts.is_empty() {
                continue;
            }
            let n = verts.len() as u32;
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, verts.iter().map(|v| v.0).collect::<Vec<_>>())
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, verts.iter().map(|v| v.1).collect::<Vec<_>>())
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, verts.iter().map(|v| v.2).collect::<Vec<_>>())
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, verts.iter().map(|v| v.3).collect::<Vec<_>>())
                .with_inserted_indices(Indices::U32((0..n).collect()));
            if scene.materials.get(&mid).is_some_and(|m| m.needs_tangents()) {
                let _ = mesh.generate_tangents();
            }
            let world = self.world();
            let mh = world.resource_mut::<Assets<Mesh>>().add(mesh);
            let e = world.spawn((Mesh3d(mh.clone()), Transform::IDENTITY, NoFrustumCulling, Visibility::Hidden)).id();
            out.push(Batch { material: mid, entity: e, mesh: mh });
        }
        Ok(ObjectEntry { _mesh: po.mesh.clone(), batches: out, last_used: self.job })
    }

    /// World-space mesh entities for one prepared object, one per material.
    fn build_object(&mut self, scene: &Scene, frame: &Frame, oi: usize) -> Result<ObjectEntry> {
        let po = &frame.prep.objects[oi];
        let vert_faces = po.mesh.vert_faces();
        #[derive(Default)]
        struct B {
            pos: Vec<[f32; 3]>,
            nrm: Vec<[f32; 3]>,
            uv: Vec<[f32; 2]>,
            col: Vec<[f32; 4]>,
        }
        let mut batches: BTreeMap<u32, B> = BTreeMap::new();
        for (&fid, f) in &po.mesh.faces {
            if !po.face_normal.contains_key(&fid) {
                continue;
            }
            let normals = frame.prep.corner_normals(oi, fid, &vert_faces);
            let b = batches.entry(f.material).or_default();
            for tri in po.face_tris(f) {
                for i in tri {
                    let c = &f.corners[i];
                    b.pos.push(po.pos[&f.verts[i]].to_array());
                    b.nrm.push(normals[i].to_array());
                    b.uv.push(c.uv.to_array());
                    b.col.push(c.color.to_array());
                }
            }
        }
        let mut out = Vec::new();
        for (mid, b) in batches {
            if b.pos.is_empty() {
                continue;
            }
            let n = b.pos.len() as u32;
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, b.pos)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, b.nrm)
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, b.uv)
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, b.col)
                .with_inserted_indices(Indices::U32((0..n).collect()));
            if scene.materials.get(&mid).is_some_and(|m| m.needs_tangents()) {
                // Degenerate UVs make mikktspace fail; the normal map then just doesn't show.
                let _ = mesh.generate_tangents();
            }
            let world = self.world();
            let mh = world.resource_mut::<Assets<Mesh>>().add(mesh);
            let e = world.spawn((Mesh3d(mh.clone()), Transform::IDENTITY, NoFrustumCulling, Visibility::Hidden)).id();
            out.push(Batch { material: mid, entity: e, mesh: mh });
        }
        Ok(ObjectEntry { _mesh: po.mesh.clone(), batches: out, last_used: self.job })
    }

    fn image(&mut self, t: &Texture, tid: u32, filter: Filter, clamp: bool) -> Handle<Image> {
        let key = hash_of((tid, Arc::as_ptr(&t.pixels) as usize, t.width, t.height, filter as u8, t.linear, clamp));
        let job = self.job;
        if let Some(c) = self.images.get_mut(&key) {
            c.last_used = job;
            return c.value.clone();
        }
        let format = if t.linear { TextureFormat::Rgba8Unorm } else { TextureFormat::Rgba8UnormSrgb };
        let mut img = Image::new(
            Extent3d { width: t.width.max(1), height: t.height.max(1), depth_or_array_layers: 1 },
            TextureDimension::D2,
            t.pixels.to_vec(),
            format,
            RenderAssetUsages::RENDER_WORLD,
        );
        let f = match filter {
            Filter::Nearest => ImageFilterMode::Nearest,
            Filter::Linear => ImageFilterMode::Linear,
        };
        let address = if clamp { ImageAddressMode::ClampToEdge } else { ImageAddressMode::Repeat };
        img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            mag_filter: f,
            min_filter: f,
            address_mode_u: address,
            address_mode_v: address,
            address_mode_w: address,
            ..default()
        });
        let h = self.world().resource_mut::<Assets<Image>>().add(img);
        self.images.insert(key, Cached { value: h.clone(), _keep: Some(t.pixels.clone()), last_used: job });
        h
    }

    fn material(&mut self, scene: &Scene, mid: u32, o: &RenderOptions) -> Result<MatHandle> {
        let missing = PMaterial::new("missing", [0.8, 0.8, 0.8, 1.0]);
        let m = scene.materials.get(&mid).unwrap_or(&missing).clone();
        let tex_ptrs: Vec<(u32, usize, bool)> = m
            .textures()
            .filter_map(|(_, t)| scene.textures.get(&t).map(|x| (t, Arc::as_ptr(&x.pixels) as usize, x.linear)))
            .collect();
        let custom = match &m.shader {
            Some(sh) => self.custom_shader(scene, &sh.name).map(|_| sh.clone()),
            None => None,
        };
        let key = hash_of((mid, serde_json::to_string(&m).unwrap_or_default(), tex_ptrs, o.filter.map(|f| f as u8), custom.is_some()));
        let job = self.job;
        if let Some(c) = self.materials.get_mut(&key) {
            c.last_used = job;
            return Ok(c.value.clone());
        }
        let tex = |slot: Option<u32>, gpu: &mut Gpu| -> Option<Handle<Image>> {
            let tid = slot?;
            let t = scene.textures.get(&tid)?;
            if t.width == 0 || t.height == 0 || t.pixels.len() != (t.width * t.height * 4) as usize {
                return None;
            }
            Some(gpu.image(t, tid, o.filter.unwrap_or(t.filter), false))
        };
        let c = m.base_color;
        let e = m.emissive.map(|x| x * m.emissive_strength.max(0.0));
        let mut sm = StandardMaterial {
            base_color: Color::linear_rgba(c[0], c[1], c[2], c[3]),
            base_color_texture: tex(m.texture, self),
            emissive: LinearRgba::rgb(e[0], e[1], e[2]),
            emissive_texture: tex(m.emissive_texture, self),
            perceptual_roughness: m.roughness,
            metallic: m.metallic,
            metallic_roughness_texture: tex(m.metallic_roughness_texture, self),
            normal_map_texture: tex(m.normal_texture, self),
            occlusion_texture: tex(m.occlusion_texture, self),
            double_sided: m.double_sided,
            cull_mode: if m.double_sided { None } else { Some(bevy::render::render_resource::Face::Back) },
            unlit: m.unlit,
            alpha_mode: match (m.alpha_cutoff, m.blend) {
                (Some(c), _) => AlphaMode::Mask(c),
                (None, true) => AlphaMode::Blend,
                _ => AlphaMode::Opaque,
            },
            ..default()
        };
        if let Some(cc) = m.clearcoat {
            sm.clearcoat = cc.factor;
            sm.clearcoat_perceptual_roughness = cc.roughness;
        }
        if let Some(a) = m.anisotropy {
            sm.anisotropy_strength = a.strength;
            sm.anisotropy_rotation = a.rotation;
        }
        if let Some(t) = m.transmission {
            sm.specular_transmission = t.factor;
            sm.ior = t.ior;
            sm.thickness = t.thickness;
        }
        if let Some(s) = m.specular {
            sm.reflectance = s.factor * 0.5;
            sm.specular_tint = Color::linear_rgb(s.color[0], s.color[1], s.color[2]);
        }
        let h = match custom {
            Some(sh) => MatHandle::Custom(
                self.world().resource_mut::<Assets<PollyMaterial>>().add(ExtendedMaterial { base: sm, extension: PollyShader::new(&sh.name, &sh.params) }),
            ),
            None => MatHandle::Standard(self.world().resource_mut::<Assets<StandardMaterial>>().add(sm)),
        };
        self.materials.insert(key, Cached { value: h.clone(), _keep: None, last_used: job });
        Ok(h)
    }

    /// Selected faces tinted orange (50 %), drawn over the shaded faces.
    fn selection_overlay(&mut self, scene: &Scene, frame: &Frame) {
        let mut pos: Vec<[f32; 3]> = Vec::new();
        for po in &frame.prep.objects {
            let obj = &scene.objects[&po.id];
            if obj.selection.is_empty() {
                continue;
            }
            for fid in obj.selection.implied_faces(&po.mesh) {
                let Some(f) = po.mesh.faces.get(&fid) else { continue };
                if !po.face_normal.contains_key(&fid) {
                    continue;
                }
                for tri in po.face_tris(f) {
                    for i in tri {
                        pos.push(po.pos[&f.verts[i]].to_array());
                    }
                }
            }
        }
        if pos.is_empty() {
            return;
        }
        let n = pos.len() as u32;
        let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
            .with_inserted_indices(Indices::U32((0..n).collect()));
        let mat = StandardMaterial {
            base_color: Color::linear_rgba(1.0, 0.33, 0.02, 0.5),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            depth_bias: 100.0,
            ..default()
        };
        let world = self.world();
        let mh = world.resource_mut::<Assets<Mesh>>().add(mesh);
        let mat = world.resource_mut::<Assets<StandardMaterial>>().add(mat);
        let e = world.spawn((Mesh3d(mh.clone()), MeshMaterial3d(mat), Transform::IDENTITY, NoFrustumCulling, NotShadowCaster, NotShadowReceiver)).id();
        self.transient.push((e, Some(mh)));
    }

    /// Reference images of the current axis view: quads just inside the far
    /// plane (behind) or the near plane (on top). In an orthographic view the
    /// depth doesn't change where they land on screen.
    fn references(&mut self, scene: &Scene, frame: &Frame, o: &RenderOptions) {
        let cam = &frame.cam;
        if !cam.ortho {
            return;
        }
        let f = cam.forward();
        let shown = RefView::ALL.into_iter().find(|v| {
            let (u, up) = v.axes();
            f.dot(-u.cross(up)) > 0.999
        });
        let eye = cam.eye_v();
        let span = cam.far - cam.near;
        for (i, rf) in scene.references.iter().enumerate() {
            let opacity = o.reference_opacity.unwrap_or(rf.opacity).clamp(0.0, 1.0);
            if !rf.visible || Some(rf.view) != shown || opacity <= 0.0 {
                continue;
            }
            if rf.width == 0 || rf.height == 0 || rf.pixels.len() != (rf.width * rf.height * 4) as usize {
                continue;
            }
            let front = o.reference_layer.unwrap_or(rf.layer) == Layer::Front;
            // Several references keep their order (later ones nearer).
            let k = (i as f32 + 1.0) * 1e-4 * span;
            let depth = if front { cam.near + span * 0.002 + k } else { cam.far - span * 0.002 - (scene.references.len() as f32 * 1e-4 * span) + k };
            let q = rf.quad();
            let pos: Vec<[f32; 3]> = q.iter().map(|(p, _)| (*p + f * (depth - (*p - eye).dot(f))).to_array()).collect();
            let uv: Vec<[f32; 2]> = q.iter().map(|(_, uv)| *uv).collect();
            let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![(-f).to_array(); 4])
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv)
                .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
            let tex = Texture {
                name: rf.name.clone(),
                width: rf.width,
                height: rf.height,
                pixels: rf.pixels.clone(),
                filter: Filter::Linear,
                palette: None,
                linear: false,
            };
            let img = self.image(&tex, u32::MAX - i as u32, Filter::Linear, true);
            let mat = StandardMaterial {
                base_color: Color::linear_rgba(1.0, 1.0, 1.0, opacity),
                base_color_texture: Some(img),
                unlit: true,
                alpha_mode: AlphaMode::Blend,
                double_sided: true,
                cull_mode: None,
                // Draw order: behind refs first, on-top refs last (after the overlay lines).
                depth_bias: if front { 1.0e7 } else { -1.0e7 },
                ..default()
            };
            let world = self.world();
            let mh = world.resource_mut::<Assets<Mesh>>().add(mesh);
            let mat = world.resource_mut::<Assets<StandardMaterial>>().add(mat);
            let e = world.spawn((Mesh3d(mh.clone()), MeshMaterial3d(mat), Transform::IDENTITY, NoFrustumCulling, NotShadowCaster, NotShadowReceiver)).id();
            self.transient.push((e, Some(mh)));
        }
    }
}
