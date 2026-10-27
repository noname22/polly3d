//! Back-end independent parts of a render: camera resolution, prepared
//! geometry, overlay lines, CPU annotation after readback and contact sheets.

use anyhow::Result;
use glam::Vec3;
use polly_core::reference::RefView;
use polly_core::scene::Scene;

use crate::annotate;
use crate::camera::{self, CameraInfo};
use crate::lines::{self, LineInst};
use crate::options::{RenderOptions, View};
use crate::prep::{Prepared, world_bounds};

/// Camera and geometry prepared for one render.
pub struct Setup {
    pub prep: Prepared,
    pub cam: CameraInfo,
    pub frame_radius: f32,
    pub notes: Vec<String>,
}

/// Prepare geometry and resolve the camera for a render of `size`.
pub fn setup(scene: &Scene, o: &RenderOptions, size: [u32; 2]) -> Setup {
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


/// Everything a back end needs to draw one image, and what annotation needs afterwards.
pub struct Frame {
    pub prep: Prepared,
    pub cam: CameraInfo,
    pub notes: Vec<String>,
    /// Half the diagonal of the framed bounds (sizes normals, depth biases, label tolerances).
    pub frame_radius: f32,
    /// Overlay lines and points (wireframe, grid, selection, bones...).
    pub lines: Vec<LineInst>,
    /// Image size in pixels.
    pub size: [u32; 2],
}

impl Frame {
    /// Depth pull of overlay lines toward the eye per unit of a line's bias:
    /// world units for orthographic cameras, a fraction of the eye distance otherwise.
    pub fn line_bias(&self) -> f32 {
        if self.cam.ortho { 0.01 * self.frame_radius } else { 0.004 }
    }
    /// Key and fill light directions (towards the lights), relative to the camera:
    /// key from the upper left front, fill from the lower right.
    pub fn light_dirs(&self) -> (Vec3, Vec3) {
        let (f, r, u) = (self.cam.forward(), self.cam.right(), self.cam.screen_up());
        ((-r * 0.55 + u * 0.75 - f * 0.45).normalize(), (r * 0.7 - u * 0.25 - f * 0.4).normalize())
    }
}

/// Prepare a render: camera, geometry and overlay lines, at the options' (clamped) size.
pub fn plan(scene: &Scene, o: &RenderOptions) -> Frame {
    plan_sized(scene, o, o.clamped_size())
}

/// [`plan`] at an explicit image size (e.g. a viewer window larger than the agent's limit).
pub fn plan_sized(scene: &Scene, o: &RenderOptions, size: [u32; 2]) -> Frame {
    let size = [size[0].max(1), size[1].max(1)];
    let Setup { prep, cam, frame_radius, notes } = setup(scene, o, size);
    let scale = (size[0].min(size[1]) as f32 / 512.0).clamp(0.75, 2.0);
    let fb = o.frame_bounds.map(|[lo, hi]| (Vec3::from(lo), Vec3::from(hi)));
    let lines = lines::build(scene, &prep, &o.overlays, scale, frame_radius, fb);
    Frame { prep, cam, notes, frame_radius, lines, size }
}

/// Turn a back end's image of `frame` into the final result: labels, legend, notes.
pub fn finish(mut image: image::RgbaImage, scene: &Scene, o: &RenderOptions, frame: &Frame) -> Rendered {
    let mut notes = frame.notes.clone();
    let labels = annotate::annotate(&mut image, scene, o, frame, &mut notes);
    Rendered { image, camera: frame.cam.clone(), labels, notes, tiles: Vec::new() }
}

/// 2x2 sheet: front, right, top, perspective, each rendered by `render`.
/// The total size is `o.width` x `o.height`.
pub fn contact_sheet(o: &RenderOptions, mut render: impl FnMut(&RenderOptions) -> Result<Rendered>) -> Result<Rendered> {
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
        let r = render(&to)?;
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
