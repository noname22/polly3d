use std::collections::BTreeMap;
use std::io::Cursor;

use anyhow::{Context, Result, bail};
use polly_core::scene::{Filter, ObjectId, Scene};
use polly_paint::{Blend, Rgba};
use polly_render::{CameraInfo, Overlays, RenderOptions, Rendered, View};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};
use crate::services::LastView;

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "render",
            "Render a screenshot. Views: persp (default 3/4), front, back, left, right, top, bottom, iso, orbit {yaw_deg,pitch_deg}, custom {eye,target}. Overlays: wireframe, face_ids, vert_ids, edge_ids, verts, selection, seams, sharp, normals, grid, bounds, object_names. Use face_ids/vert_ids to learn element ids before editing; `focus` + `zoom` to get close.",
            true,
            render,
        ),
        def(
            "render_contact_sheet",
            "Render front, right, top and perspective views in one 2x2 image (default 1024x1024) — the quickest way to check proportions from all sides.",
            true,
            contact_sheet,
        ),
        def(
            "render_uv_layout",
            "Show an object's UV layout: the texture (or a checker) upscaled, with UV edges, seams and face ids drawn on top.",
            true,
            uv_layout,
        ),
        def(
            "pick",
            "What is under pixel (x, y) of the most recent render (or contact sheet)? Returns object, face, world position, normal and texture UV.",
            true,
            pick,
        ),
        def(
            "paint_stroke",
            "Paint on the model by pixel coordinates of the most recent render: each point is ray-cast onto the surface and the stroke is drawn into the hit faces' texture at the corresponding UVs. Points are interpolated so strokes are continuous.",
            false,
            paint_stroke,
        ),
        def(
            "show_viewer",
            "Open the live viewer window on the user's screen. It mirrors edits as they happen: an overview grid of all scenes (the user can click one to view it alone), or one scene given by `show`. Only open it when the user wants to watch; it does not change what render returns.",
            true,
            show_viewer,
        ),
        def("hide_viewer", "Close the live viewer window.", true, hide_viewer),
        def(
            "set_auto_preview",
            "Attach a small perspective preview image to the result of every geometry edit (size in px, 0 = off). Handy while iterating, costs image tokens.",
            false,
            set_auto_preview,
        ),
    ]
}

#[derive(Deserialize, JsonSchema, Default)]
#[serde(default)]
struct RenderArgs {
    view: View,
    /// Image width (default 512; contact sheet 1024).
    width: Option<u32>,
    height: Option<u32>,
    /// Object names to frame (default: all visible).
    focus: Option<Vec<String>>,
    /// Zoom relative to the auto-framed view (2 = twice as close).
    zoom: Option<f32>,
    overlays: Option<Overlays>,
    /// Override all textures' filtering.
    filter: Option<Filter>,
    /// Render only the focus objects.
    isolate: bool,
    /// Label text scale (1 = 8px, 2 = 16px).
    label_scale: Option<u32>,
    /// Corner legend with view name and axes (default true).
    legend: Option<bool>,
    /// Pose with animation clips at frames, blended by weight, e.g. [{"clip":"walk","frame":6}].
    /// Without it, the static pose (pose_bone, set_morph_weight) is shown.
    pose: Option<Vec<super::anim::PlayArg>>,
    /// Show skin weights of this bone as a heatmap (blue 0 -> red 1, black = unweighted)
    /// on the focus object (or the active object).
    show_weights: Option<String>,
}

/// The scene as it should be shown for these args: posed and, optionally, as a weight heatmap.
fn display_scene(ctx: &Ctx, a: &RenderArgs) -> Result<(Scene, Vec<polly_core::anim::ClipPlay>)> {
    let s = &ctx.doc.scene;
    let plays = super::anim::plays(s, a.pose.as_deref().unwrap_or(&[]))?;
    let mut shown = polly_core::anim::pose_scene(s, &plays);
    if let Some(bone) = &a.show_weights {
        let obj = match a.focus.as_ref().and_then(|f| f.first()) {
            Some(n) => s.resolve(Some(n))?,
            None => s.resolve(None)?,
        };
        shown = super::anim::weight_view(&shown, obj, bone)?;
    }
    Ok((shown, plays))
}

fn options(scene: &Scene, a: &RenderArgs, default_size: u32) -> Result<RenderOptions> {
    let focus = a
        .focus
        .as_ref()
        .map(|names| names.iter().map(|n| scene.resolve(Some(n))).collect::<polly_core::Result<Vec<ObjectId>>>())
        .transpose()?;
    Ok(RenderOptions {
        view: a.view.clone(),
        width: a.width.unwrap_or(default_size),
        height: a.height.unwrap_or(default_size),
        focus,
        zoom: a.zoom.unwrap_or(1.0),
        overlays: a.overlays.clone().unwrap_or_default(),
        filter: a.filter,
        isolate: a.isolate,
        background: None,
        label_scale: a.label_scale,
        legend: a.legend.unwrap_or(true),
    })
}

pub fn png(img: &image::RgbaImage) -> Result<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

fn report(ctx: &mut Ctx, r: &Rendered, plays: Vec<polly_core::anim::ClipPlay>) -> Result<Out> {
    *ctx.last_view = Some(LastView { camera: r.camera.clone(), tiles: r.tiles.clone(), plays });
    let c = &r.camera;
    let mut j = json!({
        "view": c.view,
        "size": [r.image.width(), r.image.height()],
        "camera": {"eye": c.eye, "target": c.target, "ortho": c.ortho},
        "labels": r.labels,
    });
    if !r.notes.is_empty() {
        j["notes"] = json!(r.notes);
    }
    if ctx.doc.scene.objects.is_empty() {
        j["notes"] = json!(["the scene is empty"]);
    }
    Ok(Out::json(j).with_image(png(&r.image)?))
}

fn render(ctx: &mut Ctx, a: RenderArgs) -> Result<Out> {
    let o = options(&ctx.doc.scene, &a, 512)?;
    let (shown, plays) = display_scene(ctx, &a)?;
    let r = ctx.services.renderer()?.render(&shown, &o)?;
    report(ctx, &r, plays)
}

fn contact_sheet(ctx: &mut Ctx, a: RenderArgs) -> Result<Out> {
    let o = options(&ctx.doc.scene, &a, 1024)?;
    let (shown, plays) = display_scene(ctx, &a)?;
    let r = ctx.services.renderer()?.contact_sheet(&shown, &o)?;
    report(ctx, &r, plays)
}

#[derive(Deserialize, JsonSchema)]
struct UvLayout {
    object: Option<String>,
    /// Texture to show (default: the object's first textured material).
    texture: Option<String>,
    /// Image size (default 512).
    size: Option<u32>,
    #[serde(default = "yes")]
    show_ids: bool,
}

fn yes() -> bool {
    true
}

fn uv_layout(ctx: &mut Ctx, a: UvLayout) -> Result<Out> {
    let s = &ctx.doc.scene;
    let id = resolve(s, &a.object)?;
    let tex = match &a.texture {
        Some(t) => Some(s.resolve_texture(t)?),
        None => s.object(id)?.mesh.faces.values().find_map(|f| s.materials.get(&f.material)?.texture),
    };
    let img = polly_render::render_uv_layout(s, id, tex, a.size.unwrap_or(512).clamp(64, 2048), a.show_ids);
    let mut j = json!({"object": s.objects[&id].name, "texture": tex.and_then(|t| s.textures.get(&t)).map(|t| &t.name)});
    if let Some(t) = tex.and_then(|t| s.textures.get(&t)) {
        j["texture_size"] = json!([t.width, t.height]);
        j["pixel_scale"] = json!(img.width() as f32 / t.width as f32);
    }
    Ok(Out::json(j).with_image(png(&img)?))
}

/// Camera for a pixel of the last render (resolving contact-sheet tiles).
/// The geometry the last render showed (posed the same way).
fn last_scene(ctx: &Ctx) -> Scene {
    let plays = ctx.last_view.as_ref().map(|l| l.plays.clone()).unwrap_or_default();
    polly_core::anim::pose_scene(&ctx.doc.scene, &plays)
}

fn camera_at(ctx: &Ctx, x: f32, y: f32) -> Result<(CameraInfo, f32, f32)> {
    let lv = ctx.last_view.clone().context("nothing rendered in this scene yet; call render first")?;
    if lv.tiles.is_empty() {
        return Ok((lv.camera, x, y));
    }
    for t in &lv.tiles {
        let (lx, ly) = (x - t.x as f32, y - t.y as f32);
        if lx >= 0.0 && ly >= 0.0 && lx < t.camera.width as f32 && ly < t.camera.height as f32 {
            return Ok((t.camera.clone(), lx, ly));
        }
    }
    bail!("pixel ({x}, {y}) is outside the last contact sheet")
}

#[derive(Deserialize, JsonSchema)]
struct Pick {
    x: f32,
    y: f32,
}

fn pick(ctx: &mut Ctx, a: Pick) -> Result<Out> {
    let (cam, x, y) = camera_at(ctx, a.x, a.y)?;
    let s = &last_scene(ctx);
    match polly_render::pick(s, &cam, x, y) {
        None => Ok(Out::json(json!({"hit": null, "view": cam.view}))),
        Some(h) => {
            let mut j = serde_json::to_value(&h)?;
            j["face"] = json!(polly_core::ids::face(h.face));
            j["object"] = json!(h.object_name);
            j["view"] = json!(cam.view);
            Ok(Out::json(j))
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct Stroke {
    /// Pixel coordinates in the last render, e.g. [[100,120],[140,128]]. One point = a dab.
    points: Vec<[f32; 2]>,
    color: Rgba,
    /// Brush radius in texture pixels (default 1).
    radius_px: Option<f32>,
    #[serde(default)]
    blend: Blend,
}

fn paint_stroke(ctx: &mut Ctx, a: Stroke) -> Result<Out> {
    if a.points.is_empty() {
        bail!("no points");
    }
    // Sample along the screen polyline every ~2px.
    let mut samples = vec![a.points[0]];
    for w in a.points.windows(2) {
        let (p, q) = (w[0], w[1]);
        let n = (((q[0] - p[0]).hypot(q[1] - p[1])) / 2.0).ceil().max(1.0) as usize;
        for k in 1..=n {
            let t = k as f32 / n as f32;
            samples.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
        }
    }
    // Group consecutive hits into UV polylines per texture; break on face changes
    // across texture islands (different face => new segment).
    let mut segments: Vec<(u32, Vec<[f32; 2]>)> = vec![];
    let mut last: Option<(ObjectId, u32)> = None;
    let mut missed = 0;
    {
        let s = &last_scene(ctx);
        for p in &samples {
            let (cam, x, y) = camera_at(ctx, p[0], p[1])?;
            let Some(h) = polly_render::pick(s, &cam, x, y) else {
                missed += 1;
                last = None;
                continue;
            };
            let face = &s.objects[&h.object].mesh.faces[&h.face];
            let Some(tid) = s.materials.get(&face.material).and_then(|m| m.texture) else {
                missed += 1;
                last = None;
                continue;
            };
            let key = (h.object, h.face);
            match segments.last_mut() {
                Some((t, pts)) if *t == tid && last == Some(key) => pts.push(h.uv),
                _ => segments.push((tid, vec![h.uv])),
            }
            last = Some(key);
        }
    }
    if segments.is_empty() {
        bail!("the stroke did not hit any textured surface");
    }
    let radius = a.radius_px.unwrap_or(1.0);
    let mut changed: BTreeMap<String, usize> = BTreeMap::new();
    edit_scene(ctx, "paint stroke", |s| {
        for (tid, pts) in &segments {
            let palette = s.textures[tid].palette.clone().and_then(|n| crate::colors::palette_named(s, &n));
            let paint = polly_paint::Paint::new(a.color).with_blend(a.blend).with_palette(palette);
            let t = s.texture_mut(*tid)?;
            let n = polly_paint::stroke_uv(t, pts, radius, &paint);
            *changed.entry(t.name.clone()).or_default() += n;
        }
        Ok(())
    })?;
    let mut j = json!({"pixels_changed": changed, "samples": samples.len()});
    if missed > 0 {
        j["missed_samples"] = json!(missed);
    }
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct AutoPreview {
    /// Preview size in px (0 = off; 256 is a good value).
    size: u32,
}

fn set_auto_preview(ctx: &mut Ctx, a: AutoPreview) -> Result<Out> {
    ctx.services.settings.lock().unwrap().auto_preview = (a.size > 0).then(|| a.size.clamp(64, 1024));
    Ok(ok(if a.size > 0 { format!("auto preview on ({} px)", a.size) } else { "auto preview off".into() }))
}

/// Append a preview render when auto-preview is on.
pub fn maybe_preview(ctx: &mut Ctx, object: ObjectId, out: Out) -> Result<Out> {
    let Some(size) = ctx.services.settings.lock().unwrap().auto_preview else { return Ok(out) };
    let o = RenderOptions { focus: Some(vec![object]), ..RenderOptions::default().with_size(size, size) };
    let shown = polly_core::anim::pose_scene(&ctx.doc.scene, &[]);
    match ctx.services.renderer().and_then(|mut r| r.render(&shown, &o)) {
        Ok(r) => {
            *ctx.last_view = Some(LastView { camera: r.camera.clone(), tiles: vec![], plays: vec![] });
            Ok(out.with_image(png(&r.image)?))
        }
        Err(e) => {
            let mut out = out;
            out.json["preview_error"] = json!(e.to_string());
            Ok(out)
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct ShowViewer {
    /// Open on this scene's view. Default: the overview of all scenes (or "main" if it is the only scene).
    /// Note: this is not the generic `scene` argument; it picks what the window shows.
    show: Option<String>,
}

fn show_viewer(ctx: &mut Ctx, a: ShowViewer) -> Result<Out> {
    if let Some(s) = &a.show {
        ctx.workspace.get(s)?;
    }
    ctx.services.viewer(Some(a.show.clone()))?;
    Ok(ok(match a.show {
        Some(s) => format!("viewer window open on scene '{s}'"),
        None => "viewer window open".into(),
    }))
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn hide_viewer(ctx: &mut Ctx, _: NoArgs) -> Result<Out> {
    ctx.services.viewer(None)?;
    Ok(ok("viewer window closed"))
}
