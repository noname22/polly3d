use std::collections::BTreeMap;
use std::io::Cursor;

use anyhow::{Context, Result, bail};
use polly_core::glam::Vec3;
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
            "Paint on the model by pixel coordinates of the most recent render: each point is ray-cast onto the surface and painted into the hit faces' texture. mode \"line\" (default) joins the points into a continuous stroke; \"dabs\" paints each point as a separate dot. `mirror` also paints the mirror image across a world plane (both eyes, both brows).",
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
            "stamp",
            "Stamp a disc or ring onto the model, sized in world units and continuous across UV seams: at a pixel of the last render, or at a point in the object's space. Optional outline. Good for buttons, studs, eyes, coins, rivets.",
            false,
            stamp,
        ),
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
    /// Show these variants instead of the active ones, e.g. {"hair": "braids", "head": "helmet"}.
    variants: Option<BTreeMap<String, String>>,
}

/// The scene as it should be shown for these args: variants chosen, posed and,
/// optionally, as a weight heatmap.
fn display_scene(ctx: &Ctx, a: &RenderArgs) -> Result<(Scene, Vec<polly_core::anim::ClipPlay>, BTreeMap<String, String>)> {
    let s = &ctx.doc.scene;
    let plays = super::anim::plays(s, a.pose.as_deref().unwrap_or(&[]))?;
    let choice = super::variants::choice(s, &a.variants)?;
    let mut shown = polly_core::variants::display(s, &plays, &choice);
    if let Some(bone) = &a.show_weights {
        let obj = match a.focus.as_ref().and_then(|f| f.first()) {
            Some(n) => s.resolve(Some(n))?,
            None => s.resolve(None)?,
        };
        shown = super::anim::weight_view(&shown, obj, bone)?;
    }
    Ok((shown, plays, choice))
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
        frame_bounds: None,
    })
}

pub fn png(img: &image::RgbaImage) -> Result<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

fn report(ctx: &mut Ctx, r: &Rendered, plays: Vec<polly_core::anim::ClipPlay>, variants: BTreeMap<String, String>) -> Result<Out> {
    *ctx.last_view = Some(LastView { camera: r.camera.clone(), tiles: r.tiles.clone(), plays, variants });
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
    let (shown, plays, variants) = display_scene(ctx, &a)?;
    let r = ctx.services.renderer()?.render(&shown, &o)?;
    report(ctx, &r, plays, variants)
}

fn contact_sheet(ctx: &mut Ctx, a: RenderArgs) -> Result<Out> {
    let o = options(&ctx.doc.scene, &a, 1024)?;
    let (shown, plays, variants) = display_scene(ctx, &a)?;
    let r = ctx.services.renderer()?.contact_sheet(&shown, &o)?;
    report(ctx, &r, plays, variants)
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
    match ctx.last_view.as_ref() {
        Some(l) => polly_core::variants::display(&ctx.doc.scene, &l.plays, &l.variants),
        None => polly_core::variants::display(&ctx.doc.scene, &[], &BTreeMap::new()),
    }
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

#[derive(Deserialize, JsonSchema, Default, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
enum StrokeMode {
    /// Join the points into one continuous line (default).
    #[default]
    Line,
    /// Paint each point as a separate dot (eyes, buttons, studs).
    Dabs,
}

#[derive(Deserialize, JsonSchema)]
struct MirrorPaint {
    /// World axis normal to the mirror plane (default "x": left/right symmetry).
    #[serde(default = "x_axis")]
    axis: String,
    /// A point on the mirror plane (default: the world origin).
    about: Option<[f32; 3]>,
}

fn x_axis() -> String {
    "x".into()
}

#[derive(Deserialize, JsonSchema)]
struct Stroke {
    /// Pixel coordinates in the last render, e.g. [[100,120],[140,128]].
    points: Vec<[f32; 2]>,
    color: Rgba,
    /// Brush radius in texture pixels (default 1).
    radius_px: Option<f32>,
    #[serde(default)]
    blend: Blend,
    /// "line" (default) joins the points; "dabs" paints each point separately.
    #[serde(default)]
    mode: StrokeMode,
    /// Also paint the mirror image across a world plane (e.g. both eyes from one stroke).
    mirror: Option<MirrorPaint>,
}

/// One surface hit of a stroke sample.
struct StrokeHit {
    sample: usize,
    object: ObjectId,
    face: u32,
    texture: u32,
    uv: [f32; 2],
}

/// Group hits into UV polylines per texture: a new segment starts on a face
/// change, a missed sample, or (dabs) every sample.
fn stroke_segments(hits: &[StrokeHit], dabs: bool) -> Vec<(u32, Vec<[f32; 2]>)> {
    let mut segments: Vec<(u32, Vec<[f32; 2]>)> = vec![];
    let mut prev: Option<&StrokeHit> = None;
    for h in hits {
        let joins = !dabs && prev.is_some_and(|p| p.sample + 1 == h.sample && p.object == h.object && p.face == h.face && p.texture == h.texture);
        match segments.last_mut() {
            Some((_, pts)) if joins => pts.push(h.uv),
            _ => segments.push((h.texture, vec![h.uv])),
        }
        prev = Some(h);
    }
    segments
}

fn paint_stroke(ctx: &mut Ctx, a: Stroke) -> Result<Out> {
    if a.points.is_empty() {
        bail!("no points");
    }
    let dabs = a.mode == StrokeMode::Dabs;
    // Lines are sampled along the screen polyline every ~2px; dabs are used as given.
    let mut samples = vec![a.points[0]];
    for w in a.points.windows(2) {
        let (p, q) = (w[0], w[1]);
        if dabs {
            samples.push(q);
            continue;
        }
        let n = (((q[0] - p[0]).hypot(q[1] - p[1])) / 2.0).ceil().max(1.0) as usize;
        for k in 1..=n {
            let t = k as f32 / n as f32;
            samples.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
        }
    }
    let mirror_axis = match a.mirror.as_ref().map(|m| m.axis.to_lowercase()) {
        None => None,
        Some(ax) => Some(match ax.as_str() {
            "x" => Vec3::X,
            "y" => Vec3::Y,
            "z" => Vec3::Z,
            other => bail!("mirror axis must be x, y or z, not '{other}'"),
        }),
    };
    let s = last_scene(ctx);
    let texture_of = |obj: ObjectId, face: u32| -> Option<u32> {
        let f = s.objects.get(&obj)?.mesh.faces.get(&face)?;
        s.materials.get(&f.material)?.texture
    };
    let mut hits = vec![];
    let mut world_hits = vec![];
    let mut missed = 0;
    for (i, p) in samples.iter().enumerate() {
        let (cam, x, y) = camera_at(ctx, p[0], p[1])?;
        let Some(h) = polly_render::pick(&s, &cam, x, y) else {
            missed += 1;
            continue;
        };
        let Some(tid) = texture_of(h.object, h.face) else {
            missed += 1;
            continue;
        };
        world_hits.push((i, Vec3::from(h.position)));
        hits.push(StrokeHit { sample: i, object: h.object, face: h.face, texture: tid, uv: h.uv });
    }
    if hits.is_empty() {
        bail!("the stroke did not hit any textured surface");
    }
    // Mirror: reflect each hit and paint the nearest surface point there.
    let mut mirrored = vec![];
    let mut mirror_missed = 0;
    if let (Some(axis), Some(m)) = (mirror_axis, &a.mirror) {
        let about = Vec3::from_array(m.about.unwrap_or([0.0; 3]));
        let bvhs: Vec<(ObjectId, polly_core::bvh::Bvh)> = s
            .objects
            .iter()
            .filter(|(_, o)| o.visible)
            .map(|(id, o)| (*id, polly_core::bvh::Bvh::new(&o.mesh, s.world_matrix(*id))))
            .collect();
        let size = s.bounds().map_or(1.0, |(lo, hi)| (hi - lo).length());
        for (i, p) in &world_hits {
            let q = *p - 2.0 * (*p - about).dot(axis) * axis;
            let best = bvhs
                .iter()
                .filter_map(|(id, b)| b.closest_point(q).map(|c| (*id, b, c)))
                .min_by(|x, y| x.2.3.total_cmp(&y.2.3));
            match best {
                Some((obj, b, (tri, _, bary, d))) if d < size * 0.02 => {
                    let t = &b.tris[tri];
                    let uv = t.uv(&s.objects[&obj].mesh, bary).to_array();
                    match texture_of(obj, t.face) {
                        Some(tid) => mirrored.push(StrokeHit { sample: *i, object: obj, face: t.face, texture: tid, uv }),
                        None => mirror_missed += 1,
                    }
                }
                _ => mirror_missed += 1,
            }
        }
    }
    let mut segments = stroke_segments(&hits, dabs);
    segments.extend(stroke_segments(&mirrored, dabs));
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
    let mut j = json!({"pixels_changed": changed, "samples": samples.len(), "hits": hits.len()});
    if missed > 0 {
        j["missed_samples"] = json!(missed);
    }
    if a.mirror.is_some() {
        j["mirrored_hits"] = json!(mirrored.len());
        if mirror_missed > 0 {
            j["mirror_missed"] = json!(format!("{mirror_missed} mirrored points had no textured surface nearby"));
        }
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
    let shown = polly_core::variants::display(&ctx.doc.scene, &[], &BTreeMap::new());
    match ctx.services.renderer().and_then(|mut r| r.render(&shown, &o)) {
        Ok(r) => {
            *ctx.last_view = Some(LastView { camera: r.camera.clone(), tiles: vec![], plays: vec![], variants: BTreeMap::new() });
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

#[derive(Deserialize, JsonSchema)]
struct Stamp {
    /// Where: a pixel [x, y] of the last render...
    pixel: Option<[f32; 2]>,
    /// ...or a point in the object's local space (snapped to the nearest surface).
    point: Option<[f32; 3]>,
    /// Object for `point` (default: active).
    object: Option<String>,
    /// Radius in world units.
    radius: f32,
    /// Leave a hole of this radius (makes a ring). Default 0.
    #[serde(default)]
    inner_radius: f32,
    color: Rgba,
    /// Outline colour (drawn inside the edge).
    outline: Option<Rgba>,
    /// Outline width in world units (default 20% of the radius).
    outline_width: Option<f32>,
    /// Texture (default: the texture of the face under the stamp).
    texture: Option<String>,
    #[serde(default)]
    blend: Blend,
}

fn stamp(ctx: &mut Ctx, a: Stamp) -> Result<Out> {
    if a.radius <= 0.0 {
        bail!("radius must be > 0");
    }
    // Resolve the stamp centre (object space), its surface normal and the shown mesh.
    let (obj, center, normal, face, shown) = match (a.pixel, a.point) {
        (Some([px, py]), None) => {
            let (cam, x, y) = camera_at(ctx, px, py)?;
            let s = last_scene(ctx);
            let h = polly_render::pick(&s, &cam, x, y).context("nothing under that pixel")?;
            let inv = s.world_matrix(h.object).inverse();
            let c = inv.transform_point3(Vec3::from(h.position));
            let n = inv.transform_vector3(Vec3::from(h.normal)).normalize_or_zero();
            (h.object, c, n, h.face, s)
        }
        (None, Some(p)) => {
            let s = polly_core::variants::display(&ctx.doc.scene, &[], &BTreeMap::new());
            let obj = resolve(&ctx.doc.scene, &a.object)?;
            let bvh = polly_core::bvh::Bvh::new(&s.objects[&obj].mesh, polly_core::glam::Mat4::IDENTITY);
            let (tri, q, _, _) = bvh.closest_point(Vec3::from_array(p)).context("object has no faces")?;
            let t = bvh.tris[tri];
            (obj, q, t.normal(), t.face, s)
        }
        _ => bail!("give either `pixel` or `point`"),
    };
    let doc = &ctx.doc.scene;
    let tid = match &a.texture {
        Some(t) => doc.resolve_texture(t)?,
        None => {
            let f = &doc.objects[&obj].mesh.faces[&face];
            doc.materials.get(&f.material).and_then(|m| m.texture).context("the face there has no texture; pass `texture`")?
        }
    };
    let faces = super::paint::faces_with_texture(doc, obj, tid);
    let (w, h) = (doc.textures[&tid].width, doc.textures[&tid].height);
    let map = polly_paint::SurfaceMap::new(&shown.objects[&obj].mesh, &faces, w, h, 1);
    let outline = a.outline.map(|c| (c, a.outline_width.unwrap_or(a.radius * 0.2)));
    let mut changed = 0;
    let name = edit_scene(ctx, "stamp", |s| {
        let palette = s.textures[&tid].palette.clone().and_then(|n| crate::colors::palette_named(s, &n));
        let paint = polly_paint::Paint::new(a.color).with_blend(a.blend).with_palette(palette);
        let t = s.texture_mut(tid)?;
        changed = polly_paint::stamp(t, &map, center, normal, a.radius, a.inner_radius, a.color, outline, &paint);
        Ok(t.name.clone())
    })?;
    let mut j = json!({"texture": name, "pixels_changed": changed, "center": round3(center)});
    if changed == 0 {
        j["note"] = json!("no texels within the radius; the stamp may be smaller than one texel (increase radius or texture size)");
    }
    Ok(Out::json(j))
}
