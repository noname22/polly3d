//! Reference images shown behind or over the model in the axis views (front, back, left, right, top, bottom).

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use polly_core::reference::{Layer, RefView, Reference};
use polly_core::scene::Scene;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

/// Longest side of a stored reference image; larger images are downscaled on import.
pub const MAX_SIZE: u32 = 2048;

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "add_reference",
            "Show an image file (PNG, JPEG...) as a reference in an axis view (front, back, left, right, top, bottom), e.g. a character sheet to model over. It is placed in world units on that view's plane: `height` world units tall (width from the aspect ratio), centred at `center` = [u, v] in the view's screen axes (front: u=+X v=+Y; back: u=-X v=+Y; left: u=+Z v=+Y; right: u=-Z v=+Y; top: u=+X v=-Z; bottom: u=+X v=+Z), so an image reads as drawn. Defaults: height 2, standing on the ground (centre [0, height/2]; top/bottom: [0, 0]), behind the model at opacity 0.5. `crop` = [x, y, w, h] source pixels selects the part of the image to use (one view of a multi-view sheet). It is shown only when rendering that ortho view and zooms with the model. Re-adding a name replaces it.",
            false,
            add_reference,
        ),
        def(
            "update_reference",
            "Change a reference image's crop, placement (center, height), view, layer (behind/front), opacity or visibility.",
            false,
            update_reference,
        ),
        def("remove_reference", "Remove a reference image.", false, remove_reference),
        def(
            "measure_reference",
            "Measure the drawing in a reference image in world units: its top and bottom, and its outer edges at given heights (default: 12 evenly spaced). The drawing is what differs from the image's background (or its alpha); everything counts, so crop away captions and neighbouring views. Use it to block out to the drawing's proportions.",
            true,
            measure_reference,
        ),
        def(
            "compare_reference",
            "Compare the model's outline in a reference's view with the drawing: per height, how far the model sticks out (+) or falls short (-) on each side, in world units, plus a picture (red = model outside the drawing, blue = drawing not covered by the model). Outer outlines only (holes between arm and body don't count).",
            true,
            compare_reference,
        ),
        def("list_references", "List reference images with their view, crop, placement in world units, layer and opacity.", true, list_references),
    ]
}

/// Decode an image file into a reference (downscaled to `MAX_SIZE`), with
/// default placement. Shared with the viewer's drag-and-drop.
pub fn load_reference(path: &str, view: RefView, name: Option<&str>) -> Result<Reference> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let mut img = image::load_from_memory(&bytes).with_context(|| format!("decoding {path}"))?.to_rgba8();
    if img.width().max(img.height()) > MAX_SIZE {
        let k = MAX_SIZE as f32 / img.width().max(img.height()) as f32;
        let (w, h) = (((img.width() as f32 * k).round() as u32).max(1), ((img.height() as f32 * k).round() as u32).max(1));
        img = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
    }
    let name = name.map(str::to_string).unwrap_or_else(|| {
        std::path::Path::new(path).file_stem().map_or("reference".into(), |s| s.to_string_lossy().into_owned())
    });
    let size = 2.0;
    Ok(Reference {
        name,
        view,
        width: img.width(),
        height: img.height(),
        pixels: Arc::new(img.into_raw()),
        crop: None,
        center: default_center(view, size),
        size,
        layer: Layer::Behind,
        opacity: 0.5,
        visible: true,
        flip: false,
    })
}

/// Standing on the ground in the side views, centred in the top view.
fn default_center(view: RefView, size: f32) -> [f32; 2] {
    match view {
        RefView::Top | RefView::Bottom => [0.0, 0.0],
        _ => [0.0, size / 2.0],
    }
}

/// Rounded in f64 so the JSON shows 0.9, not 0.8999999761581421.
fn r4(x: f32) -> f64 {
    (x as f64 * 10000.0).round() / 10000.0 + 0.0
}

fn ref_json(r: &Reference) -> Value {
    let q = r.quad();
    let e = r.extent();
    json!({
        "name": r.name,
        "view": r.view,
        "image_size": [r.width, r.height],
        "crop": r.rect(),
        "center": r.center.map(r4),
        "size": [r4(e.x), r4(e.y)],
        "world_per_pixel": (r.scale() as f64 * 1e5).round() / 1e5,
        "corners": {
            "top_left": q[0].0.to_array().map(r4),
            "bottom_right": q[2].0.to_array().map(r4),
        },
        "layer": r.layer,
        "opacity": r4(r.opacity),
        "visible": r.visible,
        "flip": r.flip,
    })
}

/// World axis a view direction runs along: ("x", 1.0) for +X, ("z", -1.0) for -Z.
fn axis(d: polly_core::glam::Vec3) -> (&'static str, f32) {
    if d.x.abs() > 0.5 {
        ("x", d.x.signum())
    } else if d.y.abs() > 0.5 {
        ("y", d.y.signum())
    } else {
        ("z", d.z.signum())
    }
}

/// A view-u range as a sorted world-axis range.
fn world_range(view: RefView, lo: f32, hi: f32) -> [f64; 2] {
    let (_, s) = axis(view.axes().0);
    let (a, b) = (lo * s, hi * s);
    [r4(a.min(b)), r4(a.max(b))]
}

#[derive(Deserialize, JsonSchema)]
struct AddReference {
    /// Image file (PNG, JPEG, ...).
    path: String,
    view: RefView,
    /// Default: the file name without extension.
    name: Option<String>,
    /// [x, y, w, h] in source pixels (top-left origin).
    crop: Option<[u32; 4]>,
    /// Centre in the view's screen axes, world units.
    center: Option<[f32; 2]>,
    /// Height of the shown (cropped) image in world units (default 2).
    height: Option<f32>,
    /// Default "behind".
    layer: Option<Layer>,
    /// 0..1, default 0.5.
    opacity: Option<f32>,
    /// Mirror the image left/right.
    #[serde(default)]
    flip: bool,
    /// Place it by two known points instead of center/height: e.g. the ground row
    /// and the top of the head (file pixels -> view coordinates).
    calibrate: Option<[CalPoint; 2]>,
}

#[derive(Deserialize, JsonSchema, Clone, Copy)]
struct CalPoint {
    /// Image pixel [x, y] (top-left origin).
    pixel: [f32; 2],
    /// Where it belongs in the view's screen axes [u, v], world units (front: [x, y]; right: [-z, y]; ...).
    at: [f32; 2],
}

fn calibrate(r: &mut Reference, c: &[CalPoint; 2], pixel_scale: [f32; 2]) -> Result<()> {
    let p = |c: &CalPoint| (polly_core::glam::Vec2::new(c.pixel[0] * pixel_scale[0], c.pixel[1] * pixel_scale[1]), polly_core::glam::Vec2::from(c.at));
    Ok(r.calibrate(p(&c[0]), p(&c[1]))?)
}

fn add_reference(ctx: &mut Ctx, a: AddReference) -> Result<Out> {
    let mut r = load_reference(&a.path, a.view, a.name.as_deref())?;
    let (w0, h0) = (r.width, r.height);
    if let Some(c) = a.crop {
        r.crop = Some(scale_crop(c, &a.path, w0, h0)?);
    }
    if let Some(h) = a.height {
        r.size = h;
    }
    r.center = a.center.unwrap_or(default_center(r.view, r.size));
    r.layer = a.layer.unwrap_or_default();
    if let Some(o) = a.opacity {
        r.opacity = o;
    }
    r.flip = a.flip;
    if let Some(c) = &a.calibrate {
        if a.center.is_some() || a.height.is_some() {
            anyhow::bail!("give either calibrate or center/height");
        }
        let (fw, fh) = image::image_dimensions(&a.path)?;
        calibrate(&mut r, c, [w0 as f32 / fw as f32, h0 as f32 / fh as f32])?;
    }
    let replaced = ctx.doc.scene.references.iter().any(|x| x.name == r.name);
    let name = r.name.clone();
    edit_scene(ctx, "add reference", |s| Ok(s.set_reference(r)?))?;
    let mut j = ref_json(ctx.doc.scene.reference(&name)?);
    if replaced {
        j["note"] = json!(format!("replaced reference '{name}'"));
    }
    Ok(Out::json(j))
}

/// A crop given in the file's pixels, scaled to the stored (possibly downscaled) image.
fn scale_crop(c: [u32; 4], path: &str, stored_w: u32, stored_h: u32) -> Result<[u32; 4]> {
    let (fw, fh) = image::image_dimensions(path).with_context(|| format!("reading {path}"))?;
    if (fw, fh) == (stored_w, stored_h) {
        return Ok(c);
    }
    let (kx, ky) = (stored_w as f32 / fw as f32, stored_h as f32 / fh as f32);
    let x = (c[0] as f32 * kx).round() as u32;
    let y = (c[1] as f32 * ky).round() as u32;
    let w = ((c[2] as f32 * kx).round() as u32).max(1).min(stored_w.saturating_sub(x));
    let h = ((c[3] as f32 * ky).round() as u32).max(1).min(stored_h.saturating_sub(y));
    Ok([x, y, w, h])
}

#[derive(Deserialize, JsonSchema)]
struct UpdateReference {
    name: String,
    view: Option<RefView>,
    /// [x, y, w, h] in stored image pixels (see list_references image_size); [] for the whole image.
    crop: Option<Vec<u32>>,
    center: Option<[f32; 2]>,
    /// Height of the shown image in world units.
    height: Option<f32>,
    layer: Option<Layer>,
    opacity: Option<f32>,
    visible: Option<bool>,
    flip: Option<bool>,
    /// Re-place by two known points (stored image pixels -> view coordinates).
    calibrate: Option<[CalPoint; 2]>,
}

fn update_reference(ctx: &mut Ctx, a: UpdateReference) -> Result<Out> {
    ctx.doc.scene.reference(&a.name)?;
    let crop = match a.crop.as_deref() {
        None => None,
        Some([]) => Some(None),
        Some(&[x, y, w, h]) => Some(Some([x, y, w, h])),
        Some(_) => anyhow::bail!("crop must be [x, y, w, h] or [] for the whole image"),
    };
    edit_scene(ctx, "update reference", |s: &mut Scene| {
        let mut r = s.reference(&a.name)?.clone();
        if let Some(v) = a.view {
            r.view = v;
        }
        if let Some(c) = crop {
            r.crop = c;
        }
        if let Some(c) = a.center {
            r.center = c;
        }
        if let Some(h) = a.height {
            r.size = h;
        }
        if let Some(l) = a.layer {
            r.layer = l;
        }
        if let Some(o) = a.opacity {
            r.opacity = o;
        }
        if let Some(v) = a.visible {
            r.visible = v;
        }
        if let Some(f) = a.flip {
            r.flip = f;
        }
        if let Some(c) = &a.calibrate {
            let p = |c: &CalPoint| (polly_core::glam::Vec2::from(c.pixel), polly_core::glam::Vec2::from(c.at));
            r.calibrate(p(&c[0]), p(&c[1]))?;
        }
        Ok(s.set_reference(r)?)
    })?;
    Ok(Out::json(ref_json(ctx.doc.scene.reference(&a.name)?)))
}

#[derive(Deserialize, JsonSchema)]
struct NameArg {
    name: String,
}

fn remove_reference(ctx: &mut Ctx, a: NameArg) -> Result<Out> {
    ctx.doc.scene.reference(&a.name)?;
    edit_scene(ctx, "remove reference", |s| {
        s.references.retain(|r| r.name != a.name);
        Ok(())
    })?;
    Ok(ok(format!("removed reference '{}'", a.name)))
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn list_references(ctx: &mut Ctx, _: Empty) -> Result<Out> {
    Ok(Out::json(json!({"references": ctx.doc.scene.references.iter().map(ref_json).collect::<Vec<_>>()})))
}

#[derive(Deserialize, JsonSchema)]
struct MeasureRef {
    name: String,
    /// Heights to measure at, as view v coordinates (front/back/left/right: y; top: -z; bottom: z).
    heights: Option<Vec<f32>>,
    /// Colour distance from the background that counts as drawing (0-441, default 60).
    threshold: Option<f32>,
}

/// The drawing's vertical extent: (top v, bottom v) of rows with ink.
fn ink_rows(r: &Reference, bg: [u8; 4], th: f32) -> Option<(f32, f32)> {
    let [_, y0, _, h] = r.rect();
    let rows: Vec<u32> = (y0..y0 + h).filter(|y| r.ink_span(*y, bg, th).is_some()).collect();
    let (top, bottom) = (*rows.first()?, *rows.last()?);
    Some((r.pixel_to_view(0.0, top as f32).y, r.pixel_to_view(0.0, bottom as f32 + 1.0).y))
}

fn measure_reference(ctx: &mut Ctx, a: MeasureRef) -> Result<Out> {
    let r = ctx.doc.scene.reference(&a.name)?;
    let th = a.threshold.unwrap_or(60.0);
    let bg = r.background();
    let (top, bottom) = ink_rows(r, bg, th).context("no drawing found (everything looks like the background; try a lower threshold)")?;
    let heights = a.heights.clone().unwrap_or_else(|| (0..12).map(|i| bottom + (top - bottom) * (i as f32 + 0.5) / 12.0).collect());
    let (u_axis, _) = axis(r.view.axes().0);
    let (v_axis, vs) = axis(r.view.axes().1);
    let rows: Vec<Value> = heights
        .iter()
        .map(|v| match r.edges_at(*v, bg, th) {
            Some((lo, hi)) => json!({v_axis: r4(*v * vs), u_axis: world_range(r.view, lo, hi), "width": r4(hi - lo)}),
            None => json!({v_axis: r4(*v * vs), u_axis: Value::Null}),
        })
        .collect();
    Ok(Out::json(json!({
        "reference": r.name,
        "drawing": {format!("{v_axis}_range"): [r4((bottom * vs).min(top * vs)), r4((bottom * vs).max(top * vs))]},
        "rows": rows,
    })))
}

#[derive(Deserialize, JsonSchema)]
struct CompareRef {
    name: String,
    /// Objects to compare (default: everything shown).
    objects: Option<Vec<String>>,
    /// Heights to report (view v coordinates; default 16 evenly over the drawing).
    heights: Option<Vec<f32>>,
    /// Colour distance from the background that counts as drawing (default 60).
    threshold: Option<f32>,
    /// Longest side of the diff picture in pixels (default 768, at most the crop's size x 2).
    size: Option<u32>,
}

fn compare_reference(ctx: &mut Ctx, a: CompareRef) -> Result<Out> {
    use polly_core::glam::{Vec2, Vec3};
    let s = &ctx.doc.scene;
    let r = s.reference(&a.name)?.clone();
    let th = a.threshold.unwrap_or(60.0);
    let bg = r.background();
    let (top, bottom) = ink_rows(&r, bg, th).context("no drawing found in the reference")?;
    // The model as shown (active variants, static pose), as world triangles.
    let shown = polly_core::variants::display(s, &[], &Default::default());
    let ids: Vec<u32> = match &a.objects {
        Some(ns) => ns.iter().map(|n| s.resolve(Some(n))).collect::<polly_core::Result<_>>()?,
        None => shown.objects.iter().filter(|(_, o)| o.visible).map(|(id, _)| *id).collect(),
    };
    let mut tris: Vec<[Vec3; 3]> = Vec::new();
    for id in &ids {
        let o = &shown.objects[id];
        let m = shown.world_matrix(*id);
        for f in o.mesh.faces.values() {
            let pts: Vec<Vec3> = o.mesh.face_positions(f).iter().map(|p| m.transform_point3(*p)).collect();
            for [i, j, k] in polly_core::geom::triangulate_polygon(&pts) {
                tris.push([pts[i], pts[j], pts[k]]);
            }
        }
    }
    if tris.is_empty() {
        bail!("nothing to compare: no faces shown");
    }
    let (u, v) = r.view.axes();
    let model_at = |h: f32| polly_core::geom::slice_extent(&tris, u, v, h);
    let (u_axis, us) = axis(u);
    let (v_axis, vs) = axis(v);
    let heights = a.heights.clone().unwrap_or_else(|| (0..16).map(|i| bottom + (top - bottom) * (i as f32 + 0.5) / 16.0).collect());
    let side = |sign: f32| format!("{}{u_axis}", if sign > 0.0 { "+" } else { "-" });
    let (lo_name, hi_name) = (side(-us), side(us));
    let mut worst: f32 = 0.0;
    let rows: Vec<Value> = heights
        .iter()
        .map(|h| {
            let mut j = json!({v_axis: r4(*h * vs)});
            match (r.edges_at(*h, bg, th), model_at(*h)) {
                (Some((dl, dh)), Some((ml, mh))) => {
                    // Positive = the model sticks out past the drawing on that side.
                    let (out_lo, out_hi) = (dl - ml, mh - dh);
                    worst = worst.max(out_lo.abs()).max(out_hi.abs());
                    j["sticks_out"] = json!({lo_name.as_str(): r4(out_lo), hi_name.as_str(): r4(out_hi)});
                    j["drawing"] = json!(world_range(r.view, dl, dh));
                    j["model"] = json!(world_range(r.view, ml, mh));
                }
                (Some((dl, dh)), None) => {
                    j["note"] = json!("model missing here");
                    j["drawing"] = json!(world_range(r.view, dl, dh));
                }
                (None, Some((ml, mh))) => {
                    j["note"] = json!("model where the drawing has nothing");
                    j["model"] = json!(world_range(r.view, ml, mh));
                }
                (None, None) => {}
            }
            j
        })
        .collect();
    // Picture over the crop: reference faded, red = model only, blue = drawing only.
    let [x0, y0, w, h] = r.rect();
    let k = (a.size.unwrap_or(768).clamp(64, 2048) as f32 / h.max(w) as f32).min(2.0);
    let (iw, ih) = (((w as f32 * k).round() as u32).max(1), ((h as f32 * k).round() as u32).max(1));
    let mut img = image::RgbaImage::new(iw, ih);
    for iy in 0..ih {
        let sy = y0 + ((iy as f32 + 0.5) / k) as u32;
        let vrow = r.pixel_to_view(0.0, sy as f32 + 0.5).y;
        let drawing = r.ink_span(sy.min(y0 + h - 1), bg, th);
        let model = model_at(vrow).map(|(a, b)| (r.view_to_pixel(Vec2::new(a, vrow)).x, r.view_to_pixel(Vec2::new(b, vrow)).x));
        for ix in 0..iw {
            let sx = x0 + ((ix as f32 + 0.5) / k) as u32;
            let i = ((sy.min(y0 + h - 1) * r.width + sx.min(x0 + w - 1)) * 4) as usize;
            let p = &r.pixels[i..i + 4];
            let fade = |c: u8| (255.0 - (255.0 - c as f32) * 0.35) as u8;
            let fx = sx as f32 + 0.5;
            let in_d = drawing.is_some_and(|(a, b)| fx >= a as f32 && fx <= b as f32 + 1.0);
            let in_m = model.is_some_and(|(a, b)| fx >= a.min(b) && fx <= a.max(b));
            let c = match (in_m, in_d) {
                (true, false) => [230, 60, 50, 255],
                (false, true) => [60, 110, 230, 255],
                (true, true) => [fade(p[0]).saturating_sub(40), fade(p[1]).saturating_sub(40), fade(p[2]).saturating_sub(40), 255],
                _ => [fade(p[0]), fade(p[1]), fade(p[2]), 255],
            };
            img.put_pixel(ix, iy, image::Rgba(c));
        }
    }
    Ok(Out::json(json!({
        "reference": r.name,
        "max_error": r4(worst),
        "sides": {"sticks_out": format!("+ = the model extends past the drawing on that side ({lo_name} / {hi_name}); - = it falls short")},
        "rows": rows,
    }))
    .with_image(super::view::png(&img)?))
}
