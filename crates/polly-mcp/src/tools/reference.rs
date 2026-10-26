//! Reference images shown behind or over the model in the front/right/top views.

use std::sync::Arc;

use anyhow::{Context, Result};
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
            "Show an image file (PNG, JPEG...) as a reference in the front, right or top view, e.g. a character sheet to model over. It is placed in world units on that view's plane: `height` world units tall (width from the aspect ratio), centred at `center` = [u, v] in the view's screen axes (front: u=+X v=+Y; right: u=-Z v=+Y; top: u=+X v=-Z). Defaults: height 2, standing on the ground (centre [0, height/2]; top view: [0, 0]), behind the model at opacity 0.5. `crop` = [x, y, w, h] source pixels selects the part of the image to use (one view of a multi-view sheet). It is shown only when rendering that ortho view and zooms with the model. Re-adding a name replaces it.",
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
    })
}

/// Standing on the ground in the side views, centred in the top view.
fn default_center(view: RefView, size: f32) -> [f32; 2] {
    match view {
        RefView::Top => [0.0, 0.0],
        _ => [0.0, size / 2.0],
    }
}

fn r4(x: f32) -> f32 {
    (x * 10000.0).round() / 10000.0 + 0.0
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
        "world_per_pixel": (r.scale() * 1e5).round() / 1e5,
        "corners": {
            "top_left": round3(q[0].0),
            "bottom_right": round3(q[2].0),
        },
        "layer": r.layer,
        "opacity": r4(r.opacity),
        "visible": r.visible,
    })
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
