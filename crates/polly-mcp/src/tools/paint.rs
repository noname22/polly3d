use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use polly_core::mesh::FaceId;
use polly_core::scene::{Filter, Material, Scene, Texture, TextureId};
use polly_paint::{Blend, Dither, Fill, GradientDir, Paint, Region, Rgba};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use super::common::*;
use crate::colors::palette_named;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "create_texture",
            "Create a texture (small sizes like 32/64/128 suit low-poly). Optionally lock it to a palette (every painted pixel snaps to it), and create a material using it, assigned to an object.",
            false,
            create_texture,
        ),
        def("update_texture", "Change a texture's name, filtering (nearest/linear), palette lock, or resize it (nearest).", false, update_texture),
        def("get_texture", "Return a texture as PNG (upscaled for viewing) plus stats: unique colours, palette violations, most used colours.", true, get_texture),
        def("list_palettes", "List built-in and scene palettes with their colours. Pass a name to see one palette's colours.", true, list_palettes),
        def("set_palette", "Define (or replace) a named scene palette from colours, a built-in, a ramp, or colours extracted from a texture.", false, set_palette),
        def("paint_pixels", "Set individual pixels: [[x, y, colour], ...] in texture pixel coordinates (0,0 = top-left).", false, paint_pixels),
        def(
            "draw",
            "Draw crisp pixel shapes on a texture: line, polyline, rect, ellipse, polygon, flood_fill. Optionally clip to faces of an object so strokes stay inside them.",
            false,
            draw,
        ),
        def(
            "fill_faces",
            "Paint faces in surface space: fill each face's UV area with a solid colour, gradient, dither, noise, checker, stripes, bricks, planks or the vertex colours. The texture defaults to the faces' material texture. Requires UVs (unwrap first).",
            false,
            fill_faces,
        ),
        def(
            "paint_region",
            "Apply an effect to a texture region (all, rect, or faces of an object): gradient, dither, noise, outline, border, adjust (hue/sat/value), replace colours.",
            false,
            paint_region,
        ),
        def("paint_uv_stroke", "Brush stroke along a polyline in UV coordinates (0..1) with a round pixel brush.", false, paint_uv_stroke),
        def(
            "bake_shading",
            "Bake depth into a texture: darken by ambient occlusion (creases, contact areas, inside corners) and/or by height (darker towards the bottom). Multiplies the existing colours of the faces' texels. On palette-locked textures each shade snaps to the palette, which gives hard bands; add \"dither\": \"bayer4\" for a graded retro look.",
            false,
            bake_shading,
        ),
        def("quantize", "Snap all texture pixels to a palette, with optional ordered dithering.", false, quantize),
        def("bake_vertex_colors", "Rasterize faces' vertex colours into the texture.", false, bake_vertex_colors),
        def("import_texture", "Load a PNG file as a new texture.", false, import_texture),
        def("export_texture", "Write a texture to a PNG file.", true, export_texture),
    ]
}

/// The palette that palette-index colours ("p3") in tool args refer to: a
/// palette named in the same call (e.g. `create_texture` with `palette`),
/// otherwise the referenced texture's palette lock.
pub fn args_palette(ctx: &Ctx, args: &Value) -> Option<Vec<Rgba>> {
    let s = &ctx.doc.scene;
    if let Some(p) = args.get("palette").and_then(Value::as_str).and_then(|n| palette_named(s, n)) {
        return Some(p);
    }
    let tex = match args.get("texture").and_then(Value::as_str) {
        Some(t) => s.resolve_texture(t).ok(),
        None => {
            let id = s.resolve(args.get("object").and_then(Value::as_str)).ok()?;
            let o = s.objects.get(&id)?;
            let f = o.mesh.faces.values().next()?;
            s.materials.get(&f.material)?.texture
        }
    }?;
    let name = s.textures.get(&tex)?.palette.clone()?;
    palette_named(s, &name)
}

fn tex_paint(scene: &Scene, tid: TextureId, color: Rgba, blend: Blend) -> Paint {
    let palette = scene.textures.get(&tid).and_then(|t| t.palette.clone()).and_then(|n| palette_named(scene, &n));
    Paint::new(color).with_blend(blend).with_palette(palette)
}

/// Edit a texture as an undoable step; returns a short report.
fn edit_texture(ctx: &mut Ctx, tid: TextureId, label: &str, f: impl FnOnce(&mut Texture, &Scene) -> usize) -> Result<Out> {
    let name = edit_scene(ctx, label, |s| {
        let snapshot = s.clone();
        let t = s.texture_mut(tid)?;
        let n = f(t, &snapshot);
        Ok((t.name.clone(), n))
    })?;
    Ok(Out::json(json!({"texture": name.0, "pixels_changed": name.1})))
}

fn texture_of_faces(scene: &Scene, obj: u32, faces: &BTreeSet<FaceId>) -> Result<TextureId> {
    let o = scene.object(obj)?;
    let mats: BTreeSet<u32> = faces.iter().filter_map(|f| o.mesh.faces.get(f)).map(|f| f.material).collect();
    let texs: BTreeSet<TextureId> = mats.iter().filter_map(|m| scene.materials.get(m)?.texture).collect();
    match texs.len() {
        1 => Ok(*texs.iter().next().unwrap()),
        0 => bail!("these faces' material has no texture; pass `texture`, or create_texture with `material` and `object`"),
        _ => bail!("faces use several textures; pass `texture` explicitly"),
    }
}

#[derive(Deserialize, JsonSchema)]
struct CreateTexture {
    name: String,
    #[serde(default = "s64")]
    width: u32,
    #[serde(default = "s64")]
    height: u32,
    /// Initial colour (default opaque white).
    fill: Option<Rgba>,
    /// Lock painting to this palette (built-in name like "pico8", "db32", "endesga32", "sweetie16", "gameboy", "nes", or a scene palette).
    palette: Option<String>,
    /// "nearest" (default, crisp pixels) or "linear".
    filter: Option<Filter>,
    /// Create (or update) a material with this name using the texture.
    material: Option<String>,
    /// Assign that material to all faces of this object.
    object: Option<String>,
}

fn s64() -> u32 {
    64
}

fn create_texture(ctx: &mut Ctx, a: CreateTexture) -> Result<Out> {
    if !(1..=2048).contains(&a.width) || !(1..=2048).contains(&a.height) {
        bail!("texture size must be 1..2048");
    }
    if let Some(p) = &a.palette {
        palette_named(&ctx.doc.scene, p).with_context(|| format!("unknown palette '{p}' (see list_palettes)"))?;
    }
    if ctx.doc.scene.textures.values().any(|t| t.name == a.name) {
        bail!("texture '{}' already exists", a.name);
    }
    let obj = a.object.as_ref().map(|o| ctx.doc.scene.resolve(Some(o))).transpose()?;
    let (tid, mid) = edit_scene(ctx, "create texture", |s| {
        let mut t = Texture::new(&a.name, a.width, a.height, a.fill.unwrap_or([255; 4]));
        t.palette = a.palette.clone();
        t.filter = a.filter.unwrap_or_default();
        let tid = s.add_texture(t);
        let mut mid = None;
        if let Some(mname) = &a.material {
            let id = match s.resolve_material(mname) {
                Ok(id) => id,
                Err(_) => s.add_material(Material::new(mname, [1.0; 4])),
            };
            s.materials.get_mut(&id).unwrap().texture = Some(tid);
            mid = Some(id);
        }
        if let (Some(o), Some(m)) = (obj, mid) {
            let ob = s.object_mut(o)?;
            for f in std::sync::Arc::make_mut(&mut ob.mesh).faces.values_mut() {
                f.material = m;
            }
        }
        Ok((tid, mid))
    })?;
    let mut j = json!({"texture": a.name, "id": format!("t{tid}"), "size": [a.width, a.height]});
    if let Some(m) = mid {
        j["material"] = json!({"name": a.material, "index": m});
    }
    if obj.is_some() {
        j["note"] = json!("material assigned to all faces; unwrap UVs before painting");
    }
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct UpdateTexture {
    texture: String,
    name: Option<String>,
    filter: Option<Filter>,
    /// Palette name to lock to; "" removes the lock.
    palette: Option<String>,
    /// New [width, height] (nearest-neighbour resample).
    resize: Option<[u32; 2]>,
}

fn update_texture(ctx: &mut Ctx, a: UpdateTexture) -> Result<Out> {
    let tid = ctx.doc.scene.resolve_texture(&a.texture)?;
    if let Some(p) = a.palette.as_deref().filter(|p| !p.is_empty()) {
        palette_named(&ctx.doc.scene, p).with_context(|| format!("unknown palette '{p}'"))?;
    }
    edit_scene(ctx, "update texture", |s| {
        let t = s.texture_mut(tid)?;
        if let Some(n) = &a.name {
            t.name = n.clone();
        }
        if let Some(f) = a.filter {
            t.filter = f;
        }
        if let Some(p) = &a.palette {
            t.palette = (!p.is_empty()).then(|| p.clone());
        }
        if let Some([w, h]) = a.resize {
            let mut r = polly_paint::resize_nearest(t, w.clamp(1, 2048), h.clamp(1, 2048));
            r.name = t.name.clone();
            r.filter = t.filter;
            r.palette = t.palette.clone();
            *t = r;
        }
        Ok(())
    })?;
    let t = &ctx.doc.scene.textures[&tid];
    Ok(Out::json(json!({"texture": t.name, "size": [t.width, t.height], "filter": t.filter, "palette": t.palette})))
}

#[derive(Deserialize, JsonSchema)]
struct GetTexture {
    texture: String,
    /// Upscale factor for the returned image (default: so the image is ~512 px).
    scale: Option<u32>,
}

fn get_texture(ctx: &mut Ctx, a: GetTexture) -> Result<Out> {
    let s = &ctx.doc.scene;
    let t = &s.textures[&s.resolve_texture(&a.texture)?];
    let palette = t.palette.as_deref().and_then(|p| palette_named(s, p));
    let stats = polly_paint::stats(t, palette.as_deref());
    let scale = a.scale.unwrap_or_else(|| (512 / t.width.max(t.height)).max(1)).clamp(1, 32);
    let big = polly_paint::resize_nearest(t, t.width * scale, t.height * scale);
    let png = polly_paint::to_png(&big);
    Ok(Out::json(json!({"texture": t.name, "palette": t.palette, "filter": t.filter, "shown_at_scale": scale, "stats": stats})).with_image(png))
}

#[derive(Deserialize, JsonSchema)]
struct ListPalettes {
    name: Option<String>,
}

fn list_palettes(ctx: &mut Ctx, a: ListPalettes) -> Result<Out> {
    let s = &ctx.doc.scene;
    let hex = |p: &[Rgba]| p.iter().map(|c| polly_paint::to_hex(*c)).collect::<Vec<_>>();
    if let Some(n) = a.name {
        let p = palette_named(s, &n).with_context(|| format!("unknown palette '{n}'"))?;
        return Ok(Out::json(json!({"name": n, "colors": hex(&p), "note": "palette index pN refers to colors[N]"})));
    }
    let builtin: Vec<_> = polly_paint::palette::builtin_names()
        .iter()
        .map(|n| json!({"name": n, "count": polly_paint::palette::builtin(n).map_or(0, |p| p.len())}))
        .collect();
    let scene: Vec<_> = s.palettes.iter().map(|(n, p)| json!({"name": n, "colors": hex(p)})).collect();
    Ok(Out::json(json!({"builtin": builtin, "scene": scene})))
}

#[derive(Deserialize, JsonSchema)]
struct SetPalette {
    name: String,
    /// Explicit colours.
    colors: Option<Vec<Rgba>>,
    /// Copy a built-in palette.
    builtin: Option<String>,
    /// Ramps: each [from, to, steps] appended (e.g. ["#202020", "#e0d0b0", 6]).
    ramps: Option<Vec<(Rgba, Rgba, usize)>>,
    /// Extract up to `max_colors` from this texture.
    from_texture: Option<String>,
    max_colors: Option<usize>,
}

fn set_palette(ctx: &mut Ctx, a: SetPalette) -> Result<Out> {
    let mut colors: Vec<Rgba> = a.colors.clone().unwrap_or_default();
    if let Some(b) = &a.builtin {
        colors.extend(polly_paint::palette::builtin(b).with_context(|| format!("unknown built-in palette '{b}'"))?);
    }
    for (from, to, n) in a.ramps.iter().flatten() {
        colors.extend(polly_paint::palette::ramp(*from, *to, (*n).clamp(2, 64)));
    }
    if let Some(t) = &a.from_texture {
        let s = &ctx.doc.scene;
        colors.extend(polly_paint::palette::extract(&s.textures[&s.resolve_texture(t)?], a.max_colors.unwrap_or(16)));
    }
    if colors.is_empty() {
        bail!("no colours given (colors, builtin, ramps or from_texture)");
    }
    let n = colors.len();
    let hex: Vec<_> = colors.iter().map(|c| polly_paint::to_hex(*c)).collect();
    edit_scene(ctx, "set palette", |s| {
        s.palettes.insert(a.name.clone(), colors);
        Ok(())
    })?;
    Ok(Out::json(json!({"palette": a.name, "count": n, "colors": hex})))
}

#[derive(Deserialize, JsonSchema)]
struct PaintPixels {
    texture: String,
    /// [[x, y, colour], ...]
    pixels: Vec<(i32, i32, Rgba)>,
    #[serde(default)]
    blend: Blend,
}

fn paint_pixels(ctx: &mut Ctx, a: PaintPixels) -> Result<Out> {
    let tid = ctx.doc.scene.resolve_texture(&a.texture)?;
    edit_texture(ctx, tid, "paint pixels", |t, s| {
        let p = tex_paint(s, tid, [0; 4], a.blend);
        let px: Vec<_> = a
            .pixels
            .iter()
            .map(|(x, y, c)| (*x, *y, p.palette.as_deref().map_or(*c, |pal| polly_paint::palette::nearest(pal, *c).1)))
            .collect();
        polly_paint::set_pixels(t, &px, a.blend)
    })
}

#[derive(Deserialize, JsonSchema)]
#[serde(tag = "shape", rename_all = "snake_case")]
enum Shape {
    Line { from: [i32; 2], to: [i32; 2] },
    Polyline { points: Vec<[i32; 2]> },
    Rect { min: [i32; 2], max: [i32; 2] },
    Ellipse { center: [i32; 2], radii: [i32; 2] },
    Polygon { points: Vec<[i32; 2]> },
    FloodFill { at: [i32; 2], #[serde(default)] tolerance: u8 },
}

#[derive(Deserialize, JsonSchema)]
struct Draw {
    texture: String,
    #[serde(flatten)]
    shape: Shape,
    color: Rgba,
    /// Line width in px (default 1).
    width: Option<u32>,
    /// Fill rect/ellipse/polygon (default false = outline).
    #[serde(default)]
    filled: bool,
    #[serde(default)]
    blend: Blend,
    /// Clip to the UV area of these faces...
    clip_faces: Option<Vec<String>>,
    /// ...of this object (default active).
    object: Option<String>,
}

fn face_polys(scene: &Scene, object: &Option<String>, ids: &Option<Vec<String>>) -> Result<Vec<Vec<[f32; 2]>>> {
    let id = resolve(scene, object)?;
    let sel = elements(scene, id, ids)?;
    let m = &scene.object(id)?.mesh;
    Ok(polly_paint::face_uv_polys(m, &sel.implied_faces(m)))
}

fn draw(ctx: &mut Ctx, a: Draw) -> Result<Out> {
    let tid = ctx.doc.scene.resolve_texture(&a.texture)?;
    let mask = match &a.clip_faces {
        Some(_) => {
            let polys = face_polys(&ctx.doc.scene, &a.object, &a.clip_faces)?;
            let t = &ctx.doc.scene.textures[&tid];
            Some(polly_paint::face_mask(t.width, t.height, &polys, 0))
        }
        None => None,
    };
    let w = a.width.unwrap_or(1);
    edit_texture(ctx, tid, "draw", |t, s| {
        let p = tex_paint(s, tid, a.color, a.blend).with_mask(mask);
        match &a.shape {
            Shape::Line { from, to } => polly_paint::line(t, *from, *to, w, &p),
            Shape::Polyline { points } => polly_paint::polyline(t, points, w, &p),
            Shape::Rect { min, max } => polly_paint::rect(t, *min, *max, a.filled, &p),
            Shape::Ellipse { center, radii } => polly_paint::ellipse(t, *center, *radii, a.filled, &p),
            Shape::Polygon { points } => polly_paint::polygon(t, points, a.filled, &p),
            Shape::FloodFill { at, tolerance } => polly_paint::flood_fill(t, *at, *tolerance, &p),
        }
    })
}

#[derive(Deserialize, JsonSchema)]
struct FillFaces {
    object: Option<String>,
    /// Faces (default: current selection; "all" for every face).
    elements: Option<Vec<String>>,
    /// e.g. {"type":"solid","color":"#7a4b2a"} or {"type":"bricks","mortar":"#3a3030","brick":["#a04030","#903828"],"size_px":[8,4]}
    fill: Fill,
    /// Texture (default: the faces' material texture).
    texture: Option<String>,
    /// Grow coverage by N texels to hide seams with nearest filtering (default 1).
    grow_px: Option<u32>,
    #[serde(default)]
    blend: Blend,
}

fn fill_faces(ctx: &mut Ctx, a: FillFaces) -> Result<Out> {
    let s = &ctx.doc.scene;
    let id = resolve(s, &a.object)?;
    let sel = elements(s, id, &a.elements)?;
    let mesh = s.object(id)?.mesh.clone();
    let faces = sel.implied_faces(&mesh);
    if faces.is_empty() {
        bail!("no faces given");
    }
    let tid = match &a.texture {
        Some(t) => s.resolve_texture(t)?,
        None => texture_of_faces(s, id, &faces)?,
    };
    let zero_uv = faces.iter().all(|f| mesh.faces[f].corners.iter().all(|c| c.uv == polly_core::glam::Vec2::ZERO));
    if zero_uv {
        bail!("these faces have no UVs yet; run unwrap or project_uv first");
    }
    let grow = a.grow_px.unwrap_or(1);
    let mut out = edit_texture(ctx, tid, "fill faces", |t, s| {
        let p = tex_paint(s, tid, [255; 4], a.blend);
        polly_paint::fill_faces(t, &mesh, &faces, &a.fill, grow, &p)
    })?;
    out.json["faces"] = json!(faces.len());
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RegionArg {
    All,
    /// Inclusive pixel rectangle.
    Rect { min: [i32; 2], max: [i32; 2] },
    /// UV area of faces of an object.
    Faces {
        object: Option<String>,
        elements: Option<Vec<String>>,
        #[serde(default)]
        grow: u32,
    },
}

#[derive(Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case")]
enum RegionOp {
    Gradient {
        from: Rgba,
        to: Rgba,
        dir: GradientDir,
        #[serde(default)]
        dither: Dither,
        /// Posterize into N bands (retro banding).
        steps: Option<u32>,
    },
    Dither { a: Rgba, b: Rgba, density: f32, #[serde(default = "bayer4")] pattern: Dither },
    Noise { colors: Vec<Rgba>, #[serde(default)] seed: u64, #[serde(default = "four")] scale: f32 },
    /// 1px outline around differing areas (pixel-art style).
    Outline { color: Rgba },
    /// Inner border of the region (e.g. panel lines around faces).
    Border { color: Rgba, #[serde(default = "one")] width: u32 },
    /// Hue shift (degrees) and saturation/value multipliers.
    Adjust { #[serde(default)] hue_deg: f32, #[serde(default = "onef")] saturation: f32, #[serde(default = "onef")] value: f32 },
    ReplaceColors { map: Vec<(Rgba, Rgba)> },
}

fn bayer4() -> Dither {
    Dither::Bayer4
}
fn four() -> f32 {
    4.0
}
fn one() -> u32 {
    1
}
fn onef() -> f32 {
    1.0
}

#[derive(Deserialize, JsonSchema)]
struct PaintRegion {
    texture: Option<String>,
    region: RegionArg,
    effect: RegionOp,
    #[serde(default)]
    blend: Blend,
}

fn paint_region(ctx: &mut Ctx, a: PaintRegion) -> Result<Out> {
    let s = &ctx.doc.scene;
    let region = match &a.region {
        RegionArg::All => Region::All,
        RegionArg::Rect { min, max } => Region::Rect { min: *min, max: *max },
        RegionArg::Faces { object, elements: ids, grow } => Region::Faces { polys: face_polys(s, object, ids)?, grow: *grow },
    };
    let tid = match (&a.texture, &a.region) {
        (Some(t), _) => s.resolve_texture(t)?,
        (None, RegionArg::Faces { object, elements: ids, .. }) => {
            let id = resolve(s, object)?;
            let sel = elements(s, id, ids)?;
            texture_of_faces(s, id, &sel.implied_faces(&s.object(id)?.mesh))?
        }
        _ => bail!("pass `texture`"),
    };
    edit_texture(ctx, tid, "paint region", |t, s| {
        let p = tex_paint(s, tid, [255; 4], a.blend);
        match &a.effect {
            RegionOp::Gradient { from, to, dir, dither, steps } => polly_paint::gradient(t, &region, *from, *to, dir, *dither, *steps, &p),
            RegionOp::Dither { a: ca, b, density, pattern } => polly_paint::dither_fill(t, &region, *ca, *b, *density, *pattern, &p),
            RegionOp::Noise { colors, seed, scale } => polly_paint::noise(t, &region, colors, *seed, *scale, &p),
            RegionOp::Outline { color } => polly_paint::outline(t, &region, *color),
            RegionOp::Border { color, width } => polly_paint::border(t, &region, *width, &Paint { color: *color, ..p }),
            RegionOp::Adjust { hue_deg, saturation, value } => polly_paint::adjust(t, &region, *hue_deg, *saturation, *value, &p),
            RegionOp::ReplaceColors { map } => polly_paint::replace_colors(t, &region, map, &p),
        }
    })
}

#[derive(Deserialize, JsonSchema)]
struct UvStroke {
    texture: String,
    /// Polyline in UV space (0..1, v down).
    points: Vec<[f32; 2]>,
    color: Rgba,
    #[serde(default = "onef")]
    radius_px: f32,
    #[serde(default)]
    blend: Blend,
}

fn paint_uv_stroke(ctx: &mut Ctx, a: UvStroke) -> Result<Out> {
    let tid = ctx.doc.scene.resolve_texture(&a.texture)?;
    edit_texture(ctx, tid, "uv stroke", |t, s| {
        let p = tex_paint(s, tid, a.color, a.blend);
        polly_paint::stroke_uv(t, &a.points, a.radius_px, &p)
    })
}

#[derive(Deserialize, JsonSchema)]
struct Quantize {
    texture: String,
    /// Palette name (default: the texture's locked palette).
    palette: Option<String>,
    #[serde(default)]
    dither: Dither,
}

fn quantize(ctx: &mut Ctx, a: Quantize) -> Result<Out> {
    let s = &ctx.doc.scene;
    let tid = s.resolve_texture(&a.texture)?;
    let name = a.palette.clone().or_else(|| s.textures[&tid].palette.clone()).context("no palette given and the texture has none")?;
    let pal = palette_named(s, &name).with_context(|| format!("unknown palette '{name}'"))?;
    edit_texture(ctx, tid, "quantize", |t, _| polly_paint::quantize(t, &pal, a.dither))
}

#[derive(Deserialize, JsonSchema)]
struct Bake {
    object: Option<String>,
    elements: Option<Vec<String>>,
    texture: Option<String>,
}

fn bake_vertex_colors(ctx: &mut Ctx, a: Bake) -> Result<Out> {
    let s = &ctx.doc.scene;
    let id = resolve(s, &a.object)?;
    let ids = a.elements.clone().or(Some(vec!["all".into()]));
    let sel = elements(s, id, &ids)?;
    let mesh = s.object(id)?.mesh.clone();
    let faces = sel.implied_faces(&mesh);
    let tid = match &a.texture {
        Some(t) => s.resolve_texture(t)?,
        None => texture_of_faces(s, id, &faces)?,
    };
    edit_texture(ctx, tid, "bake vertex colors", |t, _| polly_paint::bake_vertex_colors(t, &mesh, &faces))
}

#[derive(Deserialize, JsonSchema)]
struct Import {
    path: String,
    name: Option<String>,
    palette: Option<String>,
}

fn import_texture(ctx: &mut Ctx, a: Import) -> Result<Out> {
    let bytes = std::fs::read(&a.path).with_context(|| format!("reading {}", a.path))?;
    let name = a.name.clone().unwrap_or_else(|| {
        std::path::Path::new(&a.path).file_stem().map_or("texture".into(), |s| s.to_string_lossy().into_owned())
    });
    let mut t = polly_paint::from_png(&name, &bytes)?;
    t.palette = a.palette.clone();
    let (w, h) = (t.width, t.height);
    let tid = edit_scene(ctx, "import texture", |s| Ok(s.add_texture(t)))?;
    Ok(Out::json(json!({"texture": name, "id": format!("t{tid}"), "size": [w, h]})))
}

#[derive(Deserialize, JsonSchema)]
struct ExportTex {
    texture: String,
    path: String,
}

fn export_texture(ctx: &mut Ctx, a: ExportTex) -> Result<Out> {
    let s = &ctx.doc.scene;
    let t = &s.textures[&s.resolve_texture(&a.texture)?];
    std::fs::write(&a.path, polly_paint::to_png(t)).with_context(|| format!("writing {}", a.path))?;
    Ok(ok(format!("wrote {}", a.path)))
}

/// Faces of `obj` whose material uses texture `tid`.
pub fn faces_with_texture(scene: &Scene, obj: u32, tid: TextureId) -> BTreeSet<FaceId> {
    let o = &scene.objects[&obj];
    o.mesh
        .faces
        .iter()
        .filter(|(_, f)| scene.materials.get(&f.material).and_then(|m| m.texture) == Some(tid))
        .map(|(id, _)| *id)
        .collect()
}

#[derive(Deserialize, JsonSchema)]
struct BakeShading {
    object: Option<String>,
    /// Faces to shade (default: all faces using the texture).
    elements: Option<Vec<String>>,
    /// Texture (default: the object's textured material).
    texture: Option<String>,
    #[serde(flatten)]
    shading: polly_paint::Shading,
    /// What casts occlusion: "self" (default), "scene" (all visible objects, with gear slots at
    /// their default variants unless `variants` says otherwise), or a list of object names.
    occluders: Option<Occluders>,
    /// Gear loadout for "scene" occluders: {slot: variant}; other slots use their default variant.
    variants: Option<std::collections::BTreeMap<String, String>>,
    /// Extend shading this many texels past UV island edges (default 1).
    grow_px: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
enum Occluders {
    /// "self" or "scene".
    Mode(String),
    /// These objects (and the baked object itself).
    Objects(Vec<String>),
}

fn bake_shading(ctx: &mut Ctx, a: BakeShading) -> Result<Out> {
    if a.shading.ao.is_none() && a.shading.height.is_none() {
        bail!("give ao (e.g. {{\"ao\":{{}}}}) and/or height (e.g. {{\"height\":{{}}}})");
    }
    let s = &ctx.doc.scene;
    let id = resolve(s, &a.object)?;
    // The baked object whole (no parts hidden by variants), in its static pose.
    let posed = polly_core::anim::pose_scene(s, &[]);
    let mesh = posed.objects[&id].mesh.clone();
    let tid = match &a.texture {
        Some(t) => s.resolve_texture(t)?,
        None => texture_of_faces(s, id, &mesh.faces.keys().copied().collect())?,
    };
    let faces = match &a.elements {
        Some(_) => elements(s, id, &a.elements)?.implied_faces(&mesh),
        None => faces_with_texture(s, id, tid),
    };
    if faces.is_empty() {
        bail!("no faces use that texture");
    }
    // Occluders in the object's local space.
    let mut occ = (*mesh).clone();
    let others: Vec<u32> = match &a.occluders {
        None => vec![],
        Some(Occluders::Mode(m)) if m == "self" || m == "scene" => vec![],
        Some(Occluders::Mode(other)) => bail!("occluders must be \"self\", \"scene\" or a list of objects, not '{other}'"),
        Some(Occluders::Objects(names)) => names.iter().map(|n| s.resolve(Some(n))).collect::<polly_core::Result<_>>()?,
    };
    let scene_mode = matches!(&a.occluders, Some(Occluders::Mode(m)) if m == "scene");
    if a.variants.is_some() && !scene_mode {
        bail!("`variants` only applies with occluders \"scene\"");
    }
    // Default loadout, overridden by `variants`.
    let mut choice: std::collections::BTreeMap<String, String> = s.slots.iter().map(|(n, sl)| (n.clone(), sl.default.clone())).collect();
    choice.extend(super::variants::choice(s, &a.variants)?);
    let shown = polly_core::variants::display(s, &[], &choice);
    let occluders: Vec<u32> = if scene_mode {
        shown.objects.iter().filter(|(oid, o)| **oid != id && o.visible).map(|(oid, _)| *oid).collect()
    } else {
        others.into_iter().filter(|o| *o != id).collect()
    };
    let inv = posed.world_matrix(id).inverse();
    let mut swappable = Vec::new();
    for oid in &occluders {
        // Explicit occluders are used whole, even if hidden or not chosen.
        let src = if scene_mode { &shown } else { &posed };
        let o = &src.objects[oid];
        let mut m = (*o.mesh).clone();
        m.transform(inv * src.world_matrix(*oid));
        occ.append(&m);
        if let Some((slot, v)) = s.variant_of(*oid) {
            swappable.push(format!("{} (slot '{slot}', variant '{v}')", o.name));
        }
    }
    let bvh = polly_core::bvh::Bvh::new(&occ, polly_core::glam::Mat4::IDENTITY);
    let (w, h) = (s.textures[&tid].width, s.textures[&tid].height);
    let map = polly_paint::SurfaceMap::new(&mesh, &faces, w, h, a.grow_px.unwrap_or(1));
    let mut out = edit_texture(ctx, tid, "bake shading", |t, s| {
        let p = tex_paint(s, tid, [0; 4], Blend::Replace);
        polly_paint::bake_shading(t, &map, &bvh, &a.shading, &p)
    })?;
    out.json["faces"] = json!(faces.len());
    if !occluders.is_empty() {
        out.json["occluders"] = json!(occluders.iter().map(|o| ctx.doc.scene.objects[o].name.clone()).collect::<Vec<_>>());
    }
    if !swappable.is_empty() {
        out.json["warning"] = json!(format!(
            "baked shadows from swappable gear: {}. They stay in the texture when that gear is swapped out; bake with occluders that are always present, or one texture per loadout.",
            swappable.join(", ")
        ));
    }
    Ok(out)
}
