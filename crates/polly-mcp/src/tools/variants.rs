//! Slots and variants: swappable hair, helmets, weapons...

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use polly_core::scene::Scene;
use polly_core::variants::{HideRule, Variant};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "add_variant",
            "Define a variant of a gear slot, e.g. slot \"hair\" variant \"braids\" = objects [\"Hair_Braids\"]. The slot is created if needed; its first variant becomes the default (shown when the model is loaded) and the active one. A variant may have no objects (\"none\"), may share objects with other variants of the same slot (\"suit\" = [\"Blouse\", \"Blazer\"] next to \"blouse\" = [\"Blouse\"]), and may hide named parts of other objects while active (a helmet hiding \"hair_top\"; auto_hide finds covered faces for you). Re-adding a variant replaces its objects; its hide rules are kept unless `hides` is given.",
            false,
            add_variant,
        ),
        def(
            "auto_hide",
            "Find the faces of `body` covered by `garment` (within max_distance outside the body, or slightly inside it) and hide them whenever the garment is shown: adds a hide rule (named `name`, default \"under_<garment>\") to every variant containing the garment, so skin can't poke through. Overlapping garments are fine: each keeps its own rule, and a face is hidden while any chosen garment covers it. Re-run after reshaping to update.",
            false,
            auto_hide,
        ),
        def("set_variant", "Make a variant the active one of its slot (what Polly shows and renders).", false, set_variant),
        def("set_default_variant", "Choose which variant of a slot is visible when the exported model is loaded.", false, set_default_variant),
        def("remove_variant", "Remove a variant from a slot (its objects stay in the scene, no longer in a slot).", false, remove_variant),
        def("delete_slot", "Remove a slot and its variants (objects stay in the scene).", false, delete_slot),
        def("list_slots", "List gear slots with their variants, objects, hide rules, default and active variant.", true, list_slots),
        def(
            "render_variants",
            "Render every variant of a slot side by side (one captioned tile each), with the other slots at their active variants.",
            true,
            render_variants,
        ),
    ]
}

/// Validate a variant choice ({slot: variant}) against the scene.
pub fn choice(scene: &Scene, c: &Option<BTreeMap<String, String>>) -> Result<BTreeMap<String, String>> {
    let c = c.clone().unwrap_or_default();
    for (slot, v) in &c {
        let Some(s) = scene.slots.get(slot) else {
            bail!("no slot '{slot}' (slots: {})", scene.slots.keys().cloned().collect::<Vec<_>>().join(", "));
        };
        s.variant(v)?;
    }
    Ok(c)
}

fn slot_json(scene: &Scene, name: &str) -> Value {
    let s = &scene.slots[name];
    let obj = |id: &u32| scene.objects.get(id).map_or(format!("o{id}"), |o| o.name.clone());
    let variants: Vec<Value> = s
        .variants
        .iter()
        .map(|v| {
            let mut j = json!({"name": v.name, "objects": v.objects.iter().map(obj).collect::<Vec<_>>()});
            if !v.hides.is_empty() {
                j["hides"] = json!(v
                    .hides
                    .iter()
                    .map(|h| {
                        let mut r = json!({"object": obj(&h.object), "part": h.part});
                        if let Some(f) = &h.faces {
                            r["faces"] = json!(polly_core::ids::compress("f", f.iter().copied()));
                        }
                        r
                    })
                    .collect::<Vec<_>>());
            }
            j
        })
        .collect();
    json!({"slot": name, "default": s.default, "active": s.active, "variants": variants})
}

#[derive(Deserialize, JsonSchema)]
struct HideArg {
    object: String,
    part: String,
}

#[derive(Deserialize, JsonSchema)]
struct AddVariant {
    slot: String,
    variant: String,
    /// Objects shown by this variant (may be empty).
    #[serde(default)]
    objects: Vec<String>,
    /// Parts of other objects to hide while this variant is active (replaces the variant's hide rules; omit to keep them).
    hides: Option<Vec<HideArg>>,
    /// Also make it the slot's default.
    #[serde(default)]
    default: bool,
}

fn add_variant(ctx: &mut Ctx, a: AddVariant) -> Result<Out> {
    let s = &ctx.doc.scene;
    let objects = a.objects.iter().map(|n| s.resolve(Some(n))).collect::<polly_core::Result<Vec<_>>>()?;
    let hides = match &a.hides {
        Some(hs) => hs
            .iter()
            .map(|h| Ok(HideRule { object: s.resolve(Some(&h.object))?, part: h.part.clone(), faces: None }))
            .collect::<polly_core::Result<Vec<_>>>()?,
        None => s.slots.get(&a.slot).and_then(|sl| sl.variant(&a.variant).ok()).map(|v| v.hides.clone()).unwrap_or_default(),
    };
    let created = !s.slots.contains_key(&a.slot);
    edit_scene(ctx, "add variant", |s| {
        s.set_variant_def(&a.slot, Variant { name: a.variant.clone(), objects, hides })?;
        if a.default {
            s.slots.get_mut(&a.slot).unwrap().default = a.variant.clone();
        }
        Ok(())
    })?;
    let mut j = slot_json(&ctx.doc.scene, &a.slot);
    if created {
        j["note"] = json!(format!("created slot '{}'", a.slot));
    }
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct SlotVariant {
    slot: String,
    variant: String,
}

fn check(scene: &Scene, slot: &str, variant: &str) -> Result<()> {
    choice(scene, &Some(BTreeMap::from([(slot.to_string(), variant.to_string())])))?;
    Ok(())
}

fn set_variant(ctx: &mut Ctx, a: SlotVariant) -> Result<Out> {
    check(&ctx.doc.scene, &a.slot, &a.variant)?;
    edit_scene(ctx, "set variant", |s| {
        s.slots.get_mut(&a.slot).unwrap().active = a.variant.clone();
        Ok(())
    })?;
    Ok(Out::json(slot_json(&ctx.doc.scene, &a.slot)))
}

fn set_default_variant(ctx: &mut Ctx, a: SlotVariant) -> Result<Out> {
    check(&ctx.doc.scene, &a.slot, &a.variant)?;
    edit_scene(ctx, "set default variant", |s| {
        s.slots.get_mut(&a.slot).unwrap().default = a.variant.clone();
        Ok(())
    })?;
    Ok(Out::json(slot_json(&ctx.doc.scene, &a.slot)))
}

fn remove_variant(ctx: &mut Ctx, a: SlotVariant) -> Result<Out> {
    check(&ctx.doc.scene, &a.slot, &a.variant)?;
    edit_scene(ctx, "remove variant", |s| {
        let slot = s.slots.get_mut(&a.slot).unwrap();
        slot.variants.retain(|v| v.name != a.variant);
        let first = slot.variants.first().map(|v| v.name.clone()).unwrap_or_default();
        if slot.default == a.variant {
            slot.default = first.clone();
        }
        if slot.active == a.variant {
            slot.active = first;
        }
        if slot.variants.is_empty() {
            s.slots.remove(&a.slot);
        }
        Ok(())
    })?;
    Ok(ok(format!("removed variant '{}' from '{}'", a.variant, a.slot)))
}

#[derive(Deserialize, JsonSchema)]
struct SlotArg {
    slot: String,
}

fn delete_slot(ctx: &mut Ctx, a: SlotArg) -> Result<Out> {
    if !ctx.doc.scene.slots.contains_key(&a.slot) {
        bail!("no slot '{}'", a.slot);
    }
    edit_scene(ctx, "delete slot", |s| {
        s.slots.remove(&a.slot);
        Ok(())
    })?;
    Ok(ok(format!("deleted slot '{}'", a.slot)))
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn list_slots(ctx: &mut Ctx, _: Empty) -> Result<Out> {
    let s = &ctx.doc.scene;
    Ok(Out::json(json!({"slots": s.slots.keys().map(|k| slot_json(s, k)).collect::<Vec<_>>()})))
}

#[derive(Deserialize, JsonSchema)]
struct RenderVariants {
    slot: String,
    #[serde(default)]
    view: polly_render::View,
    /// Tile size in px (default 256).
    tile_size: Option<u32>,
    /// Other slots' variants (default: their active ones).
    variants: Option<BTreeMap<String, String>>,
    /// Pose with clips, as for render.
    pose: Option<Vec<super::anim::PlayArg>>,
    overlays: Option<polly_render::Overlays>,
}

fn render_variants(ctx: &mut Ctx, a: RenderVariants) -> Result<Out> {
    let s = &ctx.doc.scene;
    let Some(slot) = s.slots.get(&a.slot) else { bail!("no slot '{}'", a.slot) };
    let base = choice(s, &a.variants)?;
    let plays = super::anim::plays(s, a.pose.as_deref().unwrap_or(&[]))?;
    let tile = a.tile_size.unwrap_or(256).clamp(96, 512);
    let opts = polly_render::RenderOptions {
        view: a.view.clone(),
        width: tile,
        height: tile,
        overlays: a.overlays.clone().unwrap_or(polly_render::Overlays { selection: false, ..Default::default() }),
        legend: false,
        ..Default::default()
    };
    let names: Vec<String> = slot.variants.iter().map(|v| v.name.clone()).collect();
    let cols = (names.len() as f32).sqrt().ceil() as u32;
    let rows = (names.len() as u32).div_ceil(cols);
    let gap = 4;
    let mut img = image::RgbaImage::from_pixel(cols * tile + gap * (cols + 1), rows * tile + gap * (rows + 1), image::Rgba([18, 19, 23, 255]));
    let mut renderer = ctx.services.renderer()?;
    for (i, v) in names.iter().enumerate() {
        let mut c = base.clone();
        c.insert(a.slot.clone(), v.clone());
        let shown = polly_core::variants::display(s, &plays, &c);
        let tile_img = renderer.render(&shown, &opts)?.image;
        let (x, y) = (gap + (i as u32 % cols) * (tile + gap), gap + (i as u32 / cols) * (tile + gap));
        image::imageops::overlay(&mut img, &tile_img, x as i64, y as i64);
        let mut label = v.clone();
        if *v == slot.default {
            label.push_str(" (default)");
        }
        let r = polly_render::canvas::label_rect(&label, 1, 0.0, 0.0);
        let r = polly_render::canvas::Rect { x: x as i32 + 4, y: y as i32 + 4, ..r };
        polly_render::canvas::boxed_label(&mut img, r, &label, 1, [20, 20, 26, 220], polly_render::canvas::WHITE);
    }
    drop(renderer);
    Ok(Out::json(json!({"slot": a.slot, "variants": names})).with_image(super::view::png(&img)?))
}

#[derive(Deserialize, JsonSchema)]
struct AutoHide {
    /// The covering object (must be in a gear slot).
    garment: String,
    /// The object whose covered faces get hidden.
    body: String,
    /// How far outside the body the garment may be and still count as covering (default 0.05).
    max_distance: Option<f32>,
    /// Rule name (default "under_<garment>").
    name: Option<String>,
}

fn auto_hide(ctx: &mut Ctx, a: AutoHide) -> Result<Out> {
    use polly_core::glam::Vec3;
    let s = &ctx.doc.scene;
    let garment = s.resolve(Some(&a.garment))?;
    let body = s.resolve(Some(&a.body))?;
    if garment == body {
        bail!("garment and body must be different objects");
    }
    let Some((slot, variants)) = s.variants_of(garment) else {
        bail!("'{}' is in no gear slot: add_variant it first, then auto_hide", s.object(garment)?.name);
    };
    let (slot, variants): (String, Vec<String>) = (slot.to_string(), variants.into_iter().map(str::to_string).collect());
    let max = a.max_distance.unwrap_or(0.05);
    if !(max > 0.0) {
        bail!("max_distance must be > 0");
    }
    // The garment in the body's space, as a ray-cast target (rest pose).
    let mut g = (*s.object(garment)?.mesh).clone();
    g.transform(s.world_matrix(body).inverse() * s.world_matrix(garment));
    let bvh = polly_core::bvh::Bvh::new(&g, polly_core::glam::Mat4::IDENTITY);
    let bm = &s.object(body)?.mesh;
    let back = max * 0.5;
    let covered: BTreeSet<polly_core::mesh::FaceId> = bm
        .faces
        .iter()
        .filter(|(_, f)| {
            let n = bm.face_normal(f);
            let c = bm.face_center(f);
            let samples = std::iter::once(c).chain(bm.face_positions(f).into_iter().map(|p: Vec3| c.lerp(p, 0.7)));
            samples.into_iter().all(|p| bvh.any_hit(p - n * back, n, max + back))
        })
        .map(|(id, _)| *id)
        .collect();
    let name = a.name.clone().unwrap_or_else(|| format!("under_{}", s.objects[&garment].name));
    if covered.is_empty() {
        bail!("no faces of '{}' are covered by '{}' within {max} (try a larger max_distance)", a.body, a.garment);
    }
    let n = covered.len();
    edit_scene(ctx, "auto hide", |s| {
        let sl = s.slots.get_mut(&slot).unwrap();
        for v in sl.variants.iter_mut().filter(|v| variants.contains(&v.name)) {
            v.hides.retain(|h| !(h.object == body && h.part == name));
            v.hides.push(HideRule { object: body, part: name.clone(), faces: Some(covered.clone()) });
        }
        Ok(())
    })?;
    let mut j = slot_json(&ctx.doc.scene, &slot);
    j["hidden_faces"] = json!(n);
    j["rule"] = json!(name);
    j["hint"] = json!(format!("render_variants slot '{slot}' to check for gaps or poke-through"));
    Ok(Out::json(j))
}
