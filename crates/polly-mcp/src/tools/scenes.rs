use std::time::Duration;

use anyhow::{Result, bail};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::ok;
use crate::overview::render_grid;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "create_scene",
            "Create a new, empty scene and return its name. Use one scene per independent model (e.g. per subagent), then pass `scene: \"<name>\"` on every call that should work in it.",
            false,
            create_scene,
        ),
        def("list_scenes", "List all scenes with object and triangle counts.", true, list_scenes),
        def("delete_scene", "Delete a scene and everything in it (not undoable). The \"main\" scene cannot be deleted.", false, delete_scene),
        def(
            "render_overview",
            "Render all scenes (or the listed ones) side by side in one captioned grid image, to review many models at once.",
            true,
            render_overview,
        ),
    ]
}

/// How long to wait for a scene another call is busy with.
const WAIT: Duration = Duration::from_secs(2);

#[derive(Deserialize, JsonSchema)]
struct CreateScene {
    /// Scene name (default: the first free "scene-N").
    name: Option<String>,
}

fn create_scene(ctx: &mut Ctx, a: CreateScene) -> Result<Out> {
    let name = ctx.workspace.create(a.name.as_deref())?;
    Ok(Out::json(json!({
        "scene": name,
        "note": format!("pass \"scene\": \"{name}\" on every tool call that should work in this scene"),
    })))
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn list_scenes(ctx: &mut Ctx, _: Empty) -> Result<Out> {
    let (snaps, busy) = ctx.workspace.snapshots(None, Some((ctx.scene, ctx.doc)), WAIT)?;
    let rows: Vec<_> = snaps
        .iter()
        .map(|s| {
            json!({
                "scene": s.name,
                "objects": s.scene.objects.values().map(|o| o.name.clone()).collect::<Vec<_>>(),
                "tris": s.scene.objects.values().map(|o| o.mesh.triangle_count()).sum::<usize>(),
                "edits": s.version,
            })
        })
        .collect();
    let mut j = json!({"current": ctx.scene, "scenes": rows});
    if !busy.is_empty() {
        j["busy"] = json!(busy);
    }
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct DeleteScene {
    name: String,
}

fn delete_scene(ctx: &mut Ctx, a: DeleteScene) -> Result<Out> {
    ctx.workspace.remove(&a.name)?;
    Ok(ok(format!("deleted scene '{}'", a.name)))
}

#[derive(Deserialize, JsonSchema)]
struct Overview {
    /// Scenes to show (default: all).
    scenes: Option<Vec<String>>,
    /// Tile size in px (default 256; total image is capped at about 2048 px wide).
    tile_size: Option<u32>,
}

fn render_overview(ctx: &mut Ctx, a: Overview) -> Result<Out> {
    let (snaps, busy) = ctx.workspace.snapshots(a.scenes.as_deref(), Some((ctx.scene, ctx.doc)), WAIT)?;
    if snaps.is_empty() {
        bail!("no scenes to show");
    }
    let cols = (snaps.len() as f32).sqrt().ceil() as u32;
    let tile = a.tile_size.unwrap_or(256).clamp(96, 512).min(2048 / cols);
    let mut renderer = ctx.services.renderer()?;
    let (img, rects) = render_grid(&mut renderer, &ctx.services.tiles, &snaps, cols, tile, None)?;
    drop(renderer);
    let mut j = json!({"tiles": rects, "size": [img.width(), img.height()]});
    if !busy.is_empty() {
        j["busy_scenes_not_shown"] = json!(busy);
    }
    Ok(Out::json(j).with_image(super::view::png(&img)?))
}
