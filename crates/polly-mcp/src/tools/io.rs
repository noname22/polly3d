use std::path::PathBuf;

use anyhow::{Context, Result};
use polly_export::{ExportOptions, export_glb, load_project, save_project};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::ok;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def("save_project", "Save the scene (meshes, materials, textures, palettes) to a .polly file.", true, save),
        def("load_project", "Load a .polly project into the scene, replacing its contents (clears its undo history).", false, load),
        def(
            "export_glb",
            "Export to binary glTF (.glb) with embedded textures; texture filtering (nearest/linear) is carried in the sampler. Optionally only some objects (with their children).",
            true,
            export,
        ),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct PathArg {
    /// File path (relative paths are resolved against the server's working directory). Defaults to the last used project path.
    path: Option<String>,
}

fn project_path(ctx: &Ctx, p: &Option<String>) -> Result<PathBuf> {
    match p {
        Some(p) => Ok(PathBuf::from(p)),
        None => ctx.project_path.clone().context("no path given and no project path known yet for this scene"),
    }
}

fn save(ctx: &mut Ctx, a: PathArg) -> Result<Out> {
    let mut path = project_path(ctx, &a.path)?;
    if path.extension().is_none() {
        path.set_extension("polly");
    }
    let bytes = save_project(&ctx.doc.scene)?;
    std::fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
    *ctx.project_path = Some(path.clone());
    Ok(ok(format!("saved {} ({} bytes)", std::path::absolute(&path)?.display(), bytes.len())))
}

fn load(ctx: &mut Ctx, a: PathArg) -> Result<Out> {
    let path = project_path(ctx, &a.path)?;
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let scene = load_project(&bytes)?;
    ctx.doc.replace(scene);
    *ctx.project_path = Some(path.clone());
    Ok(Out::json(json!({"loaded": path.display().to_string(), "objects": ctx.doc.scene.objects.len()})))
}

#[derive(Deserialize, JsonSchema)]
struct Export {
    path: String,
    /// Object names to export (default: all visible).
    objects: Option<Vec<String>>,
}

fn export(ctx: &mut Ctx, a: Export) -> Result<Out> {
    let objects = a
        .objects
        .as_ref()
        .map(|names| names.iter().map(|n| ctx.doc.scene.resolve(Some(n))).collect::<polly_core::Result<Vec<_>>>())
        .transpose()?;
    let bytes = export_glb(&ctx.doc.scene, &ExportOptions { objects })?;
    let mut path = PathBuf::from(&a.path);
    if path.extension().is_none() {
        path.set_extension("glb");
    }
    std::fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
    let msg = format!("exported {} ({} bytes)", std::path::absolute(&path)?.display(), bytes.len());
    // Shared-armature skins are not exported yet (planned): say so instead of failing silently.
    let bound: Vec<&str> = ctx.doc.scene.objects.values().filter(|o| o.skin.is_some()).map(|o| o.name.as_str()).collect();
    if bound.is_empty() {
        return Ok(ok(msg));
    }
    Ok(Out::json(json!({
        "result": msg,
        "warning": format!("not yet supported: meshes bound to an armature export unskinned, in their rest pose: {}", bound.join(", ")),
    })))
}
