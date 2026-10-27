//! Custom shaders: WESL files in the project's asset directory, used by
//! materials (`shader: {name, params}`) in Polly's renders and, through
//! bevy_polly, in the game.

use anyhow::{Context, Result, bail};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use super::material::valid_shader_name;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![def(
        "write_shader",
        "Write a custom fragment shader (WESL, Bevy 0.20) to <asset_dir>/shaders/<name>.wesl; materials use it with shader {name, params}. Without `source` writes a template that does standard PBR lighting with the parameter slots listed (params are packed alphabetically into `polly.values[i]`, vec4 each, binding 100). The shader reloads on the next render; compile errors come back as render notes. The game loads the same file through bevy_polly.",
        false,
        write_shader,
    )]
}

#[derive(Deserialize, JsonSchema)]
struct WriteShader {
    /// Shader name (letters, digits, '_' or '-').
    name: String,
    /// WESL source; omitted = a template to edit.
    source: Option<String>,
    /// The game's asset directory (saved in the scene). Default: the scene's, or `assets` next to the project file.
    asset_dir: Option<String>,
    /// Parameter names to list in the template's slot table (default: the params of materials using this shader).
    #[serde(default)]
    params: Vec<String>,
}

fn write_shader(ctx: &mut Ctx, a: WriteShader) -> Result<Out> {
    if !valid_shader_name(&a.name) {
        bail!("shader name '{}' must be letters, digits, '_' or '-'", a.name);
    }
    let dir = match (&a.asset_dir, &ctx.doc.scene.asset_dir, ctx.project_path.as_ref()) {
        (Some(d), _, _) => d.clone(),
        (None, Some(d), _) => d.clone(),
        (None, None, Some(p)) => p.parent().unwrap_or(std::path::Path::new(".")).join("assets").to_string_lossy().into_owned(),
        (None, None, None) => bail!("no asset directory: pass asset_dir (the game's assets folder) or save the project first"),
    };
    let dir = std::path::absolute(&dir).with_context(|| format!("bad asset directory '{dir}'"))?.to_string_lossy().into_owned();
    let mut params: Vec<String> = a.params.clone();
    if params.is_empty() {
        let set: std::collections::BTreeSet<String> = ctx
            .doc
            .scene
            .materials
            .values()
            .filter_map(|m| m.shader.as_ref().filter(|s| s.name == a.name))
            .flat_map(|s| s.params.keys().cloned())
            .collect();
        params = set.into_iter().collect();
    }
    params.sort();
    let source = a.source.clone().unwrap_or_else(|| bevy_polly::shader_template(&a.name, &params));
    let path = std::path::Path::new(&dir).join("shaders").join(format!("{}.wesl", a.name));
    std::fs::create_dir_all(path.parent().unwrap()).with_context(|| format!("creating {}", path.parent().unwrap().display()))?;
    std::fs::write(&path, &source).with_context(|| format!("writing {}", path.display()))?;
    if ctx.doc.scene.asset_dir.as_deref() != Some(dir.as_str()) {
        let d = dir.clone();
        edit_scene(ctx, "set asset directory", move |s| {
            s.asset_dir = Some(d);
            Ok(())
        })?;
    }
    let slots: Vec<_> = params.iter().enumerate().map(|(i, p)| json!(format!("polly.values[{i}] = {p}"))).collect();
    Ok(Out::json(json!({
        "path": path.to_string_lossy(),
        "asset_dir": dir,
        "template": a.source.is_none(),
        "slots": slots,
        "note": "render to compile it; errors are reported in the render's notes",
    })))
}
