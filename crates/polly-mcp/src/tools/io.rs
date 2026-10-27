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
        def(
            "export_mesh",
            "Write an object's geometry to a .obj or .json file for scripts: vertices, faces (index loops), per-face uvs/parts/smooth, and in JSON the Polly ids of each row (vertex_ids, face_ids) to refer back to elements. No row limits, unlike list_verts. add_mesh reads the same files.",
            true,
            export_mesh,
        ),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct ExportMesh {
    /// Object name or id (default: active object).
    object: Option<String>,
    /// .obj or .json file.
    path: String,
    /// "object" (default: the mesh's own coordinates) or "world" (with its transform and parents).
    space: Option<String>,
}

fn export_mesh(ctx: &mut Ctx, a: ExportMesh) -> Result<Out> {
    use polly_export::meshio;
    let s = &ctx.doc.scene;
    let id = s.resolve(a.object.as_deref())?;
    let o = s.object(id)?;
    let xform = match a.space.as_deref().unwrap_or("object") {
        "object" => polly_core::glam::Mat4::IDENTITY,
        "world" => s.world_matrix(id),
        other => anyhow::bail!("unknown space '{other}' (object, world)"),
    };
    let data = meshio::data_from_mesh(&o.mesh, xform);
    let path = PathBuf::from(&a.path);
    meshio::write_mesh_file(&path, &data, &o.name)?;
    Ok(Out::json(json!({
        "result": format!("wrote {}", std::path::absolute(&path)?.display()),
        "vertices": data.vertices.len(),
        "faces": data.faces.len(),
    })))
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
    /// Subdivision level for every object (0 = cages); default: each object's export level (set_subdivision).
    subdivision: Option<u8>,
    /// Levels of detail as fractions of each mesh's triangles, e.g. [0.5, 0.25]: decimated copies as hidden `<name>.LOD1`, `.LOD2` nodes (MSFT_lod on the base node, extras.polly.lod).
    #[serde(default)]
    lods: Vec<f32>,
}

fn export(ctx: &mut Ctx, a: Export) -> Result<Out> {
    let objects = a
        .objects
        .as_ref()
        .map(|names| names.iter().map(|n| ctx.doc.scene.resolve(Some(n))).collect::<polly_core::Result<Vec<_>>>())
        .transpose()?;
    let bytes = export_glb(&ctx.doc.scene, &ExportOptions { objects, subdivision: a.subdivision, lods: a.lods.clone() })?;
    let mut path = PathBuf::from(&a.path);
    if path.extension().is_none() {
        path.set_extension("glb");
    }
    std::fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(ok(format!("exported {} ({} bytes)", std::path::absolute(&path)?.display(), bytes.len())))
}
