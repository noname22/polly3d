use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Result, bail};
use polly_core::glam::{Mat4, Vec3};
use polly_core::mesh::Mesh;
use polly_core::primitives::Primitive;
use polly_core::scene::{Scene, quat_from_euler_degrees};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def("clear_scene", "Remove everything from this scene (undoable). To get an additional scene, use create_scene instead.", false, clear_scene),
        def(
            "describe_scene",
            "Overview of the scene: objects (with bounds, counts, transforms, parent), materials, textures, palettes, active object.",
            true,
            describe_scene,
        ),
        def(
            "add_primitive",
            "Add a new object from a primitive shape (cube, plane, cylinder/cone, uv_sphere, ico_sphere, torus, lathe, prism). Primitives are centered on the origin, Y up; the new object becomes active. Low-poly defaults (e.g. 8 segments).",
            false,
            add_primitive,
        ),
        def(
            "loft",
            "Create a tube through cross-sections: each section is a closed loop of points (any number per section; different counts are joined with evenly spread triangles), in order along the tube, e.g. a limb or torso from rings measured off a reference. caps closes both ends. Faces point outward.",
            false,
            loft,
        ),
        def(
            "replace_geometry",
            "Swap an object's mesh for new geometry (a .obj/.json file or inline data) while keeping the object: name, transform, parent, materials, gear slots, rig binding and attached children. If the new mesh has the same vertices and face loops in the same order (e.g. edited export_mesh output), only positions (and given uvs/parts) change and all ids, weights and morphs are kept; otherwise skin weights are transferred from the old mesh. Use it to rebuild a mesh from a script instead of delete + add.",
            false,
            replace_geometry,
        ),
        def(
            "add_mesh",
            "Add a new object from raw geometry: a vertex list and faces as vertex-index loops (counter-clockwise seen from outside), inline or from a file (`path`: .obj, or .json with the same `vertices`/`faces` plus optional per-face `uvs`, `parts`, `smooth`; the format export_mesh writes). Prefer a file for large or script-generated meshes.",
            false,
            add_mesh,
        ),
        def("duplicate_object", "Copy an object (mesh, transform, parent) under a new name.", false, duplicate_object),
        def(
            "array",
            "Repeat an object's geometry `count` times, each copy offset/rotated/scaled cumulatively from the previous (fences, stairs, columns, radial patterns). By default the copies are merged into the object; with separate=true new objects are created.",
            false,
            array,
        ),
        def(
            "mirror_object",
            "Create a mirrored copy of an object across a world plane (default: through the origin), e.g. the right boot from the left one. Names swap left/right markers (Boot.L -> Boot.R, LeftArm -> RightArm); morphs and bones are mirrored too.",
            false,
            mirror_object,
        ),
        def("delete_object", "Delete an object. Its children are re-parented to its parent.", false, delete_object),
        def("rename_object", "Rename an object.", false, rename_object),
        def("set_active", "Make an object the active one (the default target of tools that take `object`).", false, set_active),
        def(
            "set_transform",
            "Set an object's position, rotation (Euler degrees, see `rotation`) and/or scale. With relative=true values are added (position/rotation) or multiplied (scale).",
            false,
            set_transform,
        ),
        def(
            "apply_transform",
            "Bake an object's transform into its vertices and reset the transform to identity.",
            false,
            apply_transform,
        ),
        def(
            "match_transform",
            "Place an object where another one is (world position/rotation/scale, respecting both objects' parents), optionally with an offset in the target's local space.",
            false,
            match_transform,
        ),
        def("set_parent", "Parent an object to another (or unparent with parent = null).", false, set_parent),
        def("set_visible", "Show or hide an object (hidden objects are not rendered or exported).", false, set_visible),
        def(
            "join_objects",
            "Merge several objects into the first one (geometry transformed into its space). The others are deleted. Each object's faces become a named part (its object name), selectable with select {\"by\":\"part\"}.",
            false,
            join_objects,
        ),
        def(
            "separate",
            "Move the given faces out of an object into a new object (same transform).",
            false,
            separate,
        ),
        def("set_background", "Set the render background color (hex).", false, set_background),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn clear_scene(ctx: &mut Ctx, _: Empty) -> Result<Out> {
    edit_scene(ctx, "clear scene", |s| {
        *s = Scene::default();
        Ok(())
    })?;
    Ok(ok(format!("scene '{}' is now empty", ctx.scene)))
}

pub fn object_summary(scene: &Scene, id: u32) -> serde_json::Value {
    let o = &scene.objects[&id];
    let m = scene.world_matrix(id);
    let (lo, hi) = o.mesh.verts.values().map(|v| m.transform_point3(v.pos)).fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), p| (lo.min(p), hi.max(p)),
    );
    let mut j = json!({
        "id": format!("o{id}"),
        "name": o.name,
        "verts": o.mesh.verts.len(),
        "faces": o.mesh.faces.len(),
        "tris": o.mesh.triangle_count(),
        "position": round3(o.transform.translation),
        "rotation_deg": round3(Vec3::from_array(o.transform.euler_degrees())),
        "scale": round3(o.transform.scale),
    });
    if !o.mesh.verts.is_empty() {
        j["world_bounds"] = json!({"min": round3(lo), "max": round3(hi), "size": round3(hi - lo)});
    }
    if let Some(p) = o.parent {
        j["parent"] = json!(scene.objects.get(&p).map(|p| p.name.clone()));
    }
    if !o.visible {
        j["visible"] = json!(false);
    }
    if let Some(sk) = &o.skeleton {
        j["bones"] = json!(sk.bones.len());
    }
    if let Some(b) = o.skin {
        j["skinned_to"] = json!(scene.objects.get(&b.armature).map(|a| a.name.clone()));
    }
    if let Some((slot, variants)) = scene.variants_of(id) {
        j["variant"] = json!(format!("{slot}: {}", variants.join(", ")));
    }
    if let (Some(b), Some(p)) = (o.parent_bone, o.parent.and_then(|p| scene.objects.get(&p))) {
        j["attached_to"] = json!(format!("{} / {}", p.name, p.skeleton.as_ref().map_or("?", |sk| sk.name(b))));
    }
    if !o.selection.is_empty() {
        j["selection"] = json!({"verts": o.selection.verts.len(), "edges": o.selection.edges.len(), "faces": o.selection.faces.len()});
    }
    j
}

fn describe_scene(ctx: &mut Ctx, _: Empty) -> Result<Out> {
    let s = &ctx.doc.scene;
    let objects: Vec<_> = s.objects.keys().map(|id| object_summary(s, *id)).collect();
    let materials: Vec<_> = s.materials.keys().map(|id| super::material::material_json(s, *id)).collect();
    let textures: Vec<_> = s
        .textures
        .iter()
        .map(|(id, t)| {
            let mut j = json!({"id": format!("t{id}"), "name": t.name, "size": [t.width, t.height], "filter": t.filter, "palette": t.palette});
            if t.linear {
                j["linear"] = json!(true);
            }
            j
        })
        .collect();
    let mut j = json!({
        "scene": ctx.scene,
        "active": s.active.and_then(|a| s.objects.get(&a)).map(|o| o.name.clone()),
        "objects": objects,
        "materials": materials,
        "textures": textures,
    });
    if let Some((lo, hi)) = s.bounds() {
        j["scene_bounds"] = json!({"min": round3(lo), "max": round3(hi)});
    }
    if !s.slots.is_empty() {
        j["slots"] = json!(s.slots.iter().map(|(k, v)| (k.clone(), json!({"active": v.active, "default": v.default, "variants": v.variants.iter().map(|x| x.name.clone()).collect::<Vec<_>>()}))).collect::<serde_json::Map<_, _>>());
    }
    if let Some(d) = &s.asset_dir {
        j["asset_dir"] = json!(d);
    }
    if !s.lighting.is_default() {
        j["lighting"] = json!(s.lighting);
    }
    if !s.palettes.is_empty() {
        j["palettes"] = json!(s.palettes.iter().map(|(k, v)| (k.clone(), v.len())).collect::<std::collections::BTreeMap<_, _>>());
    }
    Ok(Out::json(j))
}

/// Linear float colour back to an sRGB hex string.
pub fn polly_hex(c: [f32; 4]) -> String {
    let enc = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
    };
    let b = |x: f32| (enc(x) * 255.0).round() as u8;
    if c[3] >= 0.999 {
        format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
    } else {
        format!("#{:02x}{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]), (c[3].clamp(0.0, 1.0) * 255.0).round() as u8)
    }
}

/// sRGB 8-bit colour (as the agent writes it) to linear float (as materials,
/// vertex colours and glTF store it). Alpha stays linear.
pub fn to_f32(c: [u8; 4]) -> [f32; 4] {
    let lin = |x: u8| {
        let s = x as f32 / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    [lin(c[0]), lin(c[1]), lin(c[2]), c[3] as f32 / 255.0]
}

#[derive(Deserialize, JsonSchema)]
struct AddPrimitive {
    /// Shape and its parameters, e.g. {"kind":"cylinder","segments":6} or {"kind":"cube","size":[2,1,1]}.
    shape: Primitive,
    /// Object name (made unique). Defaults to the shape kind.
    name: Option<String>,
    position: Option<[f32; 3]>,
    /// Euler degrees [x, y, z]: R = Rx·Ry·Rz, i.e. Z is applied first, then Y, then X about the parent's axes (equivalently X, Y, Z about the object's own axes).
    rotation: Option<[f32; 3]>,
    scale: Option<[f32; 3]>,
    /// Parent object name.
    parent: Option<String>,
}

fn kind_name(p: &Primitive) -> &'static str {
    match p {
        Primitive::Cube { .. } => "Cube",
        Primitive::Plane { .. } => "Plane",
        Primitive::Cylinder { radius_top, .. } if *radius_top <= 0.0 => "Cone",
        Primitive::Cylinder { .. } => "Cylinder",
        Primitive::UvSphere { .. } | Primitive::IcoSphere { .. } => "Sphere",
        Primitive::Torus { .. } => "Torus",
        Primitive::Lathe { .. } => "Lathe",
        Primitive::Prism { .. } => "Prism",
    }
}

fn add_object(ctx: &mut Ctx, name: &str, mesh: Mesh, pos: Option<[f32; 3]>, rot: Option<[f32; 3]>, scale: Option<[f32; 3]>, parent: &Option<String>) -> Result<Out> {
    let id = edit_scene(ctx, &format!("add {name}"), |s| {
        let parent = parent.as_deref().map(|p| s.resolve(Some(p))).transpose()?;
        let id = s.add_object(name, mesh);
        let o = s.object_mut(id)?;
        if let Some(p) = pos {
            o.transform.translation = Vec3::from_array(p);
        }
        if let Some(r) = rot {
            o.transform.rotation = quat_from_euler_degrees(r);
        }
        if let Some(sc) = scale {
            o.transform.scale = Vec3::from_array(sc);
        }
        s.set_parent(id, parent)?;
        Ok(id)
    })?;
    let mut j = object_summary(&ctx.doc.scene, id);
    j["hint"] = json!("use list_faces / render with face_ids to see element ids");
    super::view::maybe_preview(ctx, id, Out::json(j))
}

fn add_primitive(ctx: &mut Ctx, a: AddPrimitive) -> Result<Out> {
    let mesh = a.shape.build()?;
    let name = a.name.clone().unwrap_or_else(|| kind_name(&a.shape).to_string());
    add_object(ctx, &name, mesh, a.position, a.rotation, a.scale, &a.parent)
}

#[derive(Deserialize, JsonSchema)]
struct AddMesh {
    name: String,
    #[serde(default)]
    vertices: Vec<[f32; 3]>,
    /// Each face is a loop of indices into `vertices`, CCW seen from outside.
    #[serde(default)]
    faces: Vec<Vec<usize>>,
    /// Read the geometry from a .obj or .json file instead (relative to the server's working directory).
    path: Option<String>,
    position: Option<[f32; 3]>,
    parent: Option<String>,
}

fn add_mesh(ctx: &mut Ctx, a: AddMesh) -> Result<Out> {
    use polly_export::meshio;
    let data = match &a.path {
        Some(p) => {
            if !a.vertices.is_empty() || !a.faces.is_empty() {
                anyhow::bail!("give either `path` or inline `vertices`/`faces`, not both");
            }
            meshio::read_mesh_file(std::path::Path::new(p))?
        }
        None => meshio::MeshData { vertices: a.vertices.clone(), faces: a.faces.clone(), ..Default::default() },
    };
    let m = meshio::mesh_from_data(&data)?;
    add_object(ctx, &a.name, m, a.position, None, None, &a.parent)
}

#[derive(Deserialize, JsonSchema)]
struct ObjectArg {
    /// Object name or id like "o2" (default: active object).
    object: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct Duplicate {
    object: Option<String>,
    name: Option<String>,
    /// Added to the copy's position.
    offset: Option<[f32; 3]>,
}

fn duplicate_object(ctx: &mut Ctx, a: Duplicate) -> Result<Out> {
    let src = resolve(&ctx.doc.scene, &a.object)?;
    let id = edit_scene(ctx, "duplicate object", |s| {
        let o = s.object(src)?.clone();
        let id = s.add_object(a.name.as_deref().unwrap_or(&o.name), (*o.mesh).clone());
        let n = s.object_mut(id)?;
        n.transform = o.transform;
        n.transform.translation += Vec3::from_array(a.offset.unwrap_or([0.0; 3]));
        n.parent = o.parent;
        n.skin = o.skin;
        n.parent_bone = o.parent_bone;
        Ok(id)
    })?;
    Ok(Out::json(object_summary(&ctx.doc.scene, id)))
}

fn delete_object(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let slot = ctx.doc.scene.variants_of(id).map(|(s, vs)| (s.to_string(), vs.join(", ")));
    let o = edit_scene(ctx, "delete object", |s| s.remove_object(id))?;
    let mut j = json!({"result": format!("deleted '{}'", o.name)});
    if let Some((slot, vs)) = slot {
        j["note"] = json!(format!("it was removed from slot '{slot}' (variant {vs}); to rebuild a mesh in place, use replace_geometry instead"));
    }
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct Loft {
    name: String,
    /// Closed loops of [x, y, z] points (object space), in order along the tube.
    sections: Vec<Vec<[f32; 3]>>,
    /// Close both ends with a face (default false: open ends, ready to bridge).
    #[serde(default)]
    caps: bool,
    position: Option<[f32; 3]>,
    parent: Option<String>,
}

fn loft(ctx: &mut Ctx, a: Loft) -> Result<Out> {
    let sections: Vec<Vec<Vec3>> = a.sections.iter().map(|s| s.iter().map(|p| Vec3::from_array(*p)).collect()).collect();
    let m = polly_core::ops::loft::loft(&sections, a.caps)?;
    add_object(ctx, &a.name, m, a.position, None, None, &a.parent)
}

#[derive(Deserialize, JsonSchema)]
struct ReplaceGeometry {
    object: Option<String>,
    /// A .obj or .json file (export_mesh's format)...
    path: Option<String>,
    /// ...or inline vertices and faces (as add_mesh).
    #[serde(default)]
    vertices: Vec<[f32; 3]>,
    #[serde(default)]
    faces: Vec<Vec<usize>>,
}

fn replace_geometry(ctx: &mut Ctx, a: ReplaceGeometry) -> Result<Out> {
    use polly_export::meshio;
    let s = &ctx.doc.scene;
    let id = resolve(s, &a.object)?;
    let data = match &a.path {
        Some(p) if a.vertices.is_empty() && a.faces.is_empty() => meshio::read_mesh_file(std::path::Path::new(p))?,
        Some(_) => bail!("give either `path` or inline `vertices`/`faces`, not both"),
        None => meshio::MeshData { vertices: a.vertices.clone(), faces: a.faces.clone(), ..Default::default() },
    };
    let old = s.object(id)?.mesh.clone();
    // Same layout (vertex count and face loops in id order): update in place, keeping ids.
    let index: std::collections::HashMap<u32, usize> = old.verts.keys().enumerate().map(|(i, v)| (*v, i)).collect();
    let same = data.vertices.len() == old.verts.len()
        && data.faces.len() == old.faces.len()
        && old.faces.values().zip(&data.faces).all(|(f, d)| f.verts.iter().map(|v| index[v]).eq(d.iter().copied()));
    let mut notes = Vec::new();
    let mesh = if same {
        let mut m = (*old).clone();
        for (v, p) in m.verts.values_mut().zip(&data.vertices) {
            v.pos = Vec3::from_array(*p);
        }
        for (i, f) in m.faces.values_mut().enumerate() {
            if let Some(uv) = data.uvs.as_ref().map(|u| &u[i]).filter(|uv| uv.len() == f.corners.len()) {
                for (c, t) in f.corners.iter_mut().zip(uv) {
                    c.uv = polly_core::glam::Vec2::from_array(*t);
                }
            }
            if let Some(p) = data.parts.as_ref().map(|p| p[i].clone()) {
                f.part = p;
            }
            if let Some(sm) = data.smooth.as_ref().map(|sm| sm[i]) {
                f.smooth = sm;
            }
        }
        m
    } else {
        let mut m = meshio::mesh_from_data(&data)?;
        // The object's main material for every face.
        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        for f in old.faces.values() {
            *counts.entry(f.material).or_default() += 1;
        }
        if let Some((mat, _)) = counts.into_iter().max_by_key(|(_, n)| *n) {
            for f in m.faces.values_mut() {
                f.material = mat;
            }
        }
        if !old.weights.is_empty() {
            polly_core::anim::transfer_weights(&mut m, polly_core::glam::Mat4::IDENTITY, &old, polly_core::glam::Mat4::IDENTITY)?;
            notes.push("skin weights transferred from the old mesh (check with weight_report)".to_string());
        }
        if !old.morphs.is_empty() {
            notes.push(format!("{} morph(s) dropped: they need the old vertices", old.morphs.len()));
        }
        let stale: Vec<String> = s
            .slots
            .values()
            .flat_map(|sl| &sl.variants)
            .flat_map(|v| &v.hides)
            .filter(|h| h.object == id && h.faces.is_some())
            .map(|h| h.part.clone())
            .collect();
        if !stale.is_empty() {
            notes.push(format!("hide rules {} refer to the old faces: re-run auto_hide", stale.join(", ")));
        }
        notes.push("new vertex and face ids; seams, sharp edges and per-face materials start fresh".into());
        m
    };
    edit_scene(ctx, "replace geometry", |s| {
        let o = s.object_mut(id)?;
        o.mesh = Arc::new(mesh);
        o.selection = Default::default();
        Ok(())
    })?;
    let mut j = object_summary(&ctx.doc.scene, id);
    j["kept_ids"] = json!(same);
    if !notes.is_empty() {
        j["notes"] = json!(notes);
    }
    super::view::maybe_preview(ctx, id, Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct Rename {
    object: Option<String>,
    name: String,
}

fn rename_object(ctx: &mut Ctx, a: Rename) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let n = edit_scene(ctx, "rename", |s| s.rename_object(id, &a.name))?;
    Ok(ok(format!("renamed to '{n}'")))
}

fn set_active(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    ctx.doc.scene.active = Some(id);
    ctx.doc.touch();
    Ok(Out::json(object_summary(&ctx.doc.scene, id)))
}

#[derive(Deserialize, JsonSchema)]
struct SetTransform {
    object: Option<String>,
    position: Option<[f32; 3]>,
    /// Euler degrees [x, y, z]: R = Rx·Ry·Rz, i.e. Z is applied first, then Y, then X about the parent's axes (equivalently X, Y, Z about the object's own axes).
    rotation: Option<[f32; 3]>,
    /// Number for uniform scale or [x,y,z].
    scale: Option<Scalar3>,
    #[serde(default)]
    relative: bool,
}

fn set_transform(ctx: &mut Ctx, a: SetTransform) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    edit_scene(ctx, "set transform", |s| {
        let t = &mut s.object_mut(id)?.transform;
        if let Some(p) = a.position {
            let p = Vec3::from_array(p);
            t.translation = if a.relative { t.translation + p } else { p };
        }
        if let Some(r) = a.rotation {
            let q = quat_from_euler_degrees(r);
            t.rotation = if a.relative { q * t.rotation } else { q };
        }
        if let Some(sc) = &a.scale {
            let v = sc.vec();
            t.scale = if a.relative { t.scale * v } else { v };
        }
        Ok(())
    })?;
    super::view::maybe_preview(ctx, id, Out::json(object_summary(&ctx.doc.scene, id)))
}

fn apply_transform(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    edit_scene(ctx, "apply transform", |s| {
        let o = s.object_mut(id)?;
        let m = o.transform.matrix();
        Arc::make_mut(&mut o.mesh).transform(m);
        o.transform = Default::default();
        Ok(())
    })?;
    Ok(Out::json(object_summary(&ctx.doc.scene, id)))
}

#[derive(Deserialize, JsonSchema)]
struct SetParent {
    object: Option<String>,
    /// New parent object, or null to unparent.
    parent: Option<String>,
    /// Keep the world-space placement by compensating the local transform (default true).
    #[serde(default = "yes")]
    keep_world: bool,
}

fn yes() -> bool {
    true
}

fn set_parent(ctx: &mut Ctx, a: SetParent) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    edit_scene(ctx, "set parent", |s| {
        let parent = a.parent.as_deref().map(|p| s.resolve(Some(p))).transpose()?;
        let world = s.world_matrix(id);
        s.set_parent(id, parent)?;
        if a.keep_world {
            let pw = parent.map(|p| s.world_matrix(p)).unwrap_or(Mat4::IDENTITY);
            let local = pw.inverse() * world;
            let (sc, r, t) = local.to_scale_rotation_translation();
            let tr = &mut s.object_mut(id)?.transform;
            tr.translation = t;
            tr.rotation = r;
            tr.scale = sc;
        }
        Ok(())
    })?;
    Ok(Out::json(object_summary(&ctx.doc.scene, id)))
}

#[derive(Deserialize, JsonSchema)]
struct SetVisible {
    object: Option<String>,
    visible: bool,
}

fn set_visible(ctx: &mut Ctx, a: SetVisible) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    edit_scene(ctx, "set visible", |s| {
        s.object_mut(id)?.visible = a.visible;
        Ok(())
    })?;
    Ok(ok("ok"))
}

#[derive(Deserialize, JsonSchema)]
struct Join {
    /// Objects to merge; the first one receives the geometry.
    objects: Vec<String>,
}

fn join_objects(ctx: &mut Ctx, a: Join) -> Result<Out> {
    if a.objects.len() < 2 {
        bail!("need at least two objects");
    }
    let ids = a.objects.iter().map(|n| ctx.doc.scene.resolve(Some(n))).collect::<polly_core::Result<Vec<_>>>()?;
    let target = ids[0];
    let skins: std::collections::BTreeSet<Option<u32>> = ids.iter().map(|i| ctx.doc.scene.objects[i].skin.map(|b| b.armature)).collect();
    if skins.len() > 1 {
        bail!("these objects are bound to different armatures (or some are unbound); bind them to the same armature first");
    }
    edit_scene(ctx, "join objects", |s| {
        // Faces without a part keep their object's name as the part.
        let label = |m: &mut polly_core::mesh::Mesh, name: &str| {
            for f in m.faces.values_mut().filter(|f| f.part.is_none()) {
                f.part = Some(name.to_string());
            }
        };
        let inv = s.world_matrix(target).inverse();
        let tname = s.object(target)?.name.clone();
        label(Arc::make_mut(&mut s.object_mut(target)?.mesh), &tname);
        for &other in &ids[1..] {
            if other == target {
                continue;
            }
            let o = s.object(other)?;
            let mut m = (*o.mesh).clone();
            label(&mut m, &o.name.clone());
            m.transform(inv * s.world_matrix(other));
            Arc::make_mut(&mut s.object_mut(target)?.mesh).append(&m);
            s.remove_object(other)?;
        }
        s.active = Some(target);
        Ok(())
    })?;
    let mut j = object_summary(&ctx.doc.scene, target);
    j["parts"] = super::query::parts_json(&ctx.doc.scene.objects[&target].mesh);
    j["hint"] = json!("each joined object is now a named part: select {\"by\":\"part\",\"name\":...}");
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct Separate {
    object: Option<String>,
    /// Faces to move out (default: current selection).
    elements: Option<Vec<String>>,
    name: Option<String>,
}

fn separate(ctx: &mut Ctx, a: Separate) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let sel = elements(&ctx.doc.scene, id, &a.elements)?;
    let new_id = edit_scene(ctx, "separate", |s| {
        let o = s.object(id)?.clone();
        let faces: BTreeSet<_> = sel.implied_faces(&o.mesh);
        if faces.is_empty() {
            return polly_core::error::invalid("no faces selected");
        }
        let mut part = (*o.mesh).clone();
        part.faces.retain(|f, _| faces.contains(f));
        part.remove_loose_verts();
        polly_core::anim::prune_weights(&mut part, None);
        let mut rest = (*o.mesh).clone();
        rest.faces.retain(|f, _| !faces.contains(f));
        rest.remove_loose_verts();
        polly_core::anim::prune_weights(&mut rest, None);
        s.object_mut(id)?.mesh = Arc::new(rest);
        s.object_mut(id)?.selection = Default::default();
        let nid = s.add_object(a.name.as_deref().unwrap_or(&format!("{}.part", o.name)), part);
        let n = s.object_mut(nid)?;
        n.transform = o.transform;
        n.parent = o.parent;
        n.skin = o.skin;
        Ok(nid)
    })?;
    Ok(Out::json(json!({"new_object": object_summary(&ctx.doc.scene, new_id), "from": object_summary(&ctx.doc.scene, id)})))
}

#[derive(Deserialize, JsonSchema)]
struct Background {
    color: [u8; 4],
}

fn set_background(ctx: &mut Ctx, a: Background) -> Result<Out> {
    let c = to_f32(a.color);
    edit_scene(ctx, "set background", |s| {
        s.background = c;
        Ok(())
    })?;
    Ok(ok("ok"))
}

#[derive(Deserialize, JsonSchema)]
struct ArrayArgs {
    object: Option<String>,
    /// Total number of instances including the original (>= 2).
    count: u32,
    /// Translation per step (object space).
    #[serde(default)]
    offset: [f32; 3],
    /// Rotation per step about `pivot`. Euler degrees [x, y, z]: R = Rx·Ry·Rz, i.e. Z is applied first, then Y, then X about the parent's axes (equivalently X, Y, Z about the object's own axes).
    #[serde(default)]
    rotation: [f32; 3],
    /// Pivot for rotation (object space, default origin).
    #[serde(default)]
    pivot: [f32; 3],
    /// Uniform scale factor per step (default 1).
    scale: Option<f32>,
    /// Create separate objects instead of merging.
    #[serde(default)]
    separate: bool,
}

fn array(ctx: &mut Ctx, a: ArrayArgs) -> Result<Out> {
    if !(2..=256).contains(&a.count) {
        bail!("count must be 2..256");
    }
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let pivot = Vec3::from_array(a.pivot);
    let step = Mat4::from_translation(Vec3::from_array(a.offset))
        * Mat4::from_translation(pivot)
        * Mat4::from_quat(quat_from_euler_degrees(a.rotation))
        * Mat4::from_scale(Vec3::splat(a.scale.unwrap_or(1.0)))
        * Mat4::from_translation(-pivot);
    let created = edit_scene(ctx, "array", |s| {
        let o = s.object(id)?.clone();
        let mut m = Mat4::IDENTITY;
        let mut created = vec![];
        let mut merged = (*o.mesh).clone();
        for _ in 1..a.count {
            m = step * m;
            let mut copy = (*o.mesh).clone();
            copy.transform(m);
            if a.separate {
                let nid = s.add_object(&o.name, copy);
                let n = s.object_mut(nid)?;
                n.transform = o.transform;
                n.parent = o.parent;
                n.skin = o.skin;
                created.push(nid);
            } else {
                merged.append(&copy);
            }
        }
        if !a.separate {
            s.object_mut(id)?.mesh = Arc::new(merged);
        }
        s.active = Some(id);
        Ok(created)
    })?;
    let s = &ctx.doc.scene;
    let mut j = object_summary(s, id);
    if !created.is_empty() {
        j["created"] = json!(created.iter().map(|c| s.objects[c].name.clone()).collect::<Vec<_>>());
    }
    super::view::maybe_preview(ctx, id, Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct MatchTransform {
    /// Object to move (default: active).
    object: Option<String>,
    /// Object whose placement to copy.
    target: String,
    #[serde(default = "yes")]
    position: bool,
    #[serde(default = "yes")]
    rotation: bool,
    #[serde(default)]
    scale: bool,
    /// Offset in the target's local space (e.g. [0, 0.2, 0] = a bit along the target's +Y).
    offset: Option<[f32; 3]>,
}

fn match_transform(ctx: &mut Ctx, a: MatchTransform) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let target = ctx.doc.scene.resolve(Some(&a.target))?;
    if id == target {
        bail!("object and target are the same");
    }
    edit_scene(ctx, "match transform", |s| {
        let tw = s.world_matrix(target) * Mat4::from_translation(Vec3::from_array(a.offset.unwrap_or([0.0; 3])));
        let (ts, tr, tt) = tw.to_scale_rotation_translation();
        let (os, or, ot) = s.world_matrix(id).to_scale_rotation_translation();
        let world = Mat4::from_scale_rotation_translation(
            if a.scale { ts } else { os },
            if a.rotation { tr } else { or },
            if a.position { tt } else { ot },
        );
        let parent = s.object(id)?.parent.map(|p| s.world_matrix(p)).unwrap_or(Mat4::IDENTITY);
        let (sc, r, t) = (parent.inverse() * world).to_scale_rotation_translation();
        let tr = &mut s.object_mut(id)?.transform;
        tr.translation = t;
        tr.rotation = r;
        tr.scale = sc;
        Ok(())
    })?;
    super::view::maybe_preview(ctx, id, Out::json(object_summary(&ctx.doc.scene, id)))
}

#[derive(Deserialize, JsonSchema)]
struct MirrorObject {
    object: Option<String>,
    /// World axis normal to the mirror plane: "x" (default: left/right), "y" or "z".
    #[serde(default = "x_axis")]
    axis: String,
    /// A point on the mirror plane in world space (default: the origin).
    about: Option<[f32; 3]>,
    /// Name of the copy (default: left/right swapped, or "<name>.mirror").
    name: Option<String>,
}

fn x_axis() -> String {
    "x".into()
}

/// Swap left/right markers in a name: Boot.L -> Boot.R, LeftArm -> RightArm.
pub fn mirror_name(name: &str) -> String {
    for (a, b) in [("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT")] {
        if name.contains(a) {
            return name.replace(a, b);
        }
        if name.contains(b) {
            return name.replace(b, a);
        }
    }
    for sep in ['.', '_', '-', ' '] {
        for (a, b) in [('L', 'R'), ('l', 'r')] {
            for (x, y) in [(a, b), (b, a)] {
                if let Some(stem) = name.strip_suffix(&format!("{sep}{x}")) {
                    return format!("{stem}{sep}{y}");
                }
                if let Some(rest) = name.strip_prefix(&format!("{x}{sep}")) {
                    return format!("{y}{sep}{rest}");
                }
            }
        }
    }
    format!("{name}.mirror")
}

fn mirror_object(ctx: &mut Ctx, a: MirrorObject) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let axis = match a.axis.to_lowercase().as_str() {
        "x" => Vec3::X,
        "y" => Vec3::Y,
        "z" => Vec3::Z,
        other => bail!("axis must be x, y or z, not '{other}'"),
    };
    let new_id = edit_scene(ctx, "mirror object", |s| {
        let o = s.object(id)?.clone();
        let w = s.world_matrix(id);
        // World reflection across the plane through `about` with normal `axis`.
        let p = Vec3::from_array(a.about.unwrap_or([0.0; 3]));
        let refl = Mat4::from_translation(p)
            * Mat4::from_mat3(polly_core::glam::Mat3::IDENTITY - 2.0 * outer(axis))
            * Mat4::from_translation(-p);
        // Flip the local axis that is closest to the mirror axis in the mesh;
        // the remaining reflection x rotation x flip is a proper rotation.
        let k = (0..3).max_by(|&i, &j| w.col(i).truncate().normalize().dot(axis).abs().total_cmp(&w.col(j).truncate().normalize().dot(axis).abs())).unwrap();
        let mut flip = Vec3::ONE;
        flip[k] = -1.0;
        let local_flip = Mat4::from_scale(flip);
        let mut mesh = (*o.mesh).clone();
        mesh.transform(local_flip);
        for m in &mut mesh.morphs {
            m.offsets.values_mut().for_each(|d| *d *= flip);
        }
        let world = refl * w * local_flip;
        // Gear attached to a bone moves to the mirrored bone (forearm.L -> forearm.R).
        let bone = o.parent_bone.map(|b| {
            let sk = o.parent.and_then(|p| s.objects.get(&p)).and_then(|p| p.skeleton.as_ref());
            sk.and_then(|sk| {
                let name = mirror_name(sk.name(b));
                sk.bones.iter().find(|(_, x)| x.name == name).map(|(id, _)| *id)
            })
            .unwrap_or(b)
        });
        let frame = bone
            .and_then(|b| o.parent.and_then(|p| s.objects.get(&p)).and_then(|p| p.skeleton.as_ref()).and_then(|sk| sk.bones.get(&b)))
            .map_or(Mat4::IDENTITY, |b| b.rest_matrix());
        let parent = o.parent.map(|pid| s.world_matrix(pid)).unwrap_or(Mat4::IDENTITY) * frame;
        let (sc, r, t) = (parent.inverse() * world).to_scale_rotation_translation();
        let name = a.name.clone().unwrap_or_else(|| mirror_name(&o.name));
        let nid = s.add_object(&name, mesh);
        let n = s.object_mut(nid)?;
        n.transform = polly_core::scene::Transform { translation: t, rotation: r, scale: sc };
        n.parent = o.parent;
        n.morph_weights = o.morph_weights.clone();
        n.skin = o.skin;
        n.parent_bone = bone;
        if let Some(sk) = o.skin.and_then(|b| s.objects.get(&b.armature)).and_then(|a| a.skeleton.clone()) {
            // Gear bound to a symmetric rig: weights move to the mirrored bones (hand.L -> hand.R).
            let swap: std::collections::BTreeMap<u32, u32> = sk
                .bones
                .iter()
                .filter_map(|(id, b)| {
                    let other = mirror_name(&b.name);
                    sk.bones.iter().find(|(_, x)| x.name == other).map(|(oid, _)| (*id, *oid))
                })
                .collect();
            let m = Arc::make_mut(&mut s.object_mut(nid)?.mesh);
            for list in m.weights.values_mut() {
                for (b, _) in list.iter_mut() {
                    *b = swap.get(b).copied().unwrap_or(*b);
                }
            }
        }
        let n = s.object_mut(nid)?;
        if let Some(mut sk) = o.skeleton.clone() {
            for b in sk.bones.values_mut() {
                b.head *= flip;
                b.tail *= flip;
                b.roll = -b.roll;
                b.name = mirror_name(&b.name).trim_end_matches(".mirror").to_string();
            }
            n.skeleton = Some(sk);
        }
        Ok(nid)
    })?;
    super::view::maybe_preview(ctx, new_id, Out::json(object_summary(&ctx.doc.scene, new_id)))
}

fn outer(a: Vec3) -> polly_core::glam::Mat3 {
    polly_core::glam::Mat3::from_cols(a * a.x, a * a.y, a * a.z)
}

#[cfg(test)]
mod tests {
    use super::mirror_name;

    #[test]
    fn mirror_names() {
        for (a, b) in [("Boot.L", "Boot.R"), ("hand_r", "hand_l"), ("LeftArm", "RightArm"), ("R-ear", "L-ear"), ("Belt", "Belt.mirror")] {
            assert_eq!(mirror_name(a), b);
        }
    }
}
