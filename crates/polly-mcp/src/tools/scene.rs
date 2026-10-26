use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Result, bail};
use polly_core::glam::{Mat4, Vec3};
use polly_core::mesh::Mesh;
use polly_core::primitives::Primitive;
use polly_core::scene::{Material, Scene, quat_from_euler_degrees};
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
            "add_mesh",
            "Add a new object from raw geometry: a vertex list and faces as vertex-index loops (counter-clockwise seen from outside). Useful for custom shapes.",
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
        def("delete_object", "Delete an object. Its children are re-parented to its parent.", false, delete_object),
        def("rename_object", "Rename an object.", false, rename_object),
        def("set_active", "Make an object the active one (the default target of tools that take `object`).", false, set_active),
        def(
            "set_transform",
            "Set an object's position, rotation (Euler XYZ degrees) and/or scale. With relative=true values are added (position/rotation) or multiplied (scale).",
            false,
            set_transform,
        ),
        def(
            "apply_transform",
            "Bake an object's transform into its vertices and reset the transform to identity.",
            false,
            apply_transform,
        ),
        def("set_parent", "Parent an object to another (or unparent with parent = null).", false, set_parent),
        def("set_visible", "Show or hide an object (hidden objects are not rendered or exported).", false, set_visible),
        def(
            "join_objects",
            "Merge several objects into the first one (geometry transformed into its space). The others are deleted.",
            false,
            join_objects,
        ),
        def(
            "separate",
            "Move the given faces out of an object into a new object (same transform).",
            false,
            separate,
        ),
        def(
            "create_material",
            "Create a material. Faces use material 0 (\"default\") until assigned.",
            false,
            create_material,
        ),
        def("update_material", "Change properties of an existing material.", false, update_material),
        def("assign_material", "Assign a material to faces of an object.", false, assign_material),
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
    if !o.selection.is_empty() {
        j["selection"] = json!({"verts": o.selection.verts.len(), "edges": o.selection.edges.len(), "faces": o.selection.faces.len()});
    }
    j
}

fn describe_scene(ctx: &mut Ctx, _: Empty) -> Result<Out> {
    let s = &ctx.doc.scene;
    let objects: Vec<_> = s.objects.keys().map(|id| object_summary(s, *id)).collect();
    let materials: Vec<_> = s
        .materials
        .iter()
        .map(|(id, m)| {
            json!({
                "index": id, "name": m.name, "color": polly_hex(m.base_color),
                "texture": m.texture.and_then(|t| s.textures.get(&t)).map(|t| t.name.clone()),
                "unlit": m.unlit, "alpha_cutoff": m.alpha_cutoff, "double_sided": m.double_sided,
            })
        })
        .collect();
    let textures: Vec<_> = s
        .textures
        .iter()
        .map(|(id, t)| json!({"id": format!("t{id}"), "name": t.name, "size": [t.width, t.height], "filter": t.filter, "palette": t.palette}))
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
    /// Euler XYZ degrees.
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
    vertices: Vec<[f32; 3]>,
    /// Each face is a loop of indices into `vertices`, CCW seen from outside.
    faces: Vec<Vec<usize>>,
    position: Option<[f32; 3]>,
    parent: Option<String>,
}

fn add_mesh(ctx: &mut Ctx, a: AddMesh) -> Result<Out> {
    let mut m = Mesh::new();
    let ids: Vec<_> = a.vertices.iter().map(|p| m.add_vert(Vec3::from_array(*p))).collect();
    for (i, f) in a.faces.iter().enumerate() {
        let verts = f
            .iter()
            .map(|k| ids.get(*k).copied().ok_or_else(|| anyhow::anyhow!("face {i}: vertex index {k} out of range")))
            .collect::<Result<Vec<_>>>()?;
        m.add_face(verts).map_err(|e| anyhow::anyhow!("face {i}: {e}"))?;
    }
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
        Ok(id)
    })?;
    Ok(Out::json(object_summary(&ctx.doc.scene, id)))
}

fn delete_object(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let o = edit_scene(ctx, "delete object", |s| s.remove_object(id))?;
    Ok(ok(format!("deleted '{}'", o.name)))
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
    /// Euler XYZ degrees.
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
    edit_scene(ctx, "join objects", |s| {
        let inv = s.world_matrix(target).inverse();
        for &other in &ids[1..] {
            if other == target {
                continue;
            }
            let mut m = (*s.object(other)?.mesh).clone();
            m.transform(inv * s.world_matrix(other));
            Arc::make_mut(&mut s.object_mut(target)?.mesh).append(&m);
            s.remove_object(other)?;
        }
        s.active = Some(target);
        Ok(())
    })?;
    Ok(Out::json(object_summary(&ctx.doc.scene, target)))
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
        let mut rest = (*o.mesh).clone();
        rest.faces.retain(|f, _| !faces.contains(f));
        rest.remove_loose_verts();
        s.object_mut(id)?.mesh = Arc::new(rest);
        s.object_mut(id)?.selection = Default::default();
        let nid = s.add_object(a.name.as_deref().unwrap_or(&format!("{}.part", o.name)), part);
        let n = s.object_mut(nid)?;
        n.transform = o.transform;
        n.parent = o.parent;
        Ok(nid)
    })?;
    Ok(Out::json(json!({"new_object": object_summary(&ctx.doc.scene, new_id), "from": object_summary(&ctx.doc.scene, id)})))
}

#[derive(Deserialize, JsonSchema)]
struct CreateMaterial {
    name: String,
    /// Base color (default white, so textures show unmodified).
    color: Option<[u8; 4]>,
    /// Texture name or id to use as base color map.
    texture: Option<String>,
    #[serde(default)]
    unlit: bool,
    /// Discard pixels with alpha below this (cutout transparency, e.g. 0.5 for foliage).
    alpha_cutoff: Option<f32>,
    #[serde(default)]
    double_sided: bool,
}

fn create_material(ctx: &mut Ctx, a: CreateMaterial) -> Result<Out> {
    let color = a.color.map(to_f32).unwrap_or([1.0; 4]);
    let tex = a.texture.as_deref().map(|t| ctx.doc.scene.resolve_texture(t)).transpose()?;
    if ctx.doc.scene.materials.values().any(|m| m.name == a.name) {
        bail!("material '{}' already exists (use update_material)", a.name);
    }
    let id = edit_scene(ctx, "create material", |s| {
        let mut m = Material::new(&a.name, color);
        m.texture = tex;
        m.unlit = a.unlit;
        m.alpha_cutoff = a.alpha_cutoff;
        m.double_sided = a.double_sided;
        Ok(s.add_material(m))
    })?;
    Ok(Out::json(json!({"material": a.name, "index": id})))
}

#[derive(Deserialize, JsonSchema)]
struct UpdateMaterial {
    /// Material name or index.
    material: String,
    name: Option<String>,
    color: Option<[u8; 4]>,
    /// Texture name/id; empty string removes the texture.
    texture: Option<String>,
    unlit: Option<bool>,
    /// Negative value removes cutout.
    alpha_cutoff: Option<f32>,
    double_sided: Option<bool>,
}

fn update_material(ctx: &mut Ctx, a: UpdateMaterial) -> Result<Out> {
    let id = ctx.doc.scene.resolve_material(&a.material)?;
    let color = a.color.map(to_f32);
    let tex = match a.texture.as_deref() {
        None => None,
        Some("") => Some(None),
        Some(t) => Some(Some(ctx.doc.scene.resolve_texture(t)?)),
    };
    edit_scene(ctx, "update material", |s| {
        let m = s.materials.get_mut(&id).unwrap();
        if let Some(n) = &a.name {
            m.name = n.clone();
        }
        if let Some(c) = color {
            m.base_color = c;
        }
        if let Some(t) = tex {
            m.texture = t;
        }
        if let Some(u) = a.unlit {
            m.unlit = u;
        }
        if let Some(c) = a.alpha_cutoff {
            m.alpha_cutoff = (c >= 0.0).then_some(c);
        }
        if let Some(d) = a.double_sided {
            m.double_sided = d;
        }
        Ok(())
    })?;
    Ok(ok("material updated"))
}

#[derive(Deserialize, JsonSchema)]
struct AssignMaterial {
    object: Option<String>,
    /// Faces (default: current selection; "all" for every face).
    elements: Option<Vec<String>>,
    /// Material name or index.
    material: String,
}

fn assign_material(ctx: &mut Ctx, a: AssignMaterial) -> Result<Out> {
    let mid = ctx.doc.scene.resolve_material(&a.material)?;
    edit_mesh(ctx, &a.object, Some(&a.elements), "assign material", |m, sel| {
        for f in sel.implied_faces(m) {
            m.faces.get_mut(&f).unwrap().material = mid;
        }
        Ok(Default::default())
    })
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
    /// Euler XYZ degrees per step, about `pivot`.
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
