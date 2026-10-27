//! Rigging, skinning, posing, morph targets and animation clips.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use polly_core::anim::{
    self, AutoWeights, Bone, BoneId, ClipId, ClipPlay, Interpolation, Skeleton, Target, WeightMode,
};
use polly_core::glam::{Vec3, Vec4};
use polly_core::mesh::VertId;
use polly_core::scene::{ObjectId, Scene, quat_from_euler_degrees};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        // rig
        def("add_bone", "Add a bone to an object's skeleton (created on first use). Bones are given by head and tail points in the object's local space; +Y of the bone runs head->tail.", false, add_bone),
        def("add_bone_chain", "Add a chain of connected bones through a list of points (n points -> n-1 bones), e.g. a spine, tail, arm or finger.", false, add_bone_chain),
        def(
            "bind_to_armature",
            "Skin an object to another object's skeleton (an armature), so one skeleton deforms the body and its gear and one set of clips animates them all. Weights then refer to the armature's bones (auto_weights or assign_weights). Parents the object to the armature by default.",
            false,
            bind_to_armature,
        ),
        def("unbind", "Stop skinning an object to its armature (its weights are kept).", false, unbind),
        def(
            "attach_to_bone",
            "Attach rigid gear (helmet, sword, shield, bracer) to a bone: it keeps its current placement and follows the bone when posed or animated. Use set_parent to detach.",
            false,
            attach_to_bone,
        ),
        def(
            "transfer_weights",
            "Give fitted, deforming gear (skirt, sleeves, long hair, cape) skin weights copied from the nearest surface of a skinned mesh (usually the body), then bind it to that mesh's armature.",
            false,
            transfer_weights,
        ),
        def("edit_bone", "Rename, move (head/tail), re-parent or roll a bone (rest pose).", false, edit_bone),
        def("delete_bone", "Delete a bone; its children are re-parented and its weights removed.", false, delete_bone),
        def(
            "list_bones",
            "List an object's bones with rest head/tail, parent, current pose, how many vertices each one influences, and its pose axes in world terms: which world direction its x/y/z point, and which way the tail moves for a positive x or z turn (y is along the bone: twist). Use it to pick the axis and sign for pose_bone/set_key; render overlay bone_axes draws them.",
            true,
            list_bones,
        ),
        // skin
        def("assign_weights", "Set skin weights of vertices (elements) for one bone. mode: replace (default), add, subtract.", false, assign_weights),
        def("auto_weights", "Compute skin weights from distance to bones: rigid (each vertex follows its nearest bone; chunky retro look) or smooth (blend near joints).", false, auto_weights),
        def("normalize_weights", "Keep the 4 strongest bones per vertex and make each vertex's weights sum to 1.", false, normalize_weights),
        def("weight_report", "Skin weight summary: vertices per bone, unweighted vertices, vertices with more than 4 influences or unnormalized sums.", true, weight_report),
        // pose
        def(
            "solve_ik",
            "Two-bone IK on the static pose: rotate `bone` (e.g. a shin or forearm) and its parent (thigh, upper arm) so that `bone`'s tail reaches `target` (world space). The joint bends towards `pole` (a world point in front of the knee / behind the elbow; default: the current bend). Key the result with key_pose.",
            false,
            solve_ik,
        ),
        def(
            "bake_ik",
            "Two-bone IK baked into a clip: for every frame in `frames`, pose `bone` and its parent so its tail reaches `target` (world space) and key their rotations. With `lock: true` the target is where the tail is at the first frame of the range: foot locking during a stance phase, so the foot doesn't slide while the hips move. The rest of the animation (hips, other bones, the object's motion) is taken from the clip.",
            false,
            bake_ik,
        ),
        def(
            "mirror_pose",
            "Mirror the static pose across the rig's X = 0 plane, swapping left/right bones by name (hand.L <-> hand.R; centre bones mirror onto themselves). mode: flip (default; the whole pose becomes its mirror image), left_to_right or right_to_left (copy one side's pose onto the other). Signs are handled for you.",
            false,
            mirror_pose,
        ),
        def("pose_bone", "Pose a bone relative to its rest pose (rotation in Euler degrees about the bone's rest axes, +Y along the bone; translation; scale). This is the static pose shown in renders; key it into a clip with key_pose.", false, pose_bone),
        def("clear_pose", "Reset bones (default all) of an object to the rest pose.", false, clear_pose),
        // morphs
        def("begin_morph", "Start editing a morph target (shape key): move/scale/rotate vertices with the normal tools, then end_morph stores the change as the morph and restores the base shape. Editing an existing morph starts from its shape.", false, begin_morph),
        def("end_morph", "Finish the morph edit started with begin_morph.", false, end_morph),
        def("cancel_morph", "Abandon the morph edit and restore the base shape.", false, cancel_morph),
        def("morph_from_object", "Create a morph from another object with the same vertices (e.g. a duplicate you reshaped).", false, morph_from_object),
        def("set_morph_weight", "Set a morph's static weight (0 = base shape, 1 = full morph; other values extrapolate).", false, set_morph_weight),
        def("delete_morph", "Delete a morph target.", false, delete_morph),
        // clips
        def("create_clip", "Create an animation clip. Clip tools use frames at the clip's fps (default 24). Its channels interpolate smoothly by default (see set_interpolation).", false, create_clip),
        def("update_clip", "Change a clip's name, length, fps, looping, loop region (in frames) or the interpolation of channels created from now on.", false, update_clip),
        def("delete_clip", "Delete an animation clip.", false, delete_clip),
        def("list_clips", "List clips with length, fps, loop region and channel counts.", true, list_clips),
        def("describe_clip", "Show a clip's channels (what is animated) and their key frames.", true, describe_clip),
        def(
            "key_pose",
            "Record the current static pose into a clip at a frame: bone rotations/translations/scales (default all bones of the object that differ from rest, or listed bones), and optionally morph weights and the object transform. Typical loop: pose_bone ... key_pose, repeat at other frames.",
            false,
            key_pose,
        ),
        def("set_key", "Set one key directly: a bone property (rotation as Euler degrees, translation, scale), a morph weight, or an object transform property.", false, set_key),
        def(
            "copy_keys",
            "Copy a clip's bone keys in a frame range to start at another frame, optionally mirrored (left/right swapped with correct signs). A walk cycle: key frames 0..12 for one step, then copy_keys frames [0, 12] to 12 with mirror: true for the other step (frame 24 = frame 0 makes it loop seamlessly). Without mirror it is a phase-shifted copy.",
            false,
            copy_keys,
        ),
        def("delete_keys", "Delete keys of a clip, optionally only for one object/bone/morph and/or a frame range.", false, delete_keys),
        def(
            "set_interpolation",
            "Set interpolation of matching channels in a clip: smooth (curve through the keys; natural motion), ease (eases in and out of every key; deliberate moves), linear, or step (hold). In a looping clip a channel whose last key repeats its first at the loop region's ends wraps seamlessly.",
            false,
            set_interpolation,
        ),
        def(
            "render_animation",
            "Render a filmstrip: several frames of one or more blended clips in one captioned grid image, to check motion.",
            true,
            render_animation,
        ),
    ]
}

// ---- helpers ----------------------------------------------------------------

/// The object holding the skeleton for `obj`: its armature if `obj` is bound
/// to one, else `obj` itself (which may not have a skeleton yet).
pub fn rig(scene: &Scene, obj: ObjectId) -> ObjectId {
    scene.objects.get(&obj).and_then(|o| o.skin).map_or(obj, |b| b.armature)
}

fn bone_of(scene: &Scene, obj: ObjectId, name: &str) -> Result<BoneId> {
    Ok(skeleton(scene, obj)?.find(name)?)
}

/// The skeleton that deforms `obj` (its own or its armature's).
fn skeleton(scene: &Scene, obj: ObjectId) -> Result<&Skeleton> {
    let r = rig(scene, obj);
    let o = scene.object(r)?;
    o.skeleton
        .as_ref()
        .with_context(|| format!("'{}' has no skeleton; add bones with add_bone, or bind_to_armature", scene.object(obj).map_or("?", |o| o.name.as_str())))
}

fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::from_array(a)
}

fn bone_json(sk: &Skeleton, id: BoneId, b: &Bone) -> Value {
    let mut j = json!({
        "name": b.name,
        "head": round3(b.head),
        "tail": round3(b.tail),
        "length": (b.length() * 1000.0).round() / 1000.0,
    });
    if let Some(p) = b.parent {
        j["parent"] = json!(sk.name(p));
    }
    if b.roll != 0.0 {
        j["roll"] = json!(b.roll);
    }
    let _ = id;
    j
}

/// Frame -> seconds for a clip.
fn secs(scene: &Scene, clip: ClipId, frame: f32) -> f32 {
    frame / scene.clips[&clip].fps
}

/// Clip plays given by the agent, in frames.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PlayArg {
    /// Clip name.
    pub clip: String,
    /// Frame (at the clip's fps); fractional frames interpolate.
    #[serde(default)]
    pub frame: f32,
    /// Blend weight (default 1).
    pub weight: Option<f32>,
}

pub fn plays(scene: &Scene, args: &[PlayArg]) -> Result<Vec<ClipPlay>> {
    args.iter()
        .map(|p| {
            let id = scene.resolve_clip(&p.clip)?;
            Ok(ClipPlay { clip: id, time: secs(scene, id, p.frame), weight: p.weight.unwrap_or(1.0) })
        })
        .collect()
}

/// Heat colour for a weight: blue (0) -> cyan -> green -> yellow -> red (1).
fn heat(w: f32) -> Vec4 {
    let stops = [[0.1, 0.15, 0.8], [0.0, 0.8, 0.9], [0.1, 0.85, 0.2], [0.95, 0.85, 0.1], [0.9, 0.1, 0.05]];
    let x = w.clamp(0.0, 1.0) * 4.0;
    let i = (x.floor() as usize).min(3);
    let t = x - i as f32;
    let (a, b) = (stops[i], stops[i + 1]);
    // vertex colours are linear; these stops are picked in sRGB-ish space, so square them
    let c = |k: usize| (a[k] + (b[k] - a[k]) * t).powi(2);
    Vec4::new(c(0), c(1), c(2), 1.0)
}

/// Display copy of `scene` showing `bone`'s skin weights on `object` as vertex colours.
pub fn weight_view(scene: &Scene, object: ObjectId, bone: &str) -> Result<Scene> {
    let b = bone_of(scene, object, bone)?;
    let mut s = scene.clone();
    let mut m = polly_core::scene::Material::new("weights", [1.0; 4]);
    m.unlit = true;
    let mid = s.add_material(m);
    let o = s.object_mut(object)?;
    // The selection tint would distort the heatmap colours.
    o.selection = Default::default();
    let mesh = Arc::make_mut(&mut o.mesh);
    let w: BTreeMap<VertId, f32> = mesh
        .weights
        .iter()
        .filter_map(|(v, l)| l.iter().find(|(x, _)| *x == b).map(|(_, w)| (*v, *w)))
        .collect();
    for f in mesh.faces.values_mut() {
        f.material = mid;
        for (i, v) in f.verts.iter().enumerate() {
            f.corners[i].color = match (w.get(v), mesh.weights.contains_key(v)) {
                (Some(x), _) => heat(*x),
                (None, true) => heat(0.0),
                (None, false) => Vec4::new(0.05, 0.05, 0.05, 1.0), // unweighted
            };
        }
    }
    Ok(s)
}

// ---- rig ------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct AddBone {
    object: Option<String>,
    name: String,
    head: [f32; 3],
    tail: [f32; 3],
    /// Parent bone name.
    parent: Option<String>,
    /// Twist around the bone axis in degrees.
    #[serde(default)]
    roll: f32,
}

fn add_bone(ctx: &mut Ctx, a: AddBone) -> Result<Out> {
    let obj = rig(&ctx.doc.scene, resolve(&ctx.doc.scene, &a.object)?);
    edit_scene(ctx, "add bone", |s| {
        let parent = match &a.parent {
            Some(p) => Some(s.object(obj)?.skeleton.as_ref().map(|sk| sk.find(p)).transpose()?.ok_or_else(|| {
                polly_core::Error::Invalid(format!("parent bone '{p}' does not exist"))
            })?),
            None => None,
        };
        let o = s.object_mut(obj)?;
        o.skeleton.get_or_insert_with(Skeleton::default).add(Bone {
            name: a.name.clone(),
            parent,
            head: v3(a.head),
            tail: v3(a.tail),
            roll: a.roll,
        })
    })?;
    Ok(ok(format!("added bone '{}'", a.name)))
}

#[derive(Deserialize, JsonSchema)]
struct BoneChain {
    object: Option<String>,
    /// n points make n-1 bones.
    points: Vec<[f32; 3]>,
    /// Bone names (n-1). Default: `<prefix>.1`, `<prefix>.2`, ...
    names: Option<Vec<String>>,
    #[serde(default = "chain_prefix")]
    prefix: String,
    /// Parent of the first bone.
    parent: Option<String>,
}

fn chain_prefix() -> String {
    "bone".into()
}

fn add_bone_chain(ctx: &mut Ctx, a: BoneChain) -> Result<Out> {
    if a.points.len() < 2 {
        bail!("need at least 2 points");
    }
    let n = a.points.len() - 1;
    let names: Vec<String> = match &a.names {
        Some(ns) if ns.len() == n => ns.clone(),
        Some(ns) => bail!("{} points make {n} bones but {} names were given", a.points.len(), ns.len()),
        None => (1..=n).map(|i| format!("{}.{i}", a.prefix)).collect(),
    };
    let obj = rig(&ctx.doc.scene, resolve(&ctx.doc.scene, &a.object)?);
    edit_scene(ctx, "add bone chain", |s| {
        let o = s.object_mut(obj)?;
        let sk = o.skeleton.get_or_insert_with(Skeleton::default);
        let mut parent = a.parent.as_deref().map(|p| sk.find(p)).transpose()?;
        for (i, name) in names.iter().enumerate() {
            let id = sk.add(Bone { name: name.clone(), parent, head: v3(a.points[i]), tail: v3(a.points[i + 1]), roll: 0.0 })?;
            parent = Some(id);
        }
        Ok(())
    })?;
    Ok(Out::json(json!({"bones": names})))
}

#[derive(Deserialize, JsonSchema)]
struct EditBone {
    object: Option<String>,
    bone: String,
    name: Option<String>,
    head: Option<[f32; 3]>,
    tail: Option<[f32; 3]>,
    /// New parent bone name; "" removes the parent.
    parent: Option<String>,
    roll: Option<f32>,
}

fn edit_bone(ctx: &mut Ctx, a: EditBone) -> Result<Out> {
    let obj = rig(&ctx.doc.scene, resolve(&ctx.doc.scene, &a.object)?);
    let id = bone_of(&ctx.doc.scene, obj, &a.bone)?;
    edit_scene(ctx, "edit bone", |s| {
        let sk = s.object_mut(obj)?.skeleton.as_mut().unwrap();
        let parent = match a.parent.as_deref() {
            None => None,
            Some("") => Some(None),
            Some(p) => {
                let pid = sk.find(p)?;
                if sk.creates_cycle(id, pid) {
                    return polly_core::error::invalid("that parent would create a cycle");
                }
                Some(Some(pid))
            }
        };
        if let Some(n) = &a.name {
            if sk.bones.iter().any(|(i, b)| *i != id && &b.name == n) {
                return polly_core::error::invalid(format!("a bone named '{n}' already exists"));
            }
        }
        let b = sk.bones.get_mut(&id).unwrap();
        if let Some(n) = &a.name {
            b.name = n.clone();
        }
        if let Some(h) = a.head {
            b.head = v3(h);
        }
        if let Some(t) = a.tail {
            b.tail = v3(t);
        }
        if let Some(r) = a.roll {
            b.roll = r;
        }
        if let Some(p) = parent {
            b.parent = p;
        }
        if (b.tail - b.head).length() < 1e-6 {
            return polly_core::error::invalid("bone head and tail must differ");
        }
        Ok(())
    })?;
    Ok(ok("bone updated"))
}

#[derive(Deserialize, JsonSchema)]
struct BoneArg {
    object: Option<String>,
    bone: String,
}

fn delete_bone(ctx: &mut Ctx, a: BoneArg) -> Result<Out> {
    let obj = rig(&ctx.doc.scene, resolve(&ctx.doc.scene, &a.object)?);
    let id = bone_of(&ctx.doc.scene, obj, &a.bone)?;
    edit_scene(ctx, "delete bone", |s| {
        // Gear attached to this bone stays where it is, as a plain child of the rig.
        let attached: Vec<ObjectId> = s.objects.iter().filter(|(_, o)| o.parent == Some(obj) && o.parent_bone == Some(id)).map(|(i, _)| *i).collect();
        for g in attached {
            let world = s.world_matrix(g);
            let pw = s.world_matrix(obj);
            let gear = s.object_mut(g)?;
            gear.parent_bone = None;
            let (sc, r, t) = (pw.inverse() * world).to_scale_rotation_translation();
            gear.transform = polly_core::scene::Transform { translation: t, rotation: r, scale: sc };
        }
        let o = s.object_mut(obj)?;
        o.skeleton.as_mut().unwrap().remove(id)?;
        o.pose.remove(&id);
        let sk = o.skeleton.clone();
        // The rig's own mesh and every mesh bound to it lose that bone's weights.
        for (oid, ob) in s.objects.iter_mut() {
            if *oid == obj || ob.skin.is_some_and(|b| b.armature == obj) {
                anim::prune_weights(Arc::make_mut(&mut ob.mesh), sk.as_ref());
            }
        }
        for c in s.clips.values_mut() {
            c.channels.retain(|ch| {
                !matches!(ch.target,
                    Target::BoneTranslation { object, bone } | Target::BoneRotation { object, bone } | Target::BoneScale { object, bone }
                    if object == obj && bone == id)
            });
        }
        Ok(())
    })?;
    Ok(ok(format!("deleted bone '{}'", a.bone)))
}

#[derive(Deserialize, JsonSchema)]
struct ObjectArg {
    object: Option<String>,
}

/// A world direction as the nearest axis ("+Z"), or rounded components when it is between axes.
fn axis_name(d: Vec3) -> Value {
    let names = [(Vec3::X, "+X"), (Vec3::NEG_X, "-X"), (Vec3::Y, "+Y"), (Vec3::NEG_Y, "-Y"), (Vec3::Z, "+Z"), (Vec3::NEG_Z, "-Z")];
    let d = d.normalize_or_zero();
    match names.iter().find(|(a, _)| a.dot(d) > 0.966) {
        Some((_, n)) => json!(n),
        None => json!(round3(d).map(|x| (x * 100.0).round() / 100.0)),
    }
}

fn list_bones(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let s = &ctx.doc.scene;
    let obj = rig(s, resolve(s, &a.object)?);
    let o = s.object(obj)?;
    let Some(sk) = &o.skeleton else { return Ok(ok(format!("'{}' has no skeleton", o.name))) };
    // Weighted vertices per bone, over the rig's own mesh and all bound meshes.
    let skinned: Vec<&polly_core::scene::Object> =
        s.objects.iter().filter(|(id, ob)| **id == obj || ob.skin.is_some_and(|b| b.armature == obj)).map(|(_, ob)| ob).collect();
    let mut counts: BTreeMap<BoneId, usize> = BTreeMap::new();
    for ob in &skinned {
        for l in ob.mesh.weights.values() {
            for (b, _) in l {
                *counts.entry(*b).or_default() += 1;
            }
        }
    }
    let bound: Vec<&str> = s.objects.values().filter(|ob| ob.skin.is_some_and(|b| b.armature == obj)).map(|ob| ob.name.as_str()).collect();
    // The axes pose rotations turn about, in world terms (with parents' static pose).
    let frames = sk.pose_frames(&o.pose);
    let world = s.world_matrix(obj);
    let bones: Vec<Value> = sk
        .ordered()
        .into_iter()
        .map(|id| {
            let b = &sk.bones[&id];
            let mut j = bone_json(sk, id, b);
            let f = world * frames[&id];
            let along = f.transform_vector3(Vec3::Y).normalize_or_zero();
            let axis = |a: Vec3| f.transform_vector3(a).normalize_or_zero();
            // Direction the tail moves for a small positive turn about each axis.
            let moves = |a: Vec3| axis_name(axis(a).cross(along));
            j["axes"] = json!({
                "x": axis_name(axis(Vec3::X)),
                "y": axis_name(along),
                "z": axis_name(axis(Vec3::Z)),
                "positive_x_turn_moves_tail": moves(Vec3::X),
                "positive_z_turn_moves_tail": moves(Vec3::Z),
            });
            j["weighted_verts"] = json!(counts.get(&id).copied().unwrap_or(0));
            if let Some(p) = o.pose.get(&id).filter(|p| !p.is_rest()) {
                let (x, y, z) = p.rotation.to_euler(polly_core::glam::EulerRot::XYZ);
                j["pose"] = json!({
                    "rotation_deg": round3(Vec3::new(x, y, z) * 180.0 / std::f32::consts::PI),
                    "translation": round3(p.translation),
                    "scale": round3(p.scale),
                });
            }
            j
        })
        .collect();
    let mut j = json!({"object": o.name, "bones": bones});
    if !o.mesh.verts.is_empty() {
        j["unweighted_verts"] = json!(o.mesh.verts.len() - o.mesh.weights.len());
    }
    if !bound.is_empty() {
        j["bound_meshes"] = json!(bound);
    }
    Ok(Out::json(j))
}

// ---- skin -----------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct AssignWeights {
    object: Option<String>,
    /// Vertices/faces/edges (default: current selection).
    elements: Option<Vec<String>>,
    bone: String,
    /// Default 1.
    weight: Option<f32>,
    #[serde(default)]
    mode: WeightMode,
}

fn assign_weights(ctx: &mut Ctx, a: AssignWeights) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    let b = bone_of(&ctx.doc.scene, obj, &a.bone)?;
    let mut n = 0;
    let mut out = edit_mesh(ctx, &a.object, Some(&a.elements), "assign weights", |m, s| {
        let verts = s.touched_verts(m);
        n = verts.len();
        anim::assign_weights(m, &verts, b, a.weight.unwrap_or(1.0), a.mode);
        Ok(Default::default())
    })?;
    out.json["weighted_verts"] = json!(n);
    out.json["hint"] = json!("render with show_weights to check; normalize_weights before export");
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
struct AutoWeightArgs {
    object: Option<String>,
    /// Vertices to weight (default all).
    elements: Option<Vec<String>>,
    /// Bones to use (default all).
    bones: Option<Vec<String>>,
    /// {"mode":"rigid"} (default) or {"mode":"smooth","falloff":0.5}.
    method: Option<AutoWeights>,
}

fn auto_weights(ctx: &mut Ctx, a: AutoWeightArgs) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    let sk = skeleton(&ctx.doc.scene, obj)?.clone();
    let bones: BTreeSet<BoneId> = match &a.bones {
        None => sk.bones.keys().copied().collect(),
        Some(ns) => ns.iter().map(|n| sk.find(n)).collect::<polly_core::Result<_>>()?,
    };
    let ids = Some(a.elements.clone().unwrap_or_else(|| vec!["all".into()]));
    let to_skel = ctx.doc.scene.to_skeleton_space(obj);
    edit_mesh(ctx, &a.object, Some(&ids), "auto weights", |m, s| {
        let verts = s.touched_verts(m);
        anim::auto_weights_in(m, &sk, &verts, &bones, a.method.unwrap_or(AutoWeights::Rigid), to_skel)?;
        anim::normalize_weights(m);
        Ok(Default::default())
    })
}

fn normalize_weights(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    edit_mesh(ctx, &a.object, None, "normalize weights", |m, _| {
        anim::normalize_weights(m);
        Ok(Default::default())
    })
}

fn weight_report(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let s = &ctx.doc.scene;
    let obj = resolve(s, &a.object)?;
    let o = s.object(obj)?;
    let sk = skeleton(s, obj)?;
    let mut per_bone: BTreeMap<String, usize> = sk.bones.values().map(|b| (b.name.clone(), 0)).collect();
    let mut over = vec![];
    let mut unnormalized = vec![];
    for (v, l) in &o.mesh.weights {
        for (b, _) in l {
            *per_bone.entry(sk.name(*b).to_string()).or_default() += 1;
        }
        if l.len() > anim::MAX_INFLUENCES {
            over.push(polly_core::ids::vert(*v));
        }
        if (l.iter().map(|(_, w)| w).sum::<f32>() - 1.0).abs() > 1e-3 {
            unnormalized.push(polly_core::ids::vert(*v));
        }
    }
    let unweighted: Vec<String> =
        o.mesh.verts.keys().filter(|v| !o.mesh.weights.contains_key(v)).map(|v| polly_core::ids::vert(*v)).collect();
    let short = polly_core::mesh::short_list;
    Ok(Out::json(json!({
        "object": o.name,
        "verts_per_bone": per_bone,
        "unweighted": {"count": unweighted.len(), "verts": short(&unweighted)},
        "over_4_influences": short(&over),
        "unnormalized": {"count": unnormalized.len(), "verts": short(&unnormalized)},
    })))
}

// ---- pose -----------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct PoseBone {
    object: Option<String>,
    bone: String,
    /// Euler degrees about the bone's rest axes (+Y runs along the bone, head to tail). R = Rx·Ry·Rz: Z first, then Y, then X.
    rotation: Option<[f32; 3]>,
    translation: Option<[f32; 3]>,
    scale: Option<Scalar3>,
    /// Add to (rotation: apply after) the current pose instead of replacing it.
    #[serde(default)]
    relative: bool,
}

fn pose_bone(ctx: &mut Ctx, a: PoseBone) -> Result<Out> {
    let obj = rig(&ctx.doc.scene, resolve(&ctx.doc.scene, &a.object)?);
    let b = bone_of(&ctx.doc.scene, obj, &a.bone)?;
    edit_scene(ctx, "pose bone", |s| {
        let o = s.object_mut(obj)?;
        let p = o.pose.entry(b).or_default();
        if let Some(r) = a.rotation {
            let q = quat_from_euler_degrees(r);
            p.rotation = if a.relative { (p.rotation * q).normalize() } else { q };
        }
        if let Some(t) = a.translation {
            p.translation = if a.relative { p.translation + v3(t) } else { v3(t) };
        }
        if let Some(sc) = &a.scale {
            p.scale = if a.relative { p.scale * sc.vec() } else { sc.vec() };
        }
        if p.is_rest() {
            o.pose.remove(&b);
        }
        Ok(())
    })?;
    let out = Out::json(json!({"posed": a.bone, "hint": "render (bones overlay) to check; key_pose to record into a clip"}));
    super::view::maybe_preview(ctx, obj, out)
}

#[derive(Deserialize, JsonSchema)]
struct SolveIk {
    object: Option<String>,
    /// The end bone of the chain (its parent is the chain's root).
    bone: String,
    /// World point for the bone's tail.
    target: [f32; 3],
    /// World point the middle joint bends towards.
    pole: Option<[f32; 3]>,
}

fn solve_ik(ctx: &mut Ctx, a: SolveIk) -> Result<Out> {
    let s = &ctx.doc.scene;
    let obj = rig(s, resolve(s, &a.object)?);
    let lower = bone_of(s, obj, &a.bone)?;
    let o = s.object(obj)?;
    let sk = o.skeleton.clone().context("no skeleton")?;
    let inv = s.world_matrix(obj).inverse();
    let mut pose = o.pose.clone();
    let r = anim::two_bone_ik(&sk, &mut pose, lower, inv.transform_point3(v3(a.target)), a.pole.map(|p| inv.transform_point3(v3(p))))?;
    pose.retain(|_, p| !p.is_rest());
    edit_scene(ctx, "solve ik", |s| {
        s.object_mut(obj)?.pose = pose;
        Ok(())
    })?;
    let mut j = json!({"reached": r.miss < 1e-3, "hint": "key_pose to record it; render to check"});
    if r.miss >= 1e-3 {
        j["miss"] = json!((r.miss * 1000.0).round() / 1000.0);
        j["note"] = json!("target out of reach: the chain points straight at it");
    }
    super::view::maybe_preview(ctx, obj, Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct BakeIk {
    clip: String,
    object: Option<String>,
    /// The end bone of the chain (e.g. "shin.L").
    bone: String,
    /// Frame range [start, end], inclusive; every frame is keyed.
    frames: [f32; 2],
    /// World point for the bone's tail...
    target: Option<[f32; 3]>,
    /// ...or keep the tail where it is at the first frame (foot locking).
    #[serde(default)]
    lock: bool,
    /// World point the middle joint bends towards (default: the clip's own bend each frame).
    pole: Option<[f32; 3]>,
}

fn bake_ik(ctx: &mut Ctx, a: BakeIk) -> Result<Out> {
    let s = &ctx.doc.scene;
    let clip = s.resolve_clip(&a.clip)?;
    let obj = rig(s, resolve(s, &a.object)?);
    let lower = bone_of(s, obj, &a.bone)?;
    let o = s.object(obj)?;
    let sk = o.skeleton.clone().context("no skeleton")?;
    let upper = sk.bones[&lower].parent.with_context(|| format!("'{}' has no parent bone", a.bone))?;
    let c = &s.clips[&clip];
    let (f0, f1) = (a.frames[0].round() as i64, a.frames[1].round() as i64);
    if f1 < f0 || f0 < 0 {
        bail!("frames must be [start, end] with 0 <= start <= end");
    }
    // The rig's world matrix and pose at a frame, from the clip as it is now.
    let at = |f: i64| -> Result<(polly_core::glam::Mat4, BTreeMap<BoneId, anim::BonePose>)> {
        let t = (f as f32 / c.fps).min(c.length);
        let (tr, pose) = anim::evaluate_object(s, &[anim::ClipPlay { clip, time: t, weight: 1.0 }], obj).context("rig not found")?;
        Ok((s.world_matrix(obj) * o.transform.matrix().inverse() * tr.matrix(), pose))
    };
    let target = match (a.target, a.lock) {
        (Some(t), false) => v3(t),
        (None, true) => {
            let (w, pose) = at(f0)?;
            let m = sk.world_matrices(&pose);
            w.transform_point3((m[&lower].0 * m[&lower].1.inverse()).transform_point3(sk.bones[&lower].tail))
        }
        _ => bail!("give either `target` or `lock: true`"),
    };
    let mut keys = Vec::new();
    let mut worst: f32 = 0.0;
    for f in f0..=f1 {
        let (w, mut pose) = at(f)?;
        let inv = w.inverse();
        let r = anim::two_bone_ik(&sk, &mut pose, lower, inv.transform_point3(target), a.pole.map(|p| inv.transform_point3(v3(p))))?;
        worst = worst.max(r.miss);
        let t = (f as f32 / c.fps).min(c.length);
        for b in [upper, lower] {
            keys.push((b, t, pose.get(&b).copied().unwrap_or_default().rotation));
        }
    }
    let n = keys.len();
    edit_scene(ctx, "bake ik", |s| {
        let c = s.clips.get_mut(&clip).unwrap();
        for (b, t, q) in keys {
            c.channel_mut(Target::BoneRotation { object: obj, bone: b }).set_key(t, Vec4::from(q));
        }
        Ok(())
    })?;
    let mut j = json!({"keys_written": n, "target": round3(target), "frames": [f0, f1]});
    if worst >= 1e-3 {
        j["warning"] = json!(format!("the target was out of reach on some frames (up to {:.3} short)", worst));
    }
    Ok(Out::json(j))
}

/// Which side a bone name marks: Some(true) = left, Some(false) = right, None = centre.
fn side(name: &str) -> Option<bool> {
    if super::scene::mirror_name(name) == format!("{name}.mirror") {
        return None;
    }
    let l = name.to_lowercase();
    let marked = |c: char| ['.', '_', '-', ' '].iter().any(|sep| l.ends_with(&format!("{sep}{c}")) || l.starts_with(&format!("{c}{sep}")));
    Some(l.contains("left") || (!l.contains("right") && marked('l')))
}

/// Each bone's mirror partner (itself for centre bones and unmatched names).
fn partners(sk: &polly_core::anim::Skeleton) -> BTreeMap<BoneId, BoneId> {
    sk.bones
        .iter()
        .map(|(id, b)| {
            let p = sk.bones.iter().find(|(_, x)| x.name == super::scene::mirror_name(&b.name)).map_or(*id, |(pid, _)| *pid);
            (*id, p)
        })
        .collect()
}

#[derive(Deserialize, JsonSchema)]
struct MirrorPose {
    object: Option<String>,
    /// flip (default), left_to_right, right_to_left.
    mode: Option<String>,
}

fn mirror_pose(ctx: &mut Ctx, a: MirrorPose) -> Result<Out> {
    let s = &ctx.doc.scene;
    let obj = rig(s, resolve(s, &a.object)?);
    let o = s.object(obj)?;
    let Some(sk) = o.skeleton.clone() else { bail!("'{}' has no skeleton", o.name) };
    let mode = a.mode.as_deref().unwrap_or("flip");
    let from_side = match mode {
        "flip" => None,
        "left_to_right" => Some(true),
        "right_to_left" => Some(false),
        m => bail!("unknown mode '{m}' (flip, left_to_right, right_to_left)"),
    };
    let pairs = partners(&sk);
    let old = o.pose.clone();
    let mut new = if from_side.is_some() { old.clone() } else { BTreeMap::new() };
    for (b, p) in &pairs {
        if from_side.is_some_and(|l| side(&sk.bones[b].name) != Some(l) || p == b) {
            continue;
        }
        let pose = old.get(b).copied().unwrap_or_default();
        let m = polly_core::anim::mirror_bone_pose(&sk, *b, *p, pose);
        if m.is_rest() {
            new.remove(p);
        } else {
            new.insert(*p, m);
        }
    }
    let changed = pairs.keys().filter(|b| old.get(b) != new.get(b)).count();
    edit_scene(ctx, "mirror pose", |s| {
        s.object_mut(obj)?.pose = new;
        Ok(())
    })?;
    let unpaired: Vec<&str> = pairs.iter().filter(|(b, p)| b == p && side(&sk.bones[b].name).is_some()).map(|(b, _)| sk.bones[b].name.as_str()).collect();
    let mut j = json!({"bones_changed": changed});
    if !unpaired.is_empty() {
        j["warning"] = json!(format!("no partner bone found for: {} (mirrored onto themselves)", unpaired.join(", ")));
    }
    super::view::maybe_preview(ctx, obj, Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct CopyKeys {
    clip: String,
    object: Option<String>,
    /// Source frame range [start, end], inclusive.
    frames: [f32; 2],
    /// Frame the copy starts at.
    to_frame: f32,
    /// Swap left/right with mirrored values (default false).
    #[serde(default)]
    mirror: bool,
    /// Only these bones (source names; default all animated bones).
    bones: Option<Vec<String>>,
}

fn copy_keys(ctx: &mut Ctx, a: CopyKeys) -> Result<Out> {
    let s = &ctx.doc.scene;
    let clip = s.resolve_clip(&a.clip)?;
    let obj = rig(s, resolve(s, &a.object)?);
    let o = s.object(obj)?;
    let Some(sk) = o.skeleton.clone() else { bail!("'{}' has no skeleton", o.name) };
    let only: Option<BTreeSet<BoneId>> = a.bones.as_ref().map(|ns| ns.iter().map(|n| bone_of(s, obj, n)).collect::<Result<_>>()).transpose()?;
    let (t0, t1, to) = (secs(s, clip, a.frames[0]), secs(s, clip, a.frames[1]), secs(s, clip, a.to_frame));
    if t1 < t0 {
        bail!("frames must be [start, end] with start <= end");
    }
    let pairs = partners(&sk);
    let c = &s.clips[&clip];
    // Read everything first: the copy may overlap the source.
    let mut writes: Vec<(Target, f32, Vec4)> = Vec::new();
    for ch in &c.channels {
        let (bone, kind) = match ch.target {
            Target::BoneRotation { object, bone } if object == obj => (bone, 0),
            Target::BoneTranslation { object, bone } if object == obj => (bone, 1),
            Target::BoneScale { object, bone } if object == obj => (bone, 2),
            _ => continue,
        };
        if only.as_ref().is_some_and(|o| !o.contains(&bone)) {
            continue;
        }
        let dest = if a.mirror { pairs[&bone] } else { bone };
        for k in ch.keys.iter().filter(|k| k.time >= t0 - 1e-5 && k.time <= t1 + 1e-5) {
            let t = (k.time - t0 + to).min(c.length);
            let mut p = polly_core::anim::BonePose::default();
            match kind {
                0 => p.rotation = polly_core::glam::Quat::from_vec4(k.value),
                1 => p.translation = k.value.truncate(),
                _ => p.scale = k.value.truncate(),
            }
            if a.mirror {
                p = polly_core::anim::mirror_bone_pose(&sk, bone, dest, p);
            }
            let (target, v) = match kind {
                0 => (Target::BoneRotation { object: obj, bone: dest }, Vec4::from(p.rotation)),
                1 => (Target::BoneTranslation { object: obj, bone: dest }, p.translation.extend(0.0)),
                _ => (Target::BoneScale { object: obj, bone: dest }, p.scale.extend(0.0)),
            };
            writes.push((target, t, v));
        }
    }
    if writes.is_empty() {
        bail!("no bone keys of '{}' in frames {}..{}", o.name, a.frames[0], a.frames[1]);
    }
    let n = writes.len();
    edit_scene(ctx, "copy keys", |s| {
        let c = s.clips.get_mut(&clip).unwrap();
        for (target, t, v) in writes {
            c.channel_mut(target).set_key(t, v);
        }
        Ok(())
    })?;
    Ok(Out::json(json!({"keys_written": n, "clip": clip_json(&ctx.doc.scene, clip)})))
}

#[derive(Deserialize, JsonSchema)]
struct ClearPose {
    object: Option<String>,
    /// Bone names (default all).
    bones: Option<Vec<String>>,
}

fn clear_pose(ctx: &mut Ctx, a: ClearPose) -> Result<Out> {
    let obj = rig(&ctx.doc.scene, resolve(&ctx.doc.scene, &a.object)?);
    let ids = a.bones.as_ref().map(|ns| ns.iter().map(|n| bone_of(&ctx.doc.scene, obj, n)).collect::<Result<Vec<_>>>()).transpose()?;
    edit_scene(ctx, "clear pose", |s| {
        let o = s.object_mut(obj)?;
        match &ids {
            None => o.pose.clear(),
            Some(ids) => ids.iter().for_each(|b| {
                o.pose.remove(b);
            }),
        }
        Ok(())
    })?;
    Ok(ok("pose cleared"))
}

// ---- morphs ---------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct MorphName {
    object: Option<String>,
    name: String,
}

fn begin_morph(ctx: &mut Ctx, a: MorphName) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    if let Some(e) = &ctx.doc.scene.object(obj)?.morph_edit {
        bail!("already editing morph '{}'; end_morph or cancel_morph first", e.name);
    }
    let existing = edit_scene(ctx, "begin morph", |s| {
        let o = s.object_mut(obj)?;
        let edit = anim::begin_morph_edit(&o.mesh, &a.name);
        let mesh = Arc::make_mut(&mut o.mesh);
        // Continue from the stored shape of an existing morph.
        let existing = mesh.morphs.iter().find(|m| m.name == a.name).cloned();
        if let Some(m) = &existing {
            for (v, d) in &m.offsets {
                if let Some(vert) = mesh.verts.get_mut(v) {
                    vert.pos += *d;
                }
            }
        }
        o.morph_edit = Some(edit);
        Ok(existing.is_some())
    })?;
    Ok(ok(format!(
        "editing morph '{}'{}; move vertices with the usual tools (no adding/removing), then end_morph",
        a.name,
        if existing { " (starting from its current shape)" } else { "" }
    )))
}

fn end_morph(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    let (name, n) = edit_scene(ctx, "end morph", |s| {
        let o = s.object_mut(obj)?;
        let edit = o.morph_edit.take().ok_or_else(|| polly_core::Error::Invalid("no morph edit in progress".into()))?;
        let n = anim::end_morph_edit(Arc::make_mut(&mut o.mesh), &edit)?;
        Ok((edit.name, n))
    })?;
    Ok(Out::json(json!({"morph": name, "moved_verts": n, "hint": "preview with set_morph_weight or render pose"})))
}

fn cancel_morph(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    edit_scene(ctx, "cancel morph", |s| {
        let o = s.object_mut(obj)?;
        let edit = o.morph_edit.take().ok_or_else(|| polly_core::Error::Invalid("no morph edit in progress".into()))?;
        anim::cancel_morph_edit(Arc::make_mut(&mut o.mesh), &edit);
        Ok(())
    })?;
    Ok(ok("morph edit cancelled"))
}

#[derive(Deserialize, JsonSchema)]
struct MorphFrom {
    object: Option<String>,
    name: String,
    /// Object with the same vertices in the reshaped positions.
    source: String,
    /// Delete the source object afterwards (default true).
    #[serde(default = "yes")]
    delete_source: bool,
}

fn yes() -> bool {
    true
}

fn morph_from_object(ctx: &mut Ctx, a: MorphFrom) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    let src = ctx.doc.scene.resolve(Some(&a.source))?;
    if src == obj {
        bail!("source must be a different object");
    }
    let n = edit_scene(ctx, "morph from object", |s| {
        let shaped = s.object(src)?.mesh.clone();
        let o = s.object_mut(obj)?;
        let morph = anim::morph_from_mesh(&o.mesh, &shaped, &a.name)?;
        let n = morph.offsets.len();
        let mesh = Arc::make_mut(&mut o.mesh);
        match mesh.morphs.iter_mut().find(|m| m.name == a.name) {
            Some(m) => *m = morph,
            None => mesh.morphs.push(morph),
        }
        if a.delete_source {
            s.remove_object(src)?;
            s.active = Some(obj);
        }
        Ok(n)
    })?;
    Ok(Out::json(json!({"morph": a.name, "moved_verts": n})))
}

#[derive(Deserialize, JsonSchema)]
struct MorphWeight {
    object: Option<String>,
    morph: String,
    weight: f32,
}

fn set_morph_weight(ctx: &mut Ctx, a: MorphWeight) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    if !ctx.doc.scene.object(obj)?.mesh.morphs.iter().any(|m| m.name == a.morph) {
        bail!("no morph named '{}'", a.morph);
    }
    edit_scene(ctx, "morph weight", |s| {
        let o = s.object_mut(obj)?;
        if a.weight == 0.0 {
            o.morph_weights.remove(&a.morph);
        } else {
            o.morph_weights.insert(a.morph.clone(), a.weight);
        }
        Ok(())
    })?;
    super::view::maybe_preview(ctx, obj, ok(format!("'{}' weight {}", a.morph, a.weight)))
}

#[derive(Deserialize, JsonSchema)]
struct MorphRef {
    object: Option<String>,
    morph: String,
}

fn delete_morph(ctx: &mut Ctx, a: MorphRef) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    edit_scene(ctx, "delete morph", |s| {
        let o = s.object_mut(obj)?;
        let mesh = Arc::make_mut(&mut o.mesh);
        let before = mesh.morphs.len();
        mesh.morphs.retain(|m| m.name != a.morph);
        if mesh.morphs.len() == before {
            return polly_core::error::invalid(format!("no morph named '{}'", a.morph));
        }
        o.morph_weights.remove(&a.morph);
        for c in s.clips.values_mut() {
            c.channels.retain(|ch| !matches!(&ch.target, Target::MorphWeight { object, morph } if *object == obj && morph == &a.morph));
        }
        Ok(())
    })?;
    Ok(ok(format!("deleted morph '{}'", a.morph)))
}

// ---- clips ----------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct CreateClip {
    name: String,
    /// Length in frames (default 24).
    #[serde(default = "f24")]
    frames: f32,
    /// Default 24.
    #[serde(default = "f24")]
    fps: f32,
    #[serde(default = "yes")]
    looping: bool,
    /// Interpolation of its channels (default smooth).
    interpolation: Option<Interpolation>,
}

fn f24() -> f32 {
    24.0
}

fn create_clip(ctx: &mut Ctx, a: CreateClip) -> Result<Out> {
    if ctx.doc.scene.clips.values().any(|c| c.name == a.name) {
        bail!("clip '{}' already exists", a.name);
    }
    if a.fps <= 0.0 || a.frames <= 0.0 {
        bail!("frames and fps must be > 0");
    }
    edit_scene(ctx, "create clip", |s| {
        let mut c = anim::Clip::new(&a.name, a.frames / a.fps, a.fps);
        c.looping = a.looping;
        c.interpolation = a.interpolation.unwrap_or(Interpolation::Smooth);
        Ok(s.add_clip(c))
    })?;
    Ok(Out::json(json!({"clip": a.name, "frames": a.frames, "fps": a.fps})))
}

#[derive(Deserialize, JsonSchema)]
struct UpdateClip {
    clip: String,
    name: Option<String>,
    /// New length in frames.
    frames: Option<f32>,
    fps: Option<f32>,
    looping: Option<bool>,
    loop_start_frame: Option<f32>,
    /// Negative clears it (loop to the end).
    loop_end_frame: Option<f32>,
    /// Interpolation of channels created from now on (existing ones: set_interpolation).
    interpolation: Option<Interpolation>,
}

fn update_clip(ctx: &mut Ctx, a: UpdateClip) -> Result<Out> {
    let id = ctx.doc.scene.resolve_clip(&a.clip)?;
    edit_scene(ctx, "update clip", |s| {
        if let Some(n) = &a.name {
            if s.clips.iter().any(|(i, c)| *i != id && &c.name == n) {
                return polly_core::error::invalid(format!("clip '{n}' already exists"));
            }
        }
        let c = s.clips.get_mut(&id).unwrap();
        if let Some(fps) = a.fps {
            if fps <= 0.0 {
                return polly_core::error::invalid("fps must be > 0");
            }
            // Keep the length in frames; key times stay in seconds.
            c.length = c.length * c.fps / fps;
            c.fps = fps;
        }
        if let Some(n) = &a.name {
            c.name = n.clone();
        }
        if let Some(f) = a.frames {
            c.length = f.max(0.0) / c.fps;
        }
        if let Some(l) = a.looping {
            c.looping = l;
        }
        if let Some(i) = a.interpolation {
            c.interpolation = i;
        }
        if let Some(f) = a.loop_start_frame {
            c.loop_start = f.max(0.0) / c.fps;
        }
        if let Some(f) = a.loop_end_frame {
            c.loop_end = (f >= 0.0).then(|| f / c.fps);
        }
        Ok(())
    })?;
    Ok(Out::json(clip_json(&ctx.doc.scene, id)))
}

fn clip_json(s: &Scene, id: ClipId) -> Value {
    let c = &s.clips[&id];
    let (ls, le) = c.loop_range();
    let f = |t: f32| (t * c.fps * 100.0).round() / 100.0;
    json!({
        "clip": c.name,
        "frames": f(c.length),
        "fps": c.fps,
        "looping": c.looping,
        "loop_frames": [f(ls), f(le)],
        "interpolation": c.interpolation,
        "channels": c.channels.len(),
        "keys": c.key_count(),
    })
}

#[derive(Deserialize, JsonSchema)]
struct ClipArg {
    clip: String,
}

fn delete_clip(ctx: &mut Ctx, a: ClipArg) -> Result<Out> {
    let id = ctx.doc.scene.resolve_clip(&a.clip)?;
    edit_scene(ctx, "delete clip", |s| {
        s.clips.remove(&id);
        Ok(())
    })?;
    Ok(ok(format!("deleted clip '{}'", a.clip)))
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn list_clips(ctx: &mut Ctx, _: Empty) -> Result<Out> {
    let s = &ctx.doc.scene;
    Ok(Out::json(json!({"clips": s.clips.keys().map(|id| clip_json(s, *id)).collect::<Vec<_>>()})))
}

fn target_json(s: &Scene, t: &Target) -> Value {
    let obj = |o: &ObjectId| s.objects.get(o).map_or(format!("o{o}"), |x| x.name.clone());
    let bone = |o: &ObjectId, b: &BoneId| {
        s.objects.get(o).and_then(|x| x.skeleton.as_ref()).map_or(format!("#{b}"), |sk| sk.name(*b).to_string())
    };
    match t {
        Target::BoneTranslation { object, bone: b } => json!({"object": obj(object), "bone": bone(object, b), "property": "translation"}),
        Target::BoneRotation { object, bone: b } => json!({"object": obj(object), "bone": bone(object, b), "property": "rotation"}),
        Target::BoneScale { object, bone: b } => json!({"object": obj(object), "bone": bone(object, b), "property": "scale"}),
        Target::ObjectTranslation { object } => json!({"object": obj(object), "property": "translation"}),
        Target::ObjectRotation { object } => json!({"object": obj(object), "property": "rotation"}),
        Target::ObjectScale { object } => json!({"object": obj(object), "property": "scale"}),
        Target::MorphWeight { object, morph } => json!({"object": obj(object), "morph": morph}),
    }
}

fn describe_clip(ctx: &mut Ctx, a: ClipArg) -> Result<Out> {
    let s = &ctx.doc.scene;
    let id = s.resolve_clip(&a.clip)?;
    let c = &s.clips[&id];
    let channels: Vec<Value> = c
        .channels
        .iter()
        .map(|ch| {
            let mut j = target_json(s, &ch.target);
            j["frames"] = json!(ch.keys.iter().map(|k| (k.time * c.fps * 100.0).round() / 100.0).collect::<Vec<_>>());
            j["interpolation"] = json!(ch.interpolation);
            j
        })
        .collect();
    let mut j = clip_json(s, id);
    j["channels"] = json!(channels);
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct KeyPose {
    clip: String,
    frame: f32,
    object: Option<String>,
    /// Bones to key (default: all bones that are posed away from rest, plus bones already animated in this clip).
    bones: Option<Vec<String>>,
    /// Bone properties to key: any of "rotation", "translation", "scale" (default rotation and translation).
    properties: Option<Vec<String>>,
    /// Also key the object's static morph weights.
    #[serde(default)]
    morphs: bool,
    /// Also key the object's transform (position/rotation/scale).
    #[serde(default)]
    object_transform: bool,
}

fn key_pose(ctx: &mut Ctx, a: KeyPose) -> Result<Out> {
    let s = &ctx.doc.scene;
    let clip = s.resolve_clip(&a.clip)?;
    let mesh_obj = resolve(s, &a.object)?;
    let obj = rig(s, mesh_obj);
    let o = s.object(obj)?;
    let mo = s.object(mesh_obj)?;
    let props: BTreeSet<String> = a.properties.clone().unwrap_or_else(|| vec!["rotation".into(), "translation".into()]).into_iter().collect();
    for p in &props {
        if !["rotation", "translation", "scale"].contains(&p.as_str()) {
            bail!("unknown property '{p}' (rotation, translation, scale)");
        }
    }
    let bones: Vec<BoneId> = match (&a.bones, &o.skeleton) {
        (Some(ns), _) => ns.iter().map(|n| bone_of(s, obj, n)).collect::<Result<_>>()?,
        (None, Some(sk)) => {
            let animated: BTreeSet<BoneId> = s.clips[&clip]
                .channels
                .iter()
                .filter_map(|ch| match ch.target {
                    Target::BoneRotation { object, bone } | Target::BoneTranslation { object, bone } | Target::BoneScale { object, bone }
                        if object == obj => Some(bone),
                    _ => None,
                })
                .collect();
            sk.bones.keys().copied().filter(|b| o.pose.contains_key(b) || animated.contains(b)).collect()
        }
        (None, None) => vec![],
    };
    if bones.is_empty() && !a.morphs && !a.object_transform {
        bail!("nothing to key: pose some bones first (pose_bone), or pass bones / morphs / object_transform");
    }
    let t = secs(s, clip, a.frame);
    let (pose, weights, tr) = (o.pose.clone(), mo.morph_weights.clone(), mo.transform);
    let morph_names: Vec<String> = mo.mesh.morphs.iter().map(|m| m.name.clone()).collect();
    let mut n = 0;
    edit_scene(ctx, "key pose", |s| {
        let c = s.clips.get_mut(&clip).unwrap();
        for b in &bones {
            let p = pose.get(b).copied().unwrap_or_default();
            if props.contains("rotation") {
                c.channel_mut(Target::BoneRotation { object: obj, bone: *b }).set_key(t, Vec4::from(p.rotation));
                n += 1;
            }
            if props.contains("translation") {
                c.channel_mut(Target::BoneTranslation { object: obj, bone: *b }).set_key(t, p.translation.extend(0.0));
                n += 1;
            }
            if props.contains("scale") {
                c.channel_mut(Target::BoneScale { object: obj, bone: *b }).set_key(t, p.scale.extend(0.0));
                n += 1;
            }
        }
        if a.morphs {
            for m in &morph_names {
                let w = weights.get(m).copied().unwrap_or(0.0);
                c.channel_mut(Target::MorphWeight { object: mesh_obj, morph: m.clone() }).set_key(t, Vec4::new(w, 0.0, 0.0, 0.0));
                n += 1;
            }
        }
        if a.object_transform {
            c.channel_mut(Target::ObjectTranslation { object: mesh_obj }).set_key(t, tr.translation.extend(0.0));
            c.channel_mut(Target::ObjectRotation { object: mesh_obj }).set_key(t, Vec4::from(tr.rotation));
            c.channel_mut(Target::ObjectScale { object: mesh_obj }).set_key(t, tr.scale.extend(0.0));
            n += 3;
        }
        if t > c.length {
            c.length = t;
        }
        Ok(())
    })?;
    let names: Vec<&str> = {
        let sk = ctx.doc.scene.objects[&obj].skeleton.as_ref();
        bones.iter().map(|b| sk.map_or("?", |sk| sk.name(*b))).collect()
    };
    Ok(Out::json(json!({"clip": a.clip, "frame": a.frame, "keys_set": n, "bones": names})))
}

#[derive(Deserialize, JsonSchema)]
struct SetKey {
    clip: String,
    frame: f32,
    object: Option<String>,
    /// Bone to key (omit for a morph or object property).
    bone: Option<String>,
    /// Morph to key (value = [weight]).
    morph: Option<String>,
    /// "rotation" (value = Euler degrees, R = Rx·Ry·Rz), "translation" or "scale" (value = [x,y,z]). Ignored for morphs.
    property: Option<String>,
    value: Vec<f32>,
    interpolation: Option<Interpolation>,
}

fn set_key(ctx: &mut Ctx, a: SetKey) -> Result<Out> {
    let s = &ctx.doc.scene;
    let clip = s.resolve_clip(&a.clip)?;
    let obj = resolve(s, &a.object)?;
    let prop = a.property.as_deref().unwrap_or(if a.morph.is_some() { "weight" } else { "rotation" });
    let vec3 = || -> Result<Vec3> {
        match a.value.as_slice() {
            [x, y, z] => Ok(Vec3::new(*x, *y, *z)),
            [u] if prop == "scale" => Ok(Vec3::splat(*u)),
            _ => bail!("{prop} needs 3 values"),
        }
    };
    let (target, value) = match (&a.bone, &a.morph) {
        (Some(_), Some(_)) => bail!("give either bone or morph, not both"),
        (None, Some(m)) => {
            if !s.object(obj)?.mesh.morphs.iter().any(|x| &x.name == m) {
                bail!("no morph named '{m}'");
            }
            let [w] = a.value.as_slice() else { bail!("a morph key needs one value (the weight)") };
            (Target::MorphWeight { object: obj, morph: m.clone() }, Vec4::new(*w, 0.0, 0.0, 0.0))
        }
        (Some(b), None) => {
            let bone = bone_of(s, obj, b)?;
            let r = rig(s, obj);
            match prop {
                "rotation" => (Target::BoneRotation { object: r, bone }, Vec4::from(quat_from_euler_degrees(vec3()?.to_array()))),
                "translation" => (Target::BoneTranslation { object: r, bone }, vec3()?.extend(0.0)),
                "scale" => (Target::BoneScale { object: r, bone }, vec3()?.extend(0.0)),
                p => bail!("unknown property '{p}'"),
            }
        }
        (None, None) => match prop {
            "rotation" => (Target::ObjectRotation { object: obj }, Vec4::from(quat_from_euler_degrees(vec3()?.to_array()))),
            "translation" => (Target::ObjectTranslation { object: obj }, vec3()?.extend(0.0)),
            "scale" => (Target::ObjectScale { object: obj }, vec3()?.extend(0.0)),
            p => bail!("unknown property '{p}'"),
        },
    };
    let t = secs(s, clip, a.frame);
    edit_scene(ctx, "set key", |s| {
        let c = s.clips.get_mut(&clip).unwrap();
        let ch = c.channel_mut(target);
        ch.set_key(t, value);
        if let Some(i) = a.interpolation {
            ch.interpolation = i;
        }
        if t > c.length {
            c.length = t;
        }
        Ok(())
    })?;
    Ok(ok("key set"))
}

#[derive(Deserialize, JsonSchema)]
struct ChannelFilter {
    clip: String,
    object: Option<String>,
    bone: Option<String>,
    morph: Option<String>,
}

fn matches(s: &Scene, f: &ChannelFilter, t: &Target) -> Result<bool> {
    let obj = f.object.as_deref().map(|o| s.resolve(Some(o))).transpose()?;
    if obj.is_some_and(|o| o != t.object()) {
        return Ok(false);
    }
    if let Some(b) = &f.bone {
        let id = bone_of(s, t.object(), b).ok();
        return Ok(matches!(t,
            Target::BoneTranslation { bone, .. } | Target::BoneRotation { bone, .. } | Target::BoneScale { bone, .. }
            if Some(*bone) == id));
    }
    if let Some(m) = &f.morph {
        return Ok(matches!(t, Target::MorphWeight { morph, .. } if morph == m));
    }
    Ok(true)
}

#[derive(Deserialize, JsonSchema)]
struct DeleteKeys {
    #[serde(flatten)]
    filter: ChannelFilter,
    /// Inclusive frame range [from, to] (default all keys).
    frames: Option<[f32; 2]>,
}

fn delete_keys(ctx: &mut Ctx, a: DeleteKeys) -> Result<Out> {
    let s = &ctx.doc.scene;
    let clip = s.resolve_clip(&a.filter.clip)?;
    let fps = s.clips[&clip].fps;
    let keep: Vec<bool> = s.clips[&clip].channels.iter().map(|ch| matches(s, &a.filter, &ch.target).map(|m| !m)).collect::<Result<_>>()?;
    let mut removed = 0;
    edit_scene(ctx, "delete keys", |s| {
        let c = s.clips.get_mut(&clip).unwrap();
        for (ch, keep) in c.channels.iter_mut().zip(&keep) {
            if *keep {
                continue;
            }
            let before = ch.keys.len();
            ch.keys.retain(|k| a.frames.is_some_and(|[f0, f1]| { let f = k.time * fps; f < f0 - 1e-3 || f > f1 + 1e-3 }));
            removed += before - ch.keys.len();
        }
        c.channels.retain(|ch| !ch.keys.is_empty());
        Ok(())
    })?;
    Ok(Out::json(json!({"keys_removed": removed})))
}

#[derive(Deserialize, JsonSchema)]
struct SetInterp {
    #[serde(flatten)]
    filter: ChannelFilter,
    interpolation: Interpolation,
}

fn set_interpolation(ctx: &mut Ctx, a: SetInterp) -> Result<Out> {
    let s = &ctx.doc.scene;
    let clip = s.resolve_clip(&a.filter.clip)?;
    let hit: Vec<bool> = s.clips[&clip].channels.iter().map(|ch| matches(s, &a.filter, &ch.target)).collect::<Result<_>>()?;
    let n = hit.iter().filter(|h| **h).count();
    edit_scene(ctx, "set interpolation", |s| {
        for (ch, h) in s.clips.get_mut(&clip).unwrap().channels.iter_mut().zip(&hit) {
            if *h {
                ch.interpolation = a.interpolation;
            }
        }
        Ok(())
    })?;
    Ok(Out::json(json!({"channels_changed": n})))
}

#[derive(Deserialize, JsonSchema)]
struct RenderAnim {
    /// Clips to blend: [{"clip":"walk"}] or several with weights. `frame` is ignored here.
    clips: Vec<PlayArg>,
    /// Frames to show (default: 8 evenly spaced over the first clip).
    frames: Option<Vec<f32>>,
    /// Number of evenly spaced frames when `frames` is not given (default 8).
    count: Option<u32>,
    #[serde(default)]
    view: polly_render::View,
    /// Tile size in px (default 256).
    tile_size: Option<u32>,
    focus: Option<Vec<String>>,
    /// Default: wireframe and grid, no selection highlight.
    overlays: Option<polly_render::Overlays>,
    /// Variants to show instead of the active ones, e.g. {"hair": "braids"}.
    variants: Option<BTreeMap<String, String>>,
}

fn render_animation(ctx: &mut Ctx, a: RenderAnim) -> Result<Out> {
    let s = &ctx.doc.scene;
    let Some(first) = a.clips.first() else { bail!("give at least one clip") };
    let id0 = s.resolve_clip(&first.clip)?;
    let c0 = &s.clips[&id0];
    let total = c0.length * c0.fps;
    let frames: Vec<f32> = match &a.frames {
        Some(f) => f.clone(),
        None => {
            let n = a.count.unwrap_or(8).clamp(1, 32);
            (0..n).map(|i| if n == 1 { 0.0 } else { total * i as f32 / (n - 1) as f32 }).map(|f| (f * 100.0).round() / 100.0).collect()
        }
    };
    let tile = a.tile_size.unwrap_or(256).clamp(96, 512);
    let choice = super::variants::choice(s, &a.variants)?;
    let focus = a
        .focus
        .as_ref()
        .map(|ns| ns.iter().map(|n| s.resolve(Some(n))).collect::<polly_core::Result<Vec<_>>>())
        .transpose()?;
    let opts = polly_render::RenderOptions {
        view: a.view.clone(),
        width: tile,
        height: tile,
        focus,
        // Selection highlighting just clutters motion checks.
        overlays: a.overlays.clone().unwrap_or(polly_render::Overlays { selection: false, ..Default::default() }),
        legend: false,
        ..Default::default()
    };
    let cols = (frames.len() as f32).sqrt().ceil() as u32;
    let rows = (frames.len() as u32).div_ceil(cols);
    let gap = 4;
    let mut img = image::RgbaImage::from_pixel(cols * tile + gap * (cols + 1), rows * tile + gap * (rows + 1), image::Rgba([18, 19, 23, 255]));
    let renderer = ctx.services.renderer()?;
    for (i, f) in frames.iter().enumerate() {
        let args: Vec<PlayArg> = a.clips.iter().map(|p| PlayArg { frame: *f, ..p.clone() }).collect();
        let posed = polly_core::variants::display(s, &plays(s, &args)?, &choice);
        let tile_img = renderer.render(&posed, &opts)?.image;
        let (x, y) = (gap + (i as u32 % cols) * (tile + gap), gap + (i as u32 / cols) * (tile + gap));
        image::imageops::overlay(&mut img, &tile_img, x as i64, y as i64);
        let label = format!("frame {f}");
        let r = polly_render::canvas::label_rect(&label, 1, 0.0, 0.0);
        let r = polly_render::canvas::Rect { x: x as i32 + 4, y: y as i32 + 4, ..r };
        polly_render::canvas::boxed_label(&mut img, r, &label, 1, [20, 20, 26, 220], polly_render::canvas::WHITE);
    }
    drop(renderer);
    Ok(Out::json(json!({"frames": frames, "clips": a.clips.iter().map(|p| &p.clip).collect::<Vec<_>>()})).with_image(super::view::png(&img)?))
}

#[derive(Deserialize, JsonSchema)]
struct Bind {
    /// The mesh to skin (default: active).
    object: Option<String>,
    /// The object with the skeleton.
    armature: String,
    /// Parent the object to the armature, keeping its world placement (default true).
    #[serde(default = "yes")]
    parent: bool,
}

fn bind_to_armature(ctx: &mut Ctx, a: Bind) -> Result<Out> {
    let s = &ctx.doc.scene;
    let obj = resolve(s, &a.object)?;
    let arm = s.resolve(Some(&a.armature))?;
    if obj == arm {
        bail!("an object cannot be bound to itself");
    }
    let Some(sk) = s.object(arm)?.skeleton.clone() else {
        bail!("'{}' has no skeleton (add bones with add_bone first)", a.armature);
    };
    if s.object(obj)?.skeleton.is_some() {
        bail!("'{}' has its own skeleton; bind only meshes without one", s.object(obj)?.name);
    }
    if s.object(arm)?.skin.is_some() {
        bail!("'{}' is itself bound to an armature", a.armature);
    }
    edit_scene(ctx, "bind to armature", |s| {
        if a.parent {
            let world = s.world_matrix(obj);
            s.set_parent(obj, Some(arm))?;
            let local = s.world_matrix(arm).inverse() * world;
            let (sc, r, t) = local.to_scale_rotation_translation();
            let tr = &mut s.object_mut(obj)?.transform;
            tr.translation = t;
            tr.rotation = r;
            tr.scale = sc;
        }
        let o = s.object_mut(obj)?;
        o.skin = Some(polly_core::scene::SkinBinding { armature: arm });
        // Weights that don't refer to this skeleton's bones are meaningless now.
        anim::prune_weights(Arc::make_mut(&mut o.mesh), Some(&sk));
        Ok(())
    })?;
    let o = ctx.doc.scene.object(obj)?;
    let mut j = json!({"object": o.name, "armature": a.armature, "weighted_verts": o.mesh.weights.len(), "verts": o.mesh.verts.len()});
    if o.mesh.weights.is_empty() {
        j["hint"] = json!("now give it weights: auto_weights (uses the armature's bones) or assign_weights");
    }
    Ok(Out::json(j))
}

fn unbind(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let obj = resolve(&ctx.doc.scene, &a.object)?;
    if ctx.doc.scene.object(obj)?.skin.is_none() {
        bail!("'{}' is not bound to an armature", ctx.doc.scene.object(obj)?.name);
    }
    edit_scene(ctx, "unbind", |s| {
        s.object_mut(obj)?.skin = None;
        Ok(())
    })?;
    Ok(ok("unbound (weights kept)"))
}

#[derive(Deserialize, JsonSchema)]
struct Attach {
    /// The gear to attach (default: active).
    object: Option<String>,
    /// Object with the skeleton (a bound mesh resolves to its armature).
    armature: String,
    bone: String,
}

fn attach_to_bone(ctx: &mut Ctx, a: Attach) -> Result<Out> {
    let s = &ctx.doc.scene;
    let obj = resolve(s, &a.object)?;
    let arm = rig(s, s.resolve(Some(&a.armature))?);
    if obj == arm {
        bail!("an object cannot be attached to its own bones");
    }
    if s.object(obj)?.skin.is_some() {
        bail!("'{}' is skinned (bound to an armature); attach rigid gear only, or unbind it first", s.object(obj)?.name);
    }
    let bone = bone_of(s, arm, &a.bone)?;
    edit_scene(ctx, "attach to bone", |s| {
        let world = s.world_matrix(obj);
        s.set_parent(obj, Some(arm))?;
        let o = s.object_mut(obj)?;
        o.parent_bone = Some(bone);
        // Keep the world placement: local = (rig × bone rest frame)⁻¹ × world.
        let frame = s.world_matrix(arm) * s.bone_frame(obj);
        let (sc, r, t) = (frame.inverse() * world).to_scale_rotation_translation();
        s.object_mut(obj)?.transform = polly_core::scene::Transform { translation: t, rotation: r, scale: sc };
        Ok(())
    })?;
    let name = &ctx.doc.scene.objects[&arm].name;
    let out = Out::json(json!({"attached": ctx.doc.scene.objects[&obj].name, "to": format!("{name} / {}", a.bone)}));
    super::view::maybe_preview(ctx, obj, out)
}

#[derive(Deserialize, JsonSchema)]
struct Transfer {
    /// The gear to weight (default: active).
    object: Option<String>,
    /// A skinned mesh to copy weights from (usually the body).
    from: String,
    /// Report vertices further than this from the source surface (default: 10% of the source's size).
    max_distance: Option<f32>,
    /// Bind the gear to the source's armature (default true).
    #[serde(default = "yes")]
    bind: bool,
}

fn transfer_weights(ctx: &mut Ctx, a: Transfer) -> Result<Out> {
    let s = &ctx.doc.scene;
    let obj = resolve(s, &a.object)?;
    let from = s.resolve(Some(&a.from))?;
    if obj == from {
        bail!("object and source are the same");
    }
    let arm = s.skeleton_source(from).with_context(|| format!("'{}' isn't skinned (no skeleton or armature)", a.from))?;
    if let Some(cur) = s.skeleton_source(obj) {
        if cur != arm {
            bail!("'{}' is skinned to a different skeleton; unbind it first", s.object(obj)?.name);
        }
    }
    if s.object(obj)?.skeleton.is_some() {
        bail!("'{}' has its own skeleton; transfer onto meshes without one", s.object(obj)?.name);
    }
    let source = s.object(from)?.mesh.clone();
    let source_to_skel = s.to_skeleton_space(from);
    let target_to_skel = s.world_matrix(arm).inverse() * s.world_matrix(obj);
    let size = source.bounds().map_or(1.0, |(lo, hi)| (hi - lo).length());
    let limit = a.max_distance.unwrap_or(size * 0.1);
    let mut far = vec![];
    let mut worst = 0.0f32;
    edit_scene(ctx, "transfer weights", |s| {
        let o = s.object_mut(obj)?;
        let d = anim::transfer_weights(Arc::make_mut(&mut o.mesh), target_to_skel, &source, source_to_skel)?;
        far = d.iter().filter(|(_, x)| **x > limit).map(|(v, _)| *v).collect();
        worst = d.values().copied().fold(0.0, f32::max);
        if a.bind && o.skin.is_none() && obj != arm {
            o.skin = Some(polly_core::scene::SkinBinding { armature: arm });
            let world = s.world_matrix(obj);
            s.set_parent(obj, Some(arm))?;
            let (sc, r, t) = (s.world_matrix(arm).inverse() * world).to_scale_rotation_translation();
            s.object_mut(obj)?.transform = polly_core::scene::Transform { translation: t, rotation: r, scale: sc };
        }
        Ok(())
    })?;
    let o = ctx.doc.scene.object(obj)?;
    let mut j = json!({
        "object": o.name,
        "weighted_verts": o.mesh.weights.len(),
        "armature": ctx.doc.scene.objects[&arm].name,
        "max_distance_to_source": (worst * 1000.0).round() / 1000.0,
    });
    if !far.is_empty() {
        j["far_from_source"] = json!(polly_core::ids::compress("v", far.iter().copied()));
        j["hint"] = json!("these vertices are far from the source surface; check their weights (render show_weights) or assign_weights by hand");
    }
    Ok(Out::json(j))
}
