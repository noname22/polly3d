//! Rigging and animation: skeletons, skin weights, morph targets, clips,
//! clip blending and CPU deformation.
//!
//! Bones are defined like in most DCC tools: a head and tail point in the
//! object's space (the rest pose) and an optional parent. A bone's local frame
//! has its origin at the head and +Y pointing to the tail. Poses and keys are
//! expressed in that local frame, relative to the rest pose.
//!
//! Evaluation never mutates the document: [`pose_scene`] returns a display copy
//! of the scene with deformed meshes, animated transforms and bones moved to
//! their posed positions. Element ids are unchanged, so labels and picking work
//! on posed geometry.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result, invalid};
use crate::mesh::{Mesh, VertId};
use crate::scene::{ObjectId, Scene};

pub type BoneId = u32;
pub type ClipId = u32;

/// Maximum bone influences per vertex (glTF / game-engine limit).
pub const MAX_INFLUENCES: usize = 4;

// ---- skeleton -------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bone {
    pub name: String,
    #[serde(default)]
    pub parent: Option<BoneId>,
    /// Rest pose, object space.
    pub head: Vec3,
    pub tail: Vec3,
    /// Twist around the bone axis, degrees.
    #[serde(default)]
    pub roll: f32,
}

impl Bone {
    /// Rest frame in object space: origin at the head, +Y towards the tail.
    pub fn rest_matrix(&self) -> Mat4 {
        let y = (self.tail - self.head).normalize_or(Vec3::Y);
        let reference = if y.dot(Vec3::Z).abs() < 0.99 { Vec3::Z } else { Vec3::X };
        let x = y.cross(reference).normalize();
        let z = x.cross(y);
        let r = Quat::from_mat3(&glam::Mat3::from_cols(x, y, z)) * Quat::from_rotation_y(self.roll.to_radians());
        Mat4::from_rotation_translation(r.normalize(), self.head)
    }

    pub fn length(&self) -> f32 {
        (self.tail - self.head).length()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Skeleton {
    pub bones: BTreeMap<BoneId, Bone>,
    next: BoneId,
}

impl Skeleton {
    pub fn add(&mut self, bone: Bone) -> Result<BoneId> {
        if self.bones.values().any(|b| b.name == bone.name) {
            return invalid(format!("a bone named '{}' already exists", bone.name));
        }
        if let Some(p) = bone.parent {
            if !self.bones.contains_key(&p) {
                return invalid(format!("parent bone #{p} does not exist"));
            }
        }
        if (bone.tail - bone.head).length() < 1e-6 {
            return invalid("bone head and tail must differ");
        }
        let id = self.next;
        self.next += 1;
        self.bones.insert(id, bone);
        Ok(id)
    }

    pub fn find(&self, name: &str) -> Result<BoneId> {
        self.bones
            .iter()
            .find(|(_, b)| b.name == name)
            .map(|(id, _)| *id)
            .ok_or_else(|| Error::NotFound { kind: "bone", id: name.to_string() })
    }

    pub fn name(&self, id: BoneId) -> &str {
        self.bones.get(&id).map_or("?", |b| b.name.as_str())
    }

    /// Bones ordered so parents come before children.
    pub fn ordered(&self) -> Vec<BoneId> {
        let mut out = Vec::with_capacity(self.bones.len());
        let mut done = BTreeSet::new();
        while out.len() < self.bones.len() {
            let before = out.len();
            for (&id, b) in &self.bones {
                if !done.contains(&id) && b.parent.is_none_or(|p| done.contains(&p) || !self.bones.contains_key(&p)) {
                    out.push(id);
                    done.insert(id);
                }
            }
            if out.len() == before {
                break; // cycle; shouldn't happen
            }
        }
        out
    }

    /// Would making `parent` the parent of `bone` create a cycle?
    pub fn creates_cycle(&self, bone: BoneId, parent: BoneId) -> bool {
        let mut cur = Some(parent);
        while let Some(c) = cur {
            if c == bone {
                return true;
            }
            cur = self.bones.get(&c).and_then(|b| b.parent);
        }
        false
    }

    /// Remove a bone; its children are re-parented to its parent.
    pub fn remove(&mut self, id: BoneId) -> Result<Bone> {
        let b = self.bones.remove(&id).ok_or_else(|| Error::NotFound { kind: "bone", id: format!("#{id}") })?;
        for other in self.bones.values_mut() {
            if other.parent == Some(id) {
                other.parent = b.parent;
            }
        }
        Ok(b)
    }

    /// Object-space matrices of every bone for a pose: (posed, rest).
    pub fn world_matrices(&self, pose: &BTreeMap<BoneId, BonePose>) -> HashMap<BoneId, (Mat4, Mat4)> {
        let mut out: HashMap<BoneId, (Mat4, Mat4)> = HashMap::new();
        for id in self.ordered() {
            let b = &self.bones[&id];
            let rest = b.rest_matrix();
            let (parent_posed, parent_rest) = b.parent.and_then(|p| out.get(&p).copied()).unwrap_or((Mat4::IDENTITY, Mat4::IDENTITY));
            let local_rest = parent_rest.inverse() * rest;
            let local_pose = pose.get(&id).copied().unwrap_or_default().matrix();
            out.insert(id, (parent_posed * local_rest * local_pose, rest));
        }
        out
    }
}

/// A bone's pose relative to its rest frame (in the bone's local space).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BonePose {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for BonePose {
    fn default() -> Self {
        BonePose { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
}

impl BonePose {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
    pub fn is_rest(&self) -> bool {
        *self == BonePose::default()
    }
}

// ---- skin weights ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WeightMode {
    /// Set the bone's weight (other bones keep theirs).
    #[default]
    Replace,
    Add,
    Subtract,
}

/// Assign `weight` for `bone` to the given vertices.
pub fn assign_weights(mesh: &mut Mesh, verts: &BTreeSet<VertId>, bone: BoneId, weight: f32, mode: WeightMode) {
    for v in verts {
        let list = mesh.weights.entry(*v).or_default();
        let cur = list.iter().position(|(b, _)| *b == bone);
        let old = cur.map_or(0.0, |i| list[i].1);
        let w = match mode {
            WeightMode::Replace => weight,
            WeightMode::Add => old + weight,
            WeightMode::Subtract => old - weight,
        }
        .clamp(0.0, 1.0);
        match (cur, w > 1e-6) {
            (Some(i), true) => list[i].1 = w,
            (Some(i), false) => {
                list.remove(i);
            }
            (None, true) => list.push((bone, w)),
            (None, false) => {}
        }
        if list.is_empty() {
            mesh.weights.remove(v);
        }
    }
}

/// Keep the strongest [`MAX_INFLUENCES`] bones per vertex and make weights sum to 1.
pub fn normalize_weights(mesh: &mut Mesh) {
    mesh.weights.retain(|_, list| {
        list.retain(|(_, w)| *w > 1e-6);
        list.sort_by(|a, b| b.1.total_cmp(&a.1));
        list.truncate(MAX_INFLUENCES);
        let sum: f32 = list.iter().map(|(_, w)| w).sum();
        if sum > 0.0 {
            list.iter_mut().for_each(|(_, w)| *w /= sum);
        }
        !list.is_empty()
    });
}

/// Drop weights of vertices / bones that no longer exist.
pub fn prune_weights(mesh: &mut Mesh, skeleton: Option<&Skeleton>) {
    let verts: BTreeSet<VertId> = mesh.verts.keys().copied().collect();
    mesh.weights.retain(|v, list| {
        if let Some(s) = skeleton {
            list.retain(|(b, _)| s.bones.contains_key(b));
        }
        verts.contains(v) && !list.is_empty()
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum AutoWeights {
    /// Each vertex follows only its nearest bone (classic rigid low-poly look).
    Rigid,
    /// Blend between nearby bones by inverse distance; `falloff` (default 0.5)
    /// widens the blend zone: bones within (1 + falloff) x the nearest distance share.
    Smooth {
        #[serde(default = "half")]
        falloff: f32,
    },
}

fn half() -> f32 {
    0.5
}

fn segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Compute weights for `verts` from distances to `bones` (replacing existing weights of those verts).
pub fn auto_weights(mesh: &mut Mesh, skeleton: &Skeleton, verts: &BTreeSet<VertId>, bones: &BTreeSet<BoneId>, mode: AutoWeights) -> Result<()> {
    auto_weights_in(mesh, skeleton, verts, bones, mode, Mat4::IDENTITY)
}

/// [`auto_weights`] for a mesh whose local space maps into the skeleton's by `to_skel`.
pub fn auto_weights_in(
    mesh: &mut Mesh,
    skeleton: &Skeleton,
    verts: &BTreeSet<VertId>,
    bones: &BTreeSet<BoneId>,
    mode: AutoWeights,
    to_skel: Mat4,
) -> Result<()> {
    if bones.is_empty() {
        return invalid("no bones to weight to");
    }
    for v in verts {
        let p = to_skel.transform_point3(mesh.pos(*v));
        let mut d: Vec<(BoneId, f32)> =
            bones.iter().map(|b| (*b, segment_distance(p, skeleton.bones[b].head, skeleton.bones[b].tail))).collect();
        d.sort_by(|a, b| a.1.total_cmp(&b.1));
        let list = match mode {
            AutoWeights::Rigid => vec![(d[0].0, 1.0)],
            AutoWeights::Smooth { falloff } => {
                let near = d[0].1.max(1e-4);
                let cut = near * (1.0 + falloff.max(0.0));
                let mut l: Vec<(BoneId, f32)> =
                    d.iter().filter(|(_, dist)| *dist <= cut).take(MAX_INFLUENCES).map(|(b, dist)| (*b, 1.0 / dist.max(1e-4).powi(2))).collect();
                let s: f32 = l.iter().map(|(_, w)| w).sum();
                l.iter_mut().for_each(|(_, w)| *w /= s);
                l
            }
        };
        mesh.weights.insert(*v, list);
    }
    Ok(())
}

// ---- morph targets --------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Morph {
    pub name: String,
    /// Sparse per-vertex offsets from the base shape.
    pub offsets: BTreeMap<VertId, Vec3>,
}

/// An in-progress morph edit: the base positions captured when it started.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphEdit {
    pub name: String,
    pub base: BTreeMap<VertId, Vec3>,
}

pub fn begin_morph_edit(mesh: &Mesh, name: &str) -> MorphEdit {
    MorphEdit { name: name.to_string(), base: mesh.verts.iter().map(|(id, v)| (*id, v.pos)).collect() }
}

/// Finish a morph edit: store current - base as the morph's offsets and restore the base shape.
/// Applying an existing morph's offsets first means editing continues from the stored shape.
pub fn end_morph_edit(mesh: &mut Mesh, edit: &MorphEdit) -> Result<usize> {
    let now: BTreeSet<VertId> = mesh.verts.keys().copied().collect();
    let then: BTreeSet<VertId> = edit.base.keys().copied().collect();
    if now != then {
        return invalid(
            "vertices were added or removed during the morph edit; morphs can only move vertices (undo the topology change or cancel the edit)",
        );
    }
    let mut offsets = BTreeMap::new();
    for (id, base) in &edit.base {
        let d = mesh.pos(*id) - *base;
        if d.length() > 1e-6 {
            offsets.insert(*id, d);
        }
        mesh.verts.get_mut(id).unwrap().pos = *base;
    }
    let n = offsets.len();
    match mesh.morphs.iter_mut().find(|m| m.name == edit.name) {
        Some(m) => m.offsets = offsets,
        None => mesh.morphs.push(Morph { name: edit.name.clone(), offsets }),
    }
    Ok(n)
}

pub fn cancel_morph_edit(mesh: &mut Mesh, edit: &MorphEdit) {
    for (id, base) in &edit.base {
        if let Some(v) = mesh.verts.get_mut(id) {
            v.pos = *base;
        }
    }
}

/// Morph from the difference between two meshes with the same vertex ids
/// (e.g. a duplicate the agent reshaped).
pub fn morph_from_mesh(base: &Mesh, shaped: &Mesh, name: &str) -> Result<Morph> {
    if base.verts.keys().ne(shaped.verts.keys()) {
        return invalid("the shapes have different vertices (duplicate the object, only move vertices, then use it)");
    }
    let offsets = base
        .verts
        .iter()
        .filter_map(|(id, v)| {
            let d = shaped.verts[id].pos - v.pos;
            (d.length() > 1e-6).then_some((*id, d))
        })
        .collect();
    Ok(Morph { name: name.to_string(), offsets })
}

// ---- clips ----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    #[default]
    Linear,
    /// Hold each key until the next one.
    Step,
}

/// What a channel animates.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Target {
    BoneTranslation { object: ObjectId, bone: BoneId },
    BoneRotation { object: ObjectId, bone: BoneId },
    BoneScale { object: ObjectId, bone: BoneId },
    ObjectTranslation { object: ObjectId },
    ObjectRotation { object: ObjectId },
    ObjectScale { object: ObjectId },
    MorphWeight { object: ObjectId, morph: String },
}

impl Target {
    pub fn object(&self) -> ObjectId {
        match self {
            Target::BoneTranslation { object, .. }
            | Target::BoneRotation { object, .. }
            | Target::BoneScale { object, .. }
            | Target::ObjectTranslation { object }
            | Target::ObjectRotation { object }
            | Target::ObjectScale { object }
            | Target::MorphWeight { object, .. } => *object,
        }
    }
    fn is_rotation(&self) -> bool {
        matches!(self, Target::BoneRotation { .. } | Target::ObjectRotation { .. })
    }
}

/// A key's value: xyz for translation/scale, xyzw quaternion for rotation, x for morph weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Key {
    pub time: f32,
    pub value: Vec4,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    pub target: Target,
    #[serde(default)]
    pub interpolation: Interpolation,
    /// Sorted by time.
    pub keys: Vec<Key>,
}

impl Channel {
    pub fn set_key(&mut self, time: f32, value: Vec4) {
        match self.keys.iter().position(|k| (k.time - time).abs() < 1e-5) {
            Some(i) => self.keys[i].value = value,
            None => {
                let i = self.keys.partition_point(|k| k.time < time);
                self.keys.insert(i, Key { time, value });
            }
        }
    }

    pub fn sample(&self, t: f32) -> Option<Vec4> {
        let first = self.keys.first()?;
        if t <= first.time {
            return Some(first.value);
        }
        let last = self.keys.last().unwrap();
        if t >= last.time {
            return Some(last.value);
        }
        let i = self.keys.partition_point(|k| k.time <= t);
        let (a, b) = (self.keys[i - 1], self.keys[i]);
        if self.interpolation == Interpolation::Step {
            return Some(a.value);
        }
        let u = (t - a.time) / (b.time - a.time).max(1e-9);
        Some(if self.target.is_rotation() {
            let qa = Quat::from_vec4(a.value);
            let qb = Quat::from_vec4(b.value);
            Vec4::from(qa.slerp(qb, u))
        } else {
            a.value.lerp(b.value, u)
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    pub name: String,
    /// Seconds.
    pub length: f32,
    /// Frames per second used when tools talk in frames.
    #[serde(default = "fps24")]
    pub fps: f32,
    #[serde(default = "yes")]
    pub looping: bool,
    /// Loop region in seconds; playback wraps from `loop_end` back to `loop_start`.
    #[serde(default)]
    pub loop_start: f32,
    #[serde(default)]
    pub loop_end: Option<f32>,
    #[serde(default)]
    pub channels: Vec<Channel>,
}

fn fps24() -> f32 {
    24.0
}
fn yes() -> bool {
    true
}

impl Clip {
    pub fn new(name: &str, length: f32, fps: f32) -> Self {
        Clip { name: name.into(), length, fps, looping: true, loop_start: 0.0, loop_end: None, channels: vec![] }
    }

    pub fn loop_range(&self) -> (f32, f32) {
        let end = self.loop_end.unwrap_or(self.length).clamp(0.0, self.length);
        (self.loop_start.clamp(0.0, end), end)
    }

    /// Map an ever-increasing playback time to clip time, honouring the loop region.
    pub fn playback_time(&self, t: f32) -> f32 {
        let t = t.max(0.0);
        let (ls, le) = self.loop_range();
        if !self.looping || le - ls < 1e-6 || t < le {
            return t.min(self.length);
        }
        ls + (t - le).rem_euclid(le - ls)
    }

    pub fn channel_mut(&mut self, target: Target) -> &mut Channel {
        match self.channels.iter().position(|c| c.target == target) {
            Some(i) => &mut self.channels[i],
            None => {
                self.channels.push(Channel { target, interpolation: Interpolation::Linear, keys: vec![] });
                self.channels.last_mut().unwrap()
            }
        }
    }

    pub fn key_count(&self) -> usize {
        self.channels.iter().map(|c| c.keys.len()).sum()
    }
}

// ---- evaluation -----------------------------------------------------------

/// One clip contributing to a pose.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClipPlay {
    pub clip: ClipId,
    /// Clip-local time in seconds.
    pub time: f32,
    /// Blend weight (0..1).
    pub weight: f32,
}

/// Weighted blend of channel values onto a base value. Below a total weight
/// of 1 the base shows through; above it the contributions are normalized.
fn blend(base: Vec4, contribs: &[(Vec4, f32)], rotation: bool) -> Vec4 {
    let total: f32 = contribs.iter().map(|(_, w)| w.max(0.0)).sum();
    if total <= 1e-6 {
        return base;
    }
    let mix = |vals: &mut dyn Iterator<Item = (Vec4, f32)>| -> Vec4 {
        if rotation {
            let mut acc = Vec4::ZERO;
            let mut reference: Option<Vec4> = None;
            for (v, w) in vals {
                let r = *reference.get_or_insert(v);
                let v = if v.dot(r) < 0.0 { -v } else { v };
                acc += v * w;
            }
            Vec4::from(Quat::from_vec4(acc).normalize())
        } else {
            vals.fold(Vec4::ZERO, |a, (v, w)| a + v * w)
        }
    };
    let n = total.max(1.0);
    let mut it = contribs.iter().map(|(v, w)| (*v, w.max(0.0) / n)).chain(std::iter::once((base, (1.0 - total).max(0.0))));
    mix(&mut it)
}

/// Display copy of the scene with clips applied at the given times and
/// weights, static poses and morph weights applied, meshes deformed (by their
/// own skeleton or the armature they are bound to) and bones moved to their
/// posed positions.
pub fn pose_scene(scene: &Scene, plays: &[ClipPlay]) -> Scene {
    // Gather channel contributions per target.
    let mut contribs: BTreeMap<Target, Vec<(Vec4, f32)>> = BTreeMap::new();
    for p in plays {
        let Some(clip) = scene.clips.get(&p.clip) else { continue };
        if p.weight <= 0.0 {
            continue;
        }
        let t = p.time.clamp(0.0, clip.length);
        for ch in &clip.channels {
            if let Some(v) = ch.sample(t) {
                contribs.entry(ch.target.clone()).or_default().push((v, p.weight));
            }
        }
    }
    let get = |t: Target, base: Vec4| contribs.get(&t).map_or(base, |c| blend(base, c, t.is_rotation()));
    let mut out = scene.clone();
    // Pass 1: object transforms, morph weights and skeleton poses.
    let mut poses: BTreeMap<ObjectId, BTreeMap<BoneId, BonePose>> = BTreeMap::new();
    let mut morphs: BTreeMap<ObjectId, Vec<f32>> = BTreeMap::new();
    for (&id, o) in out.objects.iter_mut() {
        let tr = &mut o.transform;
        tr.translation = get(Target::ObjectTranslation { object: id }, tr.translation.extend(0.0)).truncate();
        tr.rotation = Quat::from_vec4(get(Target::ObjectRotation { object: id }, Vec4::from(tr.rotation))).normalize();
        tr.scale = get(Target::ObjectScale { object: id }, tr.scale.extend(0.0)).truncate();
        let morph_w: Vec<f32> = o
            .mesh
            .morphs
            .iter()
            .map(|m| {
                let base = o.morph_weights.get(&m.name).copied().unwrap_or(0.0);
                get(Target::MorphWeight { object: id, morph: m.name.clone() }, Vec4::new(base, 0.0, 0.0, 0.0)).x
            })
            .collect();
        morphs.insert(id, morph_w);
        if let Some(sk) = &o.skeleton {
            let mut pose = o.pose.clone();
            for &b in sk.bones.keys() {
                let cur = pose.get(&b).copied().unwrap_or_default();
                let p = BonePose {
                    translation: get(Target::BoneTranslation { object: id, bone: b }, cur.translation.extend(0.0)).truncate(),
                    rotation: Quat::from_vec4(get(Target::BoneRotation { object: id, bone: b }, Vec4::from(cur.rotation))).normalize(),
                    scale: get(Target::BoneScale { object: id, bone: b }, cur.scale.extend(0.0)).truncate(),
                };
                if p.is_rest() {
                    pose.remove(&b);
                } else {
                    pose.insert(b, p);
                }
            }
            poses.insert(id, pose);
        }
    }
    // Pass 2: deform meshes with their skeleton source (rest skeletons, posed transforms).
    let ids: Vec<ObjectId> = out.objects.keys().copied().collect();
    for id in ids {
        let src = out.skeleton_source(id);
        let pose = src.and_then(|s| poses.get(&s)).filter(|p| !p.is_empty());
        let o = &out.objects[&id];
        let morph_w = &morphs[&id];
        let skinned = pose.is_some() && !o.mesh.weights.is_empty();
        if morph_w.iter().any(|w| w.abs() > 1e-6) || skinned {
            let skel = src.and_then(|s| out.objects[&s].skeleton.as_ref());
            let empty = BTreeMap::new();
            let mesh = deform_in(&o.mesh, skel, pose.unwrap_or(&empty), morph_w, out.to_skeleton_space(id));
            out.objects.get_mut(&id).unwrap().mesh = Arc::new(mesh);
        }
    }
    // Pass 3: move the bones of posed skeletons in the display copy.
    for (id, pose) in &poses {
        if pose.is_empty() {
            continue;
        }
        let sk = out.objects.get_mut(id).unwrap().skeleton.as_mut().unwrap();
        let m = sk.world_matrices(pose);
        for (b, bone) in sk.bones.iter_mut() {
            let (posed, rest) = m[b];
            let skin = posed * rest.inverse();
            bone.head = skin.transform_point3(bone.head);
            bone.tail = skin.transform_point3(bone.tail);
        }
    }
    out
}

/// Morph then skin a mesh (positions only; ids, faces and attributes are kept).
pub fn deform(mesh: &Mesh, skeleton: Option<&Skeleton>, pose: &BTreeMap<BoneId, BonePose>, morph_weights: &[f32]) -> Mesh {
    deform_in(mesh, skeleton, pose, morph_weights, Mat4::IDENTITY)
}

/// [`deform`] for a mesh whose local space maps into the skeleton's by `to_skel`.
pub fn deform_in(
    mesh: &Mesh,
    skeleton: Option<&Skeleton>,
    pose: &BTreeMap<BoneId, BonePose>,
    morph_weights: &[f32],
    to_skel: Mat4,
) -> Mesh {
    let mut m = mesh.clone();
    for (morph, w) in mesh.morphs.iter().zip(morph_weights) {
        if w.abs() < 1e-6 {
            continue;
        }
        for (v, d) in &morph.offsets {
            if let Some(vert) = m.verts.get_mut(v) {
                vert.pos += *d * *w;
            }
        }
    }
    if let Some(sk) = skeleton {
        let from_skel = to_skel.inverse();
        let mats: HashMap<BoneId, Mat4> = sk
            .world_matrices(pose)
            .into_iter()
            .map(|(b, (posed, rest))| (b, from_skel * posed * rest.inverse() * to_skel))
            .collect();
        for (v, list) in &mesh.weights {
            let Some(vert) = m.verts.get_mut(v) else { continue };
            let total: f32 = list.iter().filter(|(b, _)| mats.contains_key(b)).map(|(_, w)| w).sum();
            if total <= 1e-6 {
                continue;
            }
            let p = vert.pos;
            vert.pos = list.iter().filter_map(|(b, w)| Some(mats.get(b)?.transform_point3(p) * (*w / total))).sum();
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Primitive;

    fn arm() -> (Scene, ObjectId, BoneId, BoneId) {
        let mut s = Scene::default();
        let mesh = Primitive::Cylinder {
            radius_bottom: 0.2,
            radius_top: 0.2,
            height: 2.0,
            segments: 6,
            height_segments: 4,
            caps: true,
        }
        .build()
        .unwrap();
        let id = s.add_object("Arm", mesh);
        let o = s.objects.get_mut(&id).unwrap();
        let mut sk = Skeleton::default();
        let upper = sk.add(Bone { name: "upper".into(), parent: None, head: Vec3::new(0.0, -1.0, 0.0), tail: Vec3::ZERO, roll: 0.0 }).unwrap();
        let lower = sk.add(Bone { name: "lower".into(), parent: Some(upper), head: Vec3::ZERO, tail: Vec3::Y, roll: 0.0 }).unwrap();
        let mesh = Arc::make_mut(&mut o.mesh);
        let all: BTreeSet<VertId> = mesh.verts.keys().copied().collect();
        auto_weights(mesh, &sk, &all, &BTreeSet::from([upper, lower]), AutoWeights::Rigid).unwrap();
        o.skeleton = Some(sk);
        (s, id, upper, lower)
    }

    #[test]
    fn rest_pose_does_not_move_vertices() {
        let (s, id, ..) = arm();
        let posed = pose_scene(&s, &[]);
        assert!(Arc::ptr_eq(&posed.objects[&id].mesh, &s.objects[&id].mesh));
    }

    #[test]
    fn bending_the_lower_bone_moves_only_its_vertices() {
        let (mut s, id, upper, lower) = arm();
        let o = s.objects.get_mut(&id).unwrap();
        o.pose.insert(lower, BonePose { rotation: Quat::from_rotation_z(90f32.to_radians()), ..Default::default() });
        let posed = pose_scene(&s, &[]);
        let (before, after) = (&s.objects[&id].mesh, &posed.objects[&id].mesh);
        for (v, list) in &before.weights {
            let moved = before.pos(*v).distance(after.pos(*v)) > 1e-4;
            let on_lower = list[0].0 == lower;
            let at_joint = before.pos(*v).y.abs() < 1e-4;
            if !at_joint {
                assert_eq!(moved, on_lower, "v{v} {:?}", before.pos(*v));
            }
            if list[0].0 == upper {
                assert!(!moved);
            }
        }
        // The tip of the lower bone swings from +Y to -X.
        let tip = before.verts.iter().find(|(_, p)| p.pos.y > 0.99).map(|(v, _)| *v).unwrap();
        assert!(after.pos(tip).x < -0.7, "{:?}", after.pos(tip));
        // The posed copy's bone moved too.
        let b = &posed.objects[&id].skeleton.as_ref().unwrap().bones[&lower];
        assert!((b.tail - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-4, "{:?}", b.tail);
    }

    #[test]
    fn clip_blend_and_loop() {
        let (mut s, id, _, lower) = arm();
        let mut c = Clip::new("wave", 2.0, 24.0);
        let ch = c.channel_mut(Target::BoneRotation { object: id, bone: lower });
        ch.set_key(0.0, Vec4::from(Quat::IDENTITY));
        ch.set_key(1.0, Vec4::from(Quat::from_rotation_z(90f32.to_radians())));
        c.loop_start = 0.5;
        c.loop_end = Some(1.5);
        let cid = s.add_clip(c);
        let tip = |s: &Scene, plays: &[ClipPlay]| {
            let p = pose_scene(s, plays);
            p.objects[&id].skeleton.as_ref().unwrap().bones[&lower].tail
        };
        let full = tip(&s, &[ClipPlay { clip: cid, time: 1.0, weight: 1.0 }]);
        let halfw = tip(&s, &[ClipPlay { clip: cid, time: 1.0, weight: 0.5 }]);
        assert!((full - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-4);
        // Half weight = halfway rotation (45 degrees).
        assert!((halfw - Vec3::new(-(0.5f32.sqrt()), 0.5f32.sqrt(), 0.0)).length() < 1e-3, "{halfw:?}");
        let clip = &s.clips[&cid];
        assert!((clip.playback_time(0.25) - 0.25).abs() < 1e-6);
        assert!((clip.playback_time(1.75) - 0.75).abs() < 1e-6);
        assert!((clip.playback_time(3.6) - 0.6).abs() < 1e-5);
    }

    /// A mesh bound to a separate armature deforms exactly like the same mesh
    /// with its own skeleton, also when the armature is moved and rotated and
    /// the motion comes from a clip on the armature.
    #[test]
    fn bound_mesh_matches_self_bound() {
        let (reference, rid, _, lower) = arm();
        let mut s = Scene::default();
        let o = &reference.objects[&rid];
        // Armature: skeleton only, placed somewhere else in the world.
        let arm_id = s.add_object("Rig", Mesh::new());
        let place = crate::scene::Transform {
            translation: Vec3::new(3.0, 1.0, -2.0),
            rotation: Quat::from_rotation_y(0.7),
            scale: Vec3::ONE,
        };
        {
            let a = s.objects.get_mut(&arm_id).unwrap();
            a.skeleton = o.skeleton.clone();
            a.transform = place;
        }
        // Skin: same geometry and weights, parented to the armature at identity.
        let skin_id = s.add_object("Skin", (*o.mesh).clone());
        {
            let k = s.objects.get_mut(&skin_id).unwrap();
            k.skin = Some(crate::scene::SkinBinding { armature: arm_id });
        }
        s.set_parent(skin_id, Some(arm_id)).unwrap();
        let mut c = Clip::new("bend", 1.0, 24.0);
        c.channel_mut(Target::BoneRotation { object: arm_id, bone: lower })
            .set_key(0.0, Vec4::from(Quat::from_rotation_z(70f32.to_radians())));
        let cid = s.add_clip(c);
        let mut r = reference.clone();
        r.objects.get_mut(&rid).unwrap().pose.insert(lower, BonePose { rotation: Quat::from_rotation_z(70f32.to_radians()), ..Default::default() });

        let posed = pose_scene(&s, &[ClipPlay { clip: cid, time: 0.0, weight: 1.0 }]);
        let expect = pose_scene(&r, &[]);
        let world = posed.world_matrix(skin_id);
        let pm = &posed.objects[&skin_id].mesh;
        let em = &expect.objects[&rid].mesh;
        let mut moved = 0;
        for (v, e) in &em.verts {
            let got = world.transform_point3(pm.pos(*v));
            let want = place.matrix().transform_point3(e.pos);
            assert!(got.distance(want) < 1e-4, "v{v}: {got} vs {want}");
            if e.pos.distance(o.mesh.pos(*v)) > 1e-3 {
                moved += 1;
            }
        }
        assert!(moved > 5, "the pose should actually bend the arm");
        // Unbinding happens when the armature is deleted.
        s.remove_object(arm_id).unwrap();
        assert!(s.objects[&skin_id].skin.is_none());
    }

    #[test]
    fn morph_edit_roundtrip() {
        let mut m = crate::primitives::cube(Vec3::ONE);
        let edit = begin_morph_edit(&m, "squash");
        let top: BTreeSet<VertId> = m.verts.iter().filter(|(_, v)| v.pos.y > 0.0).map(|(i, _)| *i).collect();
        crate::ops::transform::translate(&mut m, &top, Vec3::new(0.0, -0.5, 0.0)).unwrap();
        assert_eq!(end_morph_edit(&mut m, &edit).unwrap(), 4);
        assert!(m.verts.values().all(|v| v.pos.y.abs() > 0.49)); // base restored
        let half = deform(&m, None, &BTreeMap::new(), &[0.5]);
        let ys: BTreeSet<i32> = half.verts.values().map(|v| (v.pos.y * 100.0).round() as i32).collect();
        assert_eq!(ys, BTreeSet::from([-50, 25]));
    }

    #[test]
    fn smooth_weights_sum_to_one_and_respect_limit() {
        let (mut s, id, ..) = arm();
        let o = s.objects.get_mut(&id).unwrap();
        let sk = o.skeleton.clone().unwrap();
        let mesh = Arc::make_mut(&mut o.mesh);
        let all: BTreeSet<VertId> = mesh.verts.keys().copied().collect();
        let bones: BTreeSet<BoneId> = sk.bones.keys().copied().collect();
        auto_weights(mesh, &sk, &all, &bones, AutoWeights::Smooth { falloff: 1.0 }).unwrap();
        normalize_weights(mesh);
        for list in mesh.weights.values() {
            assert!(list.len() <= MAX_INFLUENCES);
            assert!((list.iter().map(|(_, w)| w).sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert!(mesh.weights.values().any(|l| l.len() == 2), "joint vertices should blend");
    }
}
