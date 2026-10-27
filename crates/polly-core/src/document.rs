//! Document = scene + undo history. Every mutating tool call goes through
//! [`Document::edit`], which snapshots the scene (cheap: meshes and texture
//! pixels are `Arc`s) and rolls back on error, so failed edits never leave
//! the scene half-modified.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::ids;
use crate::mesh::Mesh;
use crate::scene::{Scene, TextureId};

const MAX_UNDO: usize = 200;
/// Textures at least this many pixels are kept in history as changed tiles only.
const PATCH_MIN_PIXELS: usize = 128 * 128;
const TILE: u32 = 64;

/// The pixels a step changed in one large texture: the tiles as they were
/// before it. Undo applies them to the texture as it is after the step.
struct TexPatch {
    texture: TextureId,
    /// (tile x, tile y, RGBA rows of the tile before the step).
    tiles: Vec<(u32, u32, Vec<u8>)>,
    /// Hash of the pixels after the step, to check the patch still applies.
    after: u64,
}

struct Entry {
    label: String,
    /// The scene before the step; patched textures hold no pixels here.
    scene: Scene,
    patches: Vec<TexPatch>,
}

fn pixel_hash(p: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ p.len() as u64;
    let mut chunks = p.chunks_exact(8);
    for c in &mut chunks {
        h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
    }
    for b in chunks.remainder() {
        h = (h ^ *b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Replace large textures that changed between `before` and `after` with
/// tile patches (in `before`), so history doesn't hold whole copies.
fn compress(before: &mut Scene, after: &Scene) -> Vec<TexPatch> {
    let mut out = Vec::new();
    for (tid, bt) in before.textures.iter_mut() {
        let Some(at) = after.textures.get(tid) else { continue };
        let n = (bt.width * bt.height) as usize;
        if Arc::ptr_eq(&bt.pixels, &at.pixels)
            || n < PATCH_MIN_PIXELS
            || bt.width != at.width
            || bt.height != at.height
            || bt.pixels.len() != n * 4
            || at.pixels.len() != n * 4
        {
            continue;
        }
        let (w, h) = (bt.width, bt.height);
        let mut tiles = Vec::new();
        for ty in 0..h.div_ceil(TILE) {
            for tx in 0..w.div_ceil(TILE) {
                let (x0, y0) = (tx * TILE, ty * TILE);
                let (x1, y1) = ((x0 + TILE).min(w), (y0 + TILE).min(h));
                let row = |y: u32| -> std::ops::Range<usize> { ((y * w + x0) * 4) as usize..((y * w + x1) * 4) as usize };
                if (y0..y1).any(|y| bt.pixels[row(y)] != at.pixels[row(y)]) {
                    let mut t = Vec::with_capacity(((x1 - x0) * (y1 - y0) * 4) as usize);
                    for y in y0..y1 {
                        t.extend_from_slice(&bt.pixels[row(y)]);
                    }
                    tiles.push((tx, ty, t));
                }
            }
        }
        out.push(TexPatch { texture: *tid, tiles, after: pixel_hash(&at.pixels) });
        bt.pixels = Arc::new(Vec::new());
    }
    out
}

#[derive(Default)]
pub struct Document {
    pub scene: Scene,
    undo: VecDeque<Entry>,
    redo: Vec<(String, Scene)>,
    checkpoints: BTreeMap<String, Scene>,
    /// Bumped on every change; the viewer polls this.
    pub version: u64,
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a mutation atomically, recording it in history.
    pub fn edit<T>(&mut self, label: &str, f: impl FnOnce(&mut Scene) -> Result<T>) -> Result<T> {
        let before = self.scene.clone();
        match f(&mut self.scene) {
            Ok(v) => {
                self.push_undo(label.to_string(), before);
                self.redo.clear();
                self.version += 1;
                Ok(v)
            }
            Err(e) => {
                self.scene = before;
                Err(e)
            }
        }
    }

    /// Change state that should not be recorded in history (e.g. selection).
    pub fn touch(&mut self) {
        self.version += 1;
    }

    fn push_undo(&mut self, label: String, mut before: Scene) {
        let patches = compress(&mut before, &self.scene);
        self.undo.push_back(Entry { label, scene: before, patches });
        while self.undo.len() > MAX_UNDO {
            self.undo.pop_front();
        }
    }

    /// Number of steps that can be undone.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Bytes of texture pixels held only by the undo history (patch tiles).
    pub fn undo_patch_bytes(&self) -> usize {
        self.undo.iter().flat_map(|e| &e.patches).flat_map(|p| &p.tiles).map(|t| t.2.len()).sum()
    }

    pub fn undo(&mut self, steps: usize) -> Vec<String> {
        let mut labels = Vec::new();
        for _ in 0..steps.max(1) {
            let Some(Entry { label, mut scene, patches }) = self.undo.pop_back() else { break };
            for p in patches {
                let (Some(cur), Some(t)) = (self.scene.textures.get(&p.texture), scene.textures.get_mut(&p.texture)) else { continue };
                let mut px = cur.pixels.to_vec();
                if pixel_hash(&px) != p.after || px.len() != (t.width * t.height * 4) as usize {
                    // History and scene disagree (shouldn't happen): keep today's pixels.
                    t.pixels = cur.pixels.clone();
                    continue;
                }
                let w = t.width;
                for (tx, ty, tile) in &p.tiles {
                    let (x0, y0) = (tx * TILE, ty * TILE);
                    let tw = ((x0 + TILE).min(w) - x0) as usize * 4;
                    for (r, y) in (y0..).zip(tile.chunks_exact(tw)) {
                        let start = ((r * w + x0) * 4) as usize;
                        px[start..start + tw].copy_from_slice(y);
                    }
                }
                t.pixels = Arc::new(px);
            }
            let now = std::mem::replace(&mut self.scene, scene);
            self.redo.push((label.clone(), now));
            labels.push(label);
        }
        self.version += 1;
        labels
    }

    pub fn redo(&mut self, steps: usize) -> Vec<String> {
        let mut labels = Vec::new();
        for _ in 0..steps.max(1) {
            let Some((label, scene)) = self.redo.pop() else { break };
            let now = std::mem::replace(&mut self.scene, scene);
            self.push_undo(label.clone(), now);
            labels.push(label);
        }
        self.version += 1;
        labels
    }

    pub fn history(&self) -> Value {
        json!({
            "undo": self.undo.iter().rev().take(30).map(|e| &e.label).collect::<Vec<_>>(),
            "redo": self.redo.iter().rev().map(|(l, _)| l).collect::<Vec<_>>(),
            "checkpoints": self.checkpoints.keys().collect::<Vec<_>>(),
        })
    }

    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.insert(name.to_string(), self.scene.clone());
    }

    pub fn restore(&mut self, name: &str) -> Result<()> {
        let s = self
            .checkpoints
            .get(name)
            .cloned()
            .ok_or_else(|| Error::NotFound { kind: "checkpoint", id: name.to_string() })?;
        self.edit(&format!("restore '{name}'"), |scene| {
            *scene = s;
            Ok(())
        })
    }

    /// Replace the whole scene (e.g. after loading), clearing history.
    pub fn replace(&mut self, scene: Scene) {
        self.scene = scene;
        self.undo.clear();
        self.redo.clear();
        self.version += 1;
    }
}

/// What changed in a mesh, for tool results.
pub fn mesh_diff(before: &Mesh, after: &Mesh) -> Value {
    let bv: BTreeSet<_> = before.verts.keys().collect();
    let av: BTreeSet<_> = after.verts.keys().collect();
    let bf: BTreeSet<_> = before.faces.keys().collect();
    let af: BTreeSet<_> = after.faces.keys().collect();
    let modified = ids::compress(
        "f",
        after.faces.iter().filter(|(id, f)| before.faces.get(id).is_some_and(|b| b.verts != f.verts)).map(|(id, _)| *id),
    );
    let mut d = serde_json::Map::new();
    let added_v = ids::compress("v", av.difference(&bv).map(|v| **v));
    let removed_v = ids::compress("v", bv.difference(&av).map(|v| **v));
    let added_f = ids::compress("f", af.difference(&bf).map(|f| **f));
    let removed_f = ids::compress("f", bf.difference(&af).map(|f| **f));
    if !added_v.is_empty() {
        d.insert("added_verts".into(), json!(added_v));
    }
    if !removed_v.is_empty() {
        d.insert("removed_verts".into(), json!(removed_v));
    }
    if !added_f.is_empty() {
        d.insert("added_faces".into(), json!(added_f));
    }
    if !removed_f.is_empty() {
        d.insert("removed_faces".into(), json!(removed_f));
    }
    if !modified.is_empty() {
        d.insert("modified_faces".into(), json!(modified));
    }
    d.insert(
        "counts".into(),
        json!({"verts": after.verts.len(), "faces": after.faces.len(), "tris": after.triangle_count()}),
    );
    Value::Object(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;

    #[test]
    fn large_texture_history_keeps_only_changed_tiles() {
        let mut d = Document::new();
        let tid = d.edit("tex", |s| Ok(s.add_texture(crate::scene::Texture::new("big", 500, 300, [10, 20, 30, 255])))).unwrap();
        let snap = |d: &Document| d.scene.textures[&tid].pixels.clone();
        let mut states = vec![snap(&d)];
        for i in 0..3u32 {
            d.edit("paint", |s| {
                let t = s.texture_mut(tid)?;
                let w = t.width;
                let px = t.pixels_mut();
                for y in 100 + i * 70..110 + i * 70 {
                    for x in 450..499 {
                        let k = ((y * w + x) * 4) as usize;
                        px[k..k + 4].copy_from_slice(&[255, i as u8, 0, 255]);
                    }
                }
                Ok(())
            })
            .unwrap();
            states.push(snap(&d));
        }
        // Three steps, each touching one or two 64x64 tiles: far less than three full copies.
        let full = 500 * 300 * 4;
        assert!(d.undo_patch_bytes() < full / 4, "{} bytes", d.undo_patch_bytes());
        for i in (0..3).rev() {
            d.undo(1);
            assert_eq!(*snap(&d), *states[i], "undo to state {i}");
        }
        d.redo(2);
        assert_eq!(*snap(&d), *states[2]);
        d.undo(1);
        assert_eq!(*snap(&d), *states[1]);
    }

    #[test]
    fn failed_edit_rolls_back_and_undo_works() {
        let mut d = Document::new();
        d.edit("add", |s| Ok(s.add_object("Cube", cube(glam::Vec3::ONE)))).unwrap();
        let r: Result<()> = d.edit("bad", |s| {
            s.objects.clear();
            crate::error::invalid("nope")
        });
        assert!(r.is_err());
        assert_eq!(d.scene.objects.len(), 1);
        d.undo(1);
        assert_eq!(d.scene.objects.len(), 0);
        d.redo(1);
        assert_eq!(d.scene.objects.len(), 1);
    }
}
