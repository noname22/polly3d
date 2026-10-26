//! Document = scene + undo history. Every mutating tool call goes through
//! [`Document::edit`], which snapshots the scene (cheap: meshes and texture
//! pixels are `Arc`s) and rolls back on error, so failed edits never leave
//! the scene half-modified.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::ids;
use crate::mesh::Mesh;
use crate::scene::Scene;

const MAX_UNDO: usize = 200;

#[derive(Default)]
pub struct Document {
    pub scene: Scene,
    undo: Vec<(String, Scene)>,
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
                self.undo.push((label.to_string(), before));
                if self.undo.len() > MAX_UNDO {
                    self.undo.remove(0);
                }
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

    pub fn undo(&mut self, steps: usize) -> Vec<String> {
        let mut labels = Vec::new();
        for _ in 0..steps.max(1) {
            let Some((label, scene)) = self.undo.pop() else { break };
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
            self.undo.push((label.clone(), now));
            labels.push(label);
        }
        self.version += 1;
        labels
    }

    pub fn history(&self) -> Value {
        json!({
            "undo": self.undo.iter().rev().take(30).map(|(l, _)| l).collect::<Vec<_>>(),
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
