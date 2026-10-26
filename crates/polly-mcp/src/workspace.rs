//! Named scenes. Each scene has its own document, undo history, selection and
//! last-render camera, so unrelated models (e.g. one per subagent) never
//! interfere. Calls lock only their own scene, so scenes work in parallel.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, TryLockError};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use polly_core::document::Document;
use polly_core::scene::Scene;

use crate::services::LastView;

pub const MAIN: &str = "main";

#[derive(Default)]
pub struct SceneSlot {
    pub doc: Document,
    /// Camera(s) of this scene's most recent render, for `pick` / `paint_stroke`.
    pub last_view: Option<LastView>,
    /// Last project path used by save/load in this scene.
    pub project_path: Option<PathBuf>,
}

pub type SharedSlot = Arc<Mutex<SceneSlot>>;

pub struct Workspace {
    scenes: RwLock<BTreeMap<String, SharedSlot>>,
    /// Bumped when scenes are created or deleted (the viewer watches it).
    list_version: AtomicU64,
}

impl Default for Workspace {
    fn default() -> Self {
        Self::with_main(SceneSlot::default())
    }
}

impl Workspace {
    pub fn with_main(main: SceneSlot) -> Self {
        let scenes = BTreeMap::from([(MAIN.to_string(), Arc::new(Mutex::new(main)))]);
        Workspace { scenes: RwLock::new(scenes), list_version: AtomicU64::new(0) }
    }

    pub fn names(&self) -> Vec<String> {
        self.scenes.read().unwrap().keys().cloned().collect()
    }

    pub fn list_version(&self) -> u64 {
        self.list_version.load(Ordering::Relaxed)
    }

    pub fn get(&self, name: &str) -> Result<SharedSlot> {
        let scenes = self.scenes.read().unwrap();
        match scenes.get(name) {
            Some(s) => Ok(s.clone()),
            None => bail!(
                "unknown scene '{name}'; existing: {} (create it with create_scene)",
                scenes.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        }
    }

    /// All scenes, in name order.
    pub fn all(&self) -> Vec<(String, SharedSlot)> {
        self.scenes.read().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// Create a scene; without a name, picks the first free `scene-N`.
    pub fn create(&self, name: Option<&str>) -> Result<String> {
        let mut scenes = self.scenes.write().unwrap();
        let name = match name.map(str::trim) {
            Some("") => bail!("scene name must not be empty"),
            Some(n) if scenes.contains_key(n) => bail!("scene '{n}' already exists"),
            Some(n) => n.to_string(),
            None => (1..).map(|i| format!("scene-{i}")).find(|n| !scenes.contains_key(n)).unwrap(),
        };
        scenes.insert(name.clone(), Arc::default());
        self.list_version.fetch_add(1, Ordering::Relaxed);
        Ok(name)
    }

    pub fn remove(&self, name: &str) -> Result<()> {
        if name == MAIN {
            bail!("the '{MAIN}' scene cannot be deleted (use clear_scene to empty it)");
        }
        if self.scenes.write().unwrap().remove(name).is_none() {
            bail!("unknown scene '{name}'");
        }
        self.list_version.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// A copy of one scene's contents, taken without holding its lock.
pub struct Snapshot {
    pub name: String,
    pub scene: Scene,
    pub version: u64,
}

impl Workspace {
    /// Copy scenes for read-only use (cheap: meshes and textures are `Arc`s).
    /// `current` is the scene the caller already holds locked, passed in
    /// directly. Other scenes are only `try_lock`ed (with a short retry) so
    /// two calls reading each other's scenes can never deadlock; a scene that
    /// stays busy is reported in the second return value.
    pub fn snapshots(&self, names: Option<&[String]>, current: Option<(&str, &Document)>, wait: Duration) -> Result<(Vec<Snapshot>, Vec<String>)> {
        let list: Vec<(String, SharedSlot)> = match names {
            None => self.all(),
            Some(ns) => ns.iter().map(|n| Ok((n.clone(), self.get(n)?))).collect::<Result<_>>()?,
        };
        let deadline = Instant::now() + wait;
        let mut out = Vec::new();
        let mut busy = Vec::new();
        for (name, slot) in list {
            if let Some((cur, doc)) = current.filter(|(c, _)| *c == name) {
                out.push(Snapshot { name: cur.to_string(), scene: doc.scene.clone(), version: doc.version });
                continue;
            }
            loop {
                match slot.try_lock() {
                    Ok(g) => {
                        out.push(Snapshot { name, scene: g.doc.scene.clone(), version: g.doc.version });
                        break;
                    }
                    Err(TryLockError::Poisoned(p)) => {
                        let g = p.into_inner();
                        out.push(Snapshot { name, scene: g.doc.scene.clone(), version: g.doc.version });
                        break;
                    }
                    Err(TryLockError::WouldBlock) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
                    Err(TryLockError::WouldBlock) => {
                        busy.push(name);
                        break;
                    }
                }
            }
        }
        Ok((out, busy))
    }
}
