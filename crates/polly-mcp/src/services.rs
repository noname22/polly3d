//! Process-wide services shared by tool handlers (renderer, settings).

use std::sync::Mutex;

#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// If set, mutating mesh tools append a preview render of this size.
    pub auto_preview: Option<u32>,
    /// Last project path used by save/load.
    pub project_path: Option<std::path::PathBuf>,
}

#[derive(Default)]
pub struct Services {
    pub settings: Mutex<Settings>,
}
