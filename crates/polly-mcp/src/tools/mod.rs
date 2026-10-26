pub mod anim;
pub mod common;
pub mod edit;
pub mod history;
pub mod io;
pub mod paint;
pub mod query;
pub mod reference;
pub mod scene;
pub mod scenes;
pub mod uv;
pub mod variants;
pub mod view;

use crate::registry::Registry;

pub fn registry() -> Registry {
    let mut tools = Vec::new();
    tools.extend(scenes::tools());
    tools.extend(scene::tools());
    tools.extend(query::tools());
    tools.extend(edit::tools());
    tools.extend(history::tools());
    tools.extend(view::tools());
    tools.extend(anim::tools());
    tools.extend(variants::tools());
    tools.extend(reference::tools());
    tools.extend(uv::tools());
    tools.extend(paint::tools());
    tools.extend(io::tools());
    Registry { tools }
}
