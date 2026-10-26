pub mod common;
pub mod edit;
pub mod history;
pub mod io;
pub mod paint;
pub mod query;
pub mod scene;
pub mod view;

use crate::registry::Registry;

pub fn registry() -> Registry {
    let mut tools = Vec::new();
    tools.extend(scene::tools());
    tools.extend(query::tools());
    tools.extend(edit::tools());
    tools.extend(history::tools());
    tools.extend(view::tools());
    tools.extend(paint::tools());
    tools.extend(io::tools());
    Registry { tools }
}
