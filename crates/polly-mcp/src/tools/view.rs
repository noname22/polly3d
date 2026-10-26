use anyhow::Result;
use polly_core::scene::ObjectId;

use crate::registry::{Ctx, Out, ToolDef};

pub fn tools() -> Vec<ToolDef> {
    vec![]
}

/// Append a preview render when auto-preview is on.
pub fn maybe_preview(_ctx: &mut Ctx, _object: ObjectId, out: Out) -> Result<Out> {
    Ok(out)
}
