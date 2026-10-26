use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::ok;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def("undo", "Undo the last `steps` edits (default 1).", false, undo),
        def("redo", "Redo `steps` undone edits (default 1).", false, redo),
        def("history", "List recent undo/redo steps and named checkpoints.", true, history),
        def("checkpoint", "Save the current scene state under a name, to restore later (e.g. before risky edits).", false, checkpoint),
        def("restore", "Restore a named checkpoint (itself undoable).", false, restore),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct Steps {
    steps: Option<usize>,
}

fn undo(ctx: &mut Ctx, a: Steps) -> Result<Out> {
    let l = ctx.doc.undo(a.steps.unwrap_or(1));
    Ok(Out::json(json!({"undone": l})))
}

fn redo(ctx: &mut Ctx, a: Steps) -> Result<Out> {
    let l = ctx.doc.redo(a.steps.unwrap_or(1));
    Ok(Out::json(json!({"redone": l})))
}

#[derive(Deserialize, JsonSchema)]
struct Empty {}

fn history(ctx: &mut Ctx, _: Empty) -> Result<Out> {
    Ok(Out::json(ctx.doc.history()))
}

#[derive(Deserialize, JsonSchema)]
struct Name {
    name: String,
}

fn checkpoint(ctx: &mut Ctx, a: Name) -> Result<Out> {
    ctx.doc.checkpoint(&a.name);
    Ok(ok(format!("checkpoint '{}' saved", a.name)))
}

fn restore(ctx: &mut Ctx, a: Name) -> Result<Out> {
    ctx.doc.restore(&a.name)?;
    Ok(ok(format!("restored '{}'", a.name)))
}
