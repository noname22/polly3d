//! Tool registry: each tool is a typed args struct (JSON schema via
//! schemars) plus a synchronous handler operating on the document.

use std::sync::Arc;

use anyhow::Result;
use base64::Engine;
use polly_core::document::Document;
use rmcp::model::{ContentBlock, Tool, ToolAnnotations};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::services::Services;

/// Tool output: a JSON report plus optional PNG images.
#[derive(Default)]
pub struct Out {
    pub json: Value,
    pub images: Vec<Vec<u8>>,
}

impl Out {
    pub fn json(v: Value) -> Self {
        Out { json: v, images: vec![] }
    }
    pub fn with_image(mut self, png: Vec<u8>) -> Self {
        self.images.push(png);
        self
    }
    pub fn into_content(self) -> Vec<ContentBlock> {
        let mut c = vec![];
        if !self.json.is_null() {
            let text = match &self.json {
                Value::String(s) => s.clone(),
                v => serde_json::to_string_pretty(v).unwrap_or_default(),
            };
            c.push(ContentBlock::text(text));
        }
        for png in self.images {
            c.push(ContentBlock::image(base64::engine::general_purpose::STANDARD.encode(png), "image/png"));
        }
        c
    }
}

pub struct Ctx<'a> {
    pub doc: &'a mut Document,
    pub services: &'a Services,
}

type Handler = Box<dyn Fn(&mut Ctx, Value) -> Result<Out> + Send + Sync>;

pub struct ToolDef {
    pub tool: Tool,
    pub call: Handler,
}

/// Define a tool. `read_only` tools never modify the document.
pub fn def<A>(name: &'static str, description: &'static str, read_only: bool, f: fn(&mut Ctx, A) -> Result<Out>) -> ToolDef
where
    A: DeserializeOwned + JsonSchema + 'static,
{
    let schema = rmcp::handler::server::common::schema_for_input::<A>()
        .unwrap_or_else(|e| panic!("schema for tool {name}: {e}"));
    let mut sv = serde_json::Value::Object((*schema).clone());
    crate::colors::widen_schema(&mut sv);
    let schema = Arc::new(sv.as_object().unwrap().clone());
    let mut annotations = ToolAnnotations::new();
    annotations.read_only_hint = Some(read_only);
    annotations.destructive_hint = Some(false);
    let tool = Tool::new(name, description, schema).with_annotations(annotations);
    ToolDef {
        tool,
        call: Box::new(move |ctx, mut args| {
            let palette = crate::tools::paint::args_palette(ctx, &args);
            crate::colors::normalize(&mut args, palette.as_deref())?;
            let args: A = serde_json::from_value(args)
                .map_err(|e| anyhow::anyhow!("invalid arguments for {name}: {e}"))?;
            f(ctx, args)
        }),
    }
}

pub struct Registry {
    pub tools: Vec<ToolDef>,
}

impl Registry {
    pub fn get(&self, name: &str) -> Option<&ToolDef> {
        self.tools.iter().find(|t| t.tool.name == name)
    }
    pub fn list(&self) -> Vec<Tool> {
        self.tools.iter().map(|t| t.tool.clone()).collect()
    }
}

pub type SharedRegistry = Arc<Registry>;
