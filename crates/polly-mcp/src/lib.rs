//! MCP server exposing the Polly3D modeller as tools.

pub mod colors;
pub mod registry;
pub mod services;
pub mod tools;

use std::sync::{Arc, Mutex};

use polly_core::document::Document;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer, ServerHandler};

use crate::registry::{Ctx, Registry};
use crate::services::Services;

pub const INSTRUCTIONS: &str = r#"Polly3D is a low-poly 3-D modeller you drive entirely through these tools.

Conventions
- Y is up, right-handed; "front" looks from +Z toward -Z, "right" from +X. Units are arbitrary (1 = 1 m is a good habit).
- Objects are addressed by name (or "o<id>"); tools default to the active object (the last one created/selected).
- Mesh elements have stable ids: vertices "v12", faces "f3", edges "e3-9" (the two vertex ids). Ids never get reused.
- Most edit tools take `elements` (ids, "all", or "selection"); when omitted they act on the object's current selection. Tools that create geometry (extrude, inset, bevel, loop_cut, subdivide, ...) set the selection to the result, so you can chain them.
- Edits are atomic and undoable (undo/redo, checkpoint/restore). Each edit reports added/removed ids and topology warnings.
- UVs follow glTF: (0,0) is the top-left of the texture, v grows downward. Pixel coordinates are (0,0) at top-left.

Workflow that works well
1. Block out with primitives (add_primitive), keep segment counts low.
2. Look: render (view persp/front/right/top, overlays face_ids) or render_contact_sheet. Check ids before editing.
3. Refine with select (by normal / box / loops) + extrude, inset, bevel, loop_cut, move/scale. Render again after each meaningful change.
4. UV: mark seams (or auto_seams), unwrap with a small texture size (e.g. 64x64) and pixel snapping; inspect with render_uv_layout.
5. Paint: create_texture (optionally palette-locked), assign it through a material, fill_faces / draw / paint_stroke, check render.
6. save_project, export_glb.
"#;

#[derive(Clone)]
pub struct PollyServer {
    pub doc: Arc<Mutex<Document>>,
    pub services: Arc<Services>,
    registry: Arc<Registry>,
}

impl PollyServer {
    pub fn new(doc: Arc<Mutex<Document>>, services: Arc<Services>) -> Self {
        PollyServer { doc, services, registry: Arc::new(tools::registry()) }
    }

    pub fn tools(&self) -> Vec<Tool> {
        self.registry.list()
    }

    /// Run a tool synchronously (also used by tests and scripted sessions).
    pub fn call(&self, name: &str, args: serde_json::Value) -> CallToolResult {
        let Some(def) = self.registry.get(name) else {
            return CallToolResult::error(vec![ContentBlock::text(format!("unknown tool '{name}'"))]);
        };
        let mut doc = self.doc.lock().unwrap_or_else(|e| e.into_inner());
        let mut ctx = Ctx { doc: &mut doc, services: &self.services };
        let args = if args.is_null() { serde_json::json!({}) } else { args };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (def.call)(&mut ctx, args))) {
            Ok(Ok(out)) => CallToolResult::success(out.into_content()),
            Ok(Err(e)) => CallToolResult::error(vec![ContentBlock::text(format!("error: {e:#}"))]),
            Err(p) => {
                let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
                CallToolResult::error(vec![ContentBlock::text(format!(
                    "internal error in {name}: {}",
                    msg.unwrap_or_else(|| "panic".into())
                ))])
            }
        }
    }
}

impl ServerHandler for PollyServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("polly3d", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(self.tools()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.registry.get(name).map(|t| t.tool.clone())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let this = self.clone();
        let args = request.arguments.map(serde_json::Value::Object).unwrap_or_default();
        let name = request.name.to_string();
        let result = tokio::task::spawn_blocking(move || this.call(&name, args))
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(result.into())
    }
}
