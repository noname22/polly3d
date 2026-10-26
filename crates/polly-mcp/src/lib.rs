//! MCP server exposing the Polly3D modeller as tools.

pub mod colors;
pub mod registry;
pub mod services;
pub mod tools;
pub mod overview;
pub mod workspace;

use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer, ServerHandler};

use crate::registry::{Ctx, Registry};
use crate::services::Services;
use crate::workspace::{SceneSlot, Workspace};

pub const INSTRUCTIONS: &str = r#"Polly3D is a low-poly 3-D modeller you drive entirely through these tools.

Conventions
- Y is up, right-handed; "front" looks from +Z toward -Z, "right" from +X. Units are arbitrary (1 = 1 m is a good habit).
- Rotations are Euler degrees [x, y, z] with R = Rx·Ry·Rz: about the parent's (or world) axes Z is applied first, then Y, then X; equivalently X, Y, Z about the object's own axes. Positive angles turn counter-clockwise when looking down the axis toward the origin (right-hand rule).
- Parenting (set_parent) makes a part follow another (e.g. a bracer parented to its forearm); match_transform copies one object's placement onto another.
- Objects are addressed by name (or "o<id>"); tools default to the active object (the last one created/selected).
- Mesh elements have stable ids: vertices "v12", faces "f3", edges "e3-9" (the two vertex ids). Ids never get reused.
- Element arguments accept ids, ranges ("f400..f507" = the faces that exist in that range; also "v1..v9"), comma lists in one string ("f1, f4..f9"), "all" and "selection". Results report ids in the same compact form, so they can be pasted back.
- Faces can carry a part name (join_objects names parts after the joined objects; set_part labels faces); select {"by":"part","name":"head"}.
- Most edit tools take `elements` (ids, "all", or "selection"); when omitted they act on the object's current selection. Tools that create geometry (extrude, inset, bevel, loop_cut, subdivide, ...) set the selection to the result, so you can chain them.
- Edits are atomic and undoable (undo/redo, checkpoint/restore). Each edit reports added/removed ids and topology warnings.
- UVs follow glTF: (0,0) is the top-left of the texture, v grows downward. Pixel coordinates are (0,0) at top-left.

Workflow that works well
1. Block out with primitives (add_primitive), keep segment counts low.
2. Look: render (view persp/front/right/top, overlays face_ids) or render_contact_sheet. Check ids before editing.
3. Refine with select (by normal / box / loops / part) + extrude, inset, bevel, loop_cut, move/scale. Render again after each meaningful change.
   - Organic shaping: move/rotate/scale with `falloff` (soft selection; `connected: true` to leave nearby separate parts alone).
   - Single-sided sheets (hair, cloth, leaves): solidify to give them thickness.
   - Symmetric parts: build one side, then mirror_object for the other (names swap .L/.R), or mirror for a symmetric single mesh.
4. UV: mark seams (or auto_seams), unwrap with a small texture size (e.g. 64x64) and pixel snapping; inspect with render_uv_layout.
5. Paint: create_texture (optionally palette-locked), assign it through a material, fill_faces / draw / paint_stroke, check render.
   - unwrap/pack_uvs `density` gives important parts more texels ({"head": 2}).
   - fill_faces with {"type":"noise3d"} for organic surfaces (no seams); stamp for buttons/eyes/rivets sized in metres; paint_stroke "dabs" + "mirror" for symmetric details.
   - Finish with bake_shading (ao and/or height) for depth; use dither on palette-locked textures.
6. save_project, export_glb.

Rigging & animation (optional)
- Rig: add_bone / add_bone_chain (head/tail points in object space), check with render overlays {"bones":true,"bone_names":true}.
- Characters with separate parts or gear: put the skeleton on one object (an armature, e.g. an empty "Rig" from add_mesh with no faces), then bind_to_armature each mesh (body, hair, clothing). One pose and one set of clips then drive them all. Bone, pose and clip tools accept a bound mesh and act on its armature; weights stay per mesh.
- Gear: rigid items (helmet, sword, bracer) -> attach_to_bone; fitted deforming items (skirt, sleeves, cape, long hair) -> transfer_weights from the body (binds them too). mirror_object moves both to the mirrored bones.
- Swappable gear: add_variant groups objects into slots ("hair": short/long/none, "head": none/helmet). One variant per slot is active (shown) and one is the default (visible when the exported model loads). A variant can hide named parts of other objects (label them with set_part) to avoid poke-through, e.g. a helmet hiding "hair_top". render accepts {"variants": {...}}; render_variants shows a slot's options side by side.
- Skin: auto_weights (rigid = each vertex follows one bone, the classic low-poly look; smooth = blended joints) or assign_weights; check with render {"show_weights":"<bone>"} and weight_report.
- Pose: pose_bone (Euler degrees around the bone's own axes), then key_pose into a clip (create_clip; frames at the clip's fps). Repeat at other frames. Loop regions via update_clip.
- Morphs: begin_morph, move vertices with the normal tools, end_morph; set_morph_weight or key them (key_pose with morphs, or set_key).
- Check motion with render_animation (a filmstrip) or render {"pose":[{"clip":"walk","frame":6}]}; several clips with weights blend.
- export_glb writes skins (one per armature, shared by its meshes), bone-attached gear, morph targets and animations. Gear variants are all exported: defaults visible, alternatives hidden (KHR_node_visibility) and described in extras for runtime swapping.

The user may ask to watch live: show_viewer opens a window on their screen, hide_viewer closes it. Don't open it unasked.
Scenes: everything above happens in a scene ("main" unless a tool call passes `scene`). Each scene has its own objects, selection, undo history and render camera.
To build unrelated models in parallel (e.g. one per subagent), call create_scene once per model and tell each subagent to pass `scene: "<its name>"` on EVERY call. render_overview shows all scenes in one image; list_scenes lists them. Subagents sharing one scene should each work on their own object and pass explicit `object` and element ids.
"#;

#[derive(Clone)]
pub struct PollyServer {
    pub workspace: Arc<Workspace>,
    pub services: Arc<Services>,
    registry: Arc<Registry>,
}

impl PollyServer {
    pub fn new(workspace: Arc<Workspace>, services: Arc<Services>) -> Self {
        PollyServer { workspace, services, registry: Arc::new(tools::registry()) }
    }

    pub fn tools(&self) -> Vec<Tool> {
        self.registry.list()
    }

    /// Run a tool synchronously (also used by tests and scripted sessions).
    pub fn call(&self, name: &str, args: serde_json::Value) -> CallToolResult {
        let Some(def) = self.registry.get(name) else {
            return CallToolResult::error(vec![ContentBlock::text(format!("unknown tool '{name}'"))]);
        };
        let mut args = if args.is_null() { serde_json::json!({}) } else { args };
        let scene = match args.as_object_mut().and_then(|m| m.remove("scene")) {
            None | Some(serde_json::Value::Null) => workspace::MAIN.to_string(),
            Some(serde_json::Value::String(s)) => s,
            Some(other) => return CallToolResult::error(vec![ContentBlock::text(format!("error: `scene` must be a string, got {other}"))]),
        };
        let slot = match self.workspace.get(&scene) {
            Ok(s) => s,
            Err(e) => return CallToolResult::error(vec![ContentBlock::text(format!("error: {e}"))]),
        };
        // Only this scene is locked: calls in other scenes proceed in parallel.
        let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
        let SceneSlot { doc, last_view, project_path } = &mut *guard;
        let mut ctx = Ctx {
            doc,
            scene: &scene,
            last_view,
            project_path,
            workspace: &self.workspace,
            services: &self.services,
        };
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
