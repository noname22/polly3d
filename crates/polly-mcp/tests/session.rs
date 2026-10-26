//! Drive the tool layer like an agent would.

use std::sync::{Arc, Mutex};

use polly_core::document::Document;
use polly_mcp::PollyServer;
use polly_mcp::services::Services;
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{Value, json};

fn server() -> PollyServer {
    PollyServer::new(Arc::new(Mutex::new(Document::new())), Arc::new(Services::default()))
}

fn text(r: &CallToolResult) -> String {
    r.content
        .iter()
        .filter_map(|c| match c {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[track_caller]
fn call(s: &PollyServer, name: &str, args: Value) -> Value {
    let r = s.call(name, args.clone());
    let t = text(&r);
    assert_ne!(r.is_error, Some(true), "{name} {args} failed: {t}");
    serde_json::from_str(&t).unwrap_or(Value::String(t))
}

#[track_caller]
fn call_err(s: &PollyServer, name: &str, args: Value) -> String {
    let r = s.call(name, args);
    assert_eq!(r.is_error, Some(true), "{name} unexpectedly succeeded: {}", text(&r));
    text(&r)
}

#[test]
fn all_tool_schemas_build() {
    let s = server();
    let tools = s.tools();
    assert!(tools.len() > 50);
    for t in &tools {
        assert_eq!(t.input_schema.get("type"), Some(&json!("object")), "{}", t.name);
    }
    // colour widening reached nested Fill schemas
    let ff = tools.iter().find(|t| t.name == "fill_faces").unwrap();
    assert!(serde_json::to_string(&*ff.input_schema).unwrap().contains("palette index"));
}

#[test]
fn model_a_crate() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}, "name": "Crate"}));
    let sel = call(&s, "select", json!({"query": {"by": "normal", "direction": [0, 1, 0]}}));
    assert_eq!(sel["selection"]["faces"].as_array().unwrap().len(), 1);
    let r = call(&s, "inset", json!({"thickness": 0.1, "depth": -0.05}));
    assert!(r.get("warnings").is_none(), "{r}");
    let r = call(&s, "bevel", json!({"elements": ["all"], "width": 0.02}));
    assert!(r["changes"]["counts"]["faces"].as_u64().unwrap() > 10);
    call(&s, "undo", json!({}));
    let d = call(&s, "describe_mesh", json!({}));
    assert_eq!(d["faces"], 10);
    let e = call_err(&s, "extrude", json!({"elements": ["f999"], "distance": 1}));
    assert!(e.contains("f999"), "{e}");
    let faces = call(&s, "list_faces", json!({"limit": 3}));
    assert_eq!(faces["rows"].as_array().unwrap().len(), 3);
    assert_eq!(faces["next_offset"], 3);
    call(&s, "mirror", json!({"axis": "x"}));
    call(&s, "checkpoint", json!({"name": "a"}));
    call(&s, "new_scene", json!({}));
    call(&s, "restore", json!({"name": "a"}));
    let sc = call(&s, "describe_scene", json!({}));
    assert_eq!(sc["objects"].as_array().unwrap().len(), 1);
}

#[test]
fn unwrap_cube() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}}));
    let r = call(&s, "auto_seams", json!({}));
    assert!(r["seams_marked"].as_u64().unwrap() > 0);
    let r = call(&s, "unwrap", json!({"texture_size": [64, 64]}));
    assert!(r["overlapping_faces"].as_array().unwrap().is_empty(), "{r}");
    assert!(!r["islands"].as_array().unwrap().is_empty());
    call(&s, "uv_report", json!({}));
    call(&s, "transform_uvs", json!({"elements": ["f0"], "translate": [0.1, 0.0]}));
}

#[test]
fn paint_flow_with_hex_and_palette_colours() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "plane"}}));
    call(&s, "create_texture", json!({"name": "T", "width": 8, "height": 8, "palette": "pico8", "material": "M", "object": "Plane"}));
    // no uvs yet
    let e = call_err(&s, "fill_faces", json!({"elements": ["all"], "fill": {"type": "solid", "color": "#ff0000"}}));
    assert!(e.contains("UV"), "{e}");
    call(&s, "project_uv", json!({"projection": {"method": "per_face"}}));
    let r = call(&s, "fill_faces", json!({"elements": ["all"], "fill": {"type": "checker", "a": "p0", "b": "#ff004d", "size_px": 2}}));
    assert_eq!(r["pixels_changed"], 64);
    call(&s, "draw", json!({"texture": "T", "shape": "line", "from": [0, 0], "to": [7, 7], "color": "white"}));
    let r = s.call("get_texture", json!({"texture": "T"}));
    assert!(r.content.iter().any(|c| matches!(c, ContentBlock::Image(_))));
    let t = text(&r);
    let v: Value = serde_json::from_str(&t).unwrap();
    assert_eq!(v["stats"]["palette_violations"], 0, "{t}");
    let dir = std::env::temp_dir().join(format!("polly-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let glb = dir.join("out.glb");
    call(&s, "export_glb", json!({"path": glb.to_str().unwrap()}));
    assert!(std::fs::metadata(&glb).unwrap().len() > 100);
    let proj = dir.join("p.polly");
    call(&s, "save_project", json!({"path": proj.to_str().unwrap()}));
    call(&s, "new_scene", json!({}));
    call(&s, "load_project", json!({}));
    assert_eq!(call(&s, "describe_scene", json!({}))["textures"].as_array().unwrap().len(), 1);
    std::fs::remove_dir_all(dir).ok();
}

fn images(r: &CallToolResult) -> Vec<Vec<u8>> {
    use base64::Engine;
    r.content
        .iter()
        .filter_map(|c| match c {
            ContentBlock::Image(i) => Some(base64::engine::general_purpose::STANDARD.decode(&i.data).unwrap()),
            _ => None,
        })
        .collect()
}

/// Full agent loop with eyes: model, unwrap, paint by screen coords, look.
/// Skips when no GPU/software adapter is available.
#[test]
fn render_pick_and_paint_stroke() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}, "name": "Box"}));
    call(&s, "auto_seams", json!({}));
    call(&s, "unwrap", json!({"texture_size": [32, 32]}));
    call(&s, "create_texture", json!({"name": "T", "width": 32, "height": 32, "fill": "#808080", "material": "M", "object": "Box"}));
    let r = s.call("render", json!({"view": "front", "overlays": {"face_ids": true}}));
    if r.is_error == Some(true) {
        eprintln!("skipping: {}", text(&r));
        return;
    }
    let img = images(&r);
    assert_eq!(img.len(), 1);
    let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::write(out.join("session_front.png"), &img[0]).unwrap();
    let hit = call(&s, "pick", json!({"x": 256, "y": 256}));
    assert_eq!(hit["object"], "Box", "{hit}");
    let miss = call(&s, "pick", json!({"x": 2, "y": 2}));
    assert!(miss["hit"].is_null());
    let st = call(&s, "paint_stroke", json!({"points": [[200, 200], [312, 312]], "color": "#ff0000", "radius_px": 1}));
    assert!(st["pixels_changed"]["T"].as_u64().unwrap() > 5, "{st}");
    let r = s.call("render", json!({"view": "front", "overlays": {"wireframe": false, "grid": false}, "legend": false}));
    let img = image::load_from_memory(&images(&r)[0]).unwrap().to_rgba8();
    std::fs::write(out.join("session_painted.png"), &images(&r)[0]).unwrap();
    let p = img.get_pixel(256, 256);
    assert!(p[0] > 150 && p[1] < 80, "stroke should be visible at the image centre, got {p:?}");
    let sheet = s.call("render_contact_sheet", json!({}));
    assert_eq!(images(&sheet).len(), 1);
    call(&s, "pick", json!({"x": 256, "y": 256})); // front tile of the sheet
    let uv = s.call("render_uv_layout", json!({}));
    std::fs::write(out.join("session_uv.png"), &images(&uv)[0]).unwrap();
    call(&s, "set_auto_preview", json!({"size": 128}));
    let r = s.call("extrude", json!({"elements": ["f3"], "distance": 0.3}));
    assert_eq!(images(&r).len(), 1);
}
