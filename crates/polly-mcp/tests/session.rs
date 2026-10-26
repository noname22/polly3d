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
fn paint_flow_with_hex_and_palette_colours() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "plane"}}));
    call(&s, "create_texture", json!({"name": "T", "width": 8, "height": 8, "palette": "pico8", "material": "M", "object": "Plane"}));
    // no uvs yet
    let e = call_err(&s, "fill_faces", json!({"elements": ["all"], "fill": {"type": "solid", "color": "#ff0000"}}));
    assert!(e.contains("UV"), "{e}");
    // give the plane full-square uvs by hand
    {
        let doc = s.doc.lock().unwrap();
        drop(doc);
    }
    let mut doc = s.doc.lock().unwrap();
    let o = doc.scene.objects.values_mut().next().unwrap();
    let m = Arc::make_mut(&mut o.mesh);
    for f in m.faces.values_mut() {
        let uv = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        for (c, u) in f.corners.iter_mut().zip(uv) {
            c.uv = polly_core::glam::Vec2::from_array(u);
        }
    }
    drop(doc);
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
