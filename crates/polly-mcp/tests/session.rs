//! Drive the tool layer like an agent would.

use std::sync::Arc;

use polly_mcp::PollyServer;
use polly_mcp::services::Services;
use polly_mcp::workspace::Workspace;
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{Value, json};

fn server() -> PollyServer {
    PollyServer::new(Arc::new(Workspace::default()), Arc::new(Services::default()))
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
        assert!(t.input_schema["properties"].get("scene").is_some(), "{} lacks `scene`", t.name);
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
    assert_eq!(sel["selection"]["counts"]["faces"], 1);
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
    call(&s, "clear_scene", json!({}));
    call(&s, "restore", json!({"name": "a"}));
    let sc = call(&s, "describe_scene", json!({}));
    assert_eq!(sc["objects"].as_array().unwrap().len(), 1);
}

#[test]
fn palette_index_with_palette_in_same_call() {
    let s = server();
    // Regression: "p21" must resolve against the palette given in the same call.
    call(&s, "create_texture", json!({"name": "T", "width": 4, "height": 4, "palette": "db32", "fill": "p21"}));
    let db32 = polly_paint::palette::builtin("db32").unwrap();
    let px = s.workspace.get("main").unwrap().lock().unwrap().doc.scene.textures.values().next().unwrap().get(0, 0);
    assert_eq!(px, db32[21]);
    // Without any palette it is still a clear error.
    let e = call_err(&s, "create_texture", json!({"name": "U", "fill": "p21"}));
    assert!(e.contains("p21"), "{e}");
}

#[test]
fn unwrap_cube() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}}));
    let r = call(&s, "auto_seams", json!({}));
    assert!(r["seams_marked"].as_u64().unwrap() > 0);
    let r = call(&s, "unwrap", json!({"texture_size": [64, 64]}));
    assert_eq!(r["overlapping_faces"], "", "{r}");
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
    // colours nested in tuples (regression: palette indices used to be rejected here)
    let r = call(&s, "paint_pixels", json!({"texture": "T", "pixels": [[1, 0, "p5"], [2, 0, "#29adff"], [3, 0, [0, 0, 0, 255]]]}));
    assert_eq!(r["pixels_changed"], 3, "{r}");
    let e = call_err(&s, "paint_pixels", json!({"texture": "T", "pixels": [[1, 0, "p99"]]}));
    assert!(e.contains("p99"), "{e}");
    // strings that are not colours stay untouched
    call(&s, "rename_object", json!({"name": "p5"}));
    call(&s, "rename_object", json!({"object": "p5", "name": "Plane"}));
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
    call(&s, "clear_scene", json!({}));
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

#[test]
fn scenes_are_isolated() {
    let s = server();
    let a = call(&s, "create_scene", json!({"name": "a"}))["scene"].as_str().unwrap().to_string();
    let b = call(&s, "create_scene", json!({}))["scene"].as_str().unwrap().to_string();
    assert_eq!(b, "scene-1");
    call(&s, "add_primitive", json!({"scene": a, "shape": {"kind": "cube"}, "name": "Box"}));
    call(&s, "add_primitive", json!({"scene": b, "shape": {"kind": "torus"}, "name": "Ring"}));
    call(&s, "select", json!({"scene": a, "query": {"by": "all"}}));
    // main is untouched, and names resolve only within their scene
    assert!(call(&s, "describe_scene", json!({}))["objects"].as_array().unwrap().is_empty());
    let e = call_err(&s, "describe_mesh", json!({"scene": b, "object": "Box"}));
    assert!(e.contains("Box"), "{e}");
    // undo is per scene
    call(&s, "undo", json!({"scene": a}));
    assert!(call(&s, "describe_scene", json!({"scene": a}))["objects"].as_array().unwrap().is_empty());
    assert_eq!(call(&s, "describe_scene", json!({"scene": b}))["objects"].as_array().unwrap().len(), 1);
    // pick camera is per scene
    let r = s.call("render", json!({"scene": b}));
    if r.is_error != Some(true) {
        let e = call_err(&s, "pick", json!({"scene": a, "x": 10, "y": 10}));
        assert!(e.contains("nothing rendered"), "{e}");
        call(&s, "pick", json!({"scene": b, "x": 256, "y": 256}));
    }
    let l = call(&s, "list_scenes", json!({}));
    let names: Vec<_> = l["scenes"].as_array().unwrap().iter().map(|r| r["scene"].as_str().unwrap()).collect();
    assert_eq!(names, ["a", "main", "scene-1"]);
    // errors
    let e = call_err(&s, "describe_scene", json!({"scene": "nope"}));
    assert!(e.contains("unknown scene 'nope'") && e.contains("scene-1"), "{e}");
    assert!(call_err(&s, "create_scene", json!({"name": "a"})).contains("already exists"));
    assert!(call_err(&s, "delete_scene", json!({"name": "main"})).contains("cannot be deleted"));
    call(&s, "delete_scene", json!({"name": "a"}));
    call_err(&s, "describe_scene", json!({"scene": "a"}));
}

/// Many subagents at once, each in its own scene.
#[test]
fn parallel_scenes() {
    let s = server();
    let gpu = s.call("render", json!({})).is_error != Some(true);
    let names: Vec<String> = (0..8).map(|_| call(&s, "create_scene", json!({}))["scene"].as_str().unwrap().to_string()).collect();
    let handles: Vec<_> = names
        .iter()
        .cloned()
        .map(|sc| {
            let s = s.clone();
            std::thread::spawn(move || {
                let c = |name: &str, mut args: Value| {
                    args["scene"] = json!(sc);
                    call(&s, name, args)
                };
                c("add_primitive", json!({"shape": {"kind": "cube"}, "name": "Box"}));
                c("select", json!({"query": {"by": "normal", "direction": [0, 1, 0]}}));
                c("extrude", json!({"distance": 0.5}));
                c("auto_seams", json!({}));
                c("unwrap", json!({"texture_size": [32, 32]}));
                c("create_texture", json!({"name": "T", "width": 32, "height": 32, "material": "M", "object": "Box"}));
                c("fill_faces", json!({"elements": ["all"], "fill": {"type": "solid", "color": "#ff004d"}}));
                if gpu {
                    c("render", json!({"width": 128, "height": 128}));
                }
                c("describe_scene", json!({}))
            })
        })
        .collect();
    for h in handles {
        let d = h.join().unwrap();
        let objs = d["objects"].as_array().unwrap();
        assert_eq!(objs.len(), 1, "{d}");
        assert_eq!(objs[0]["name"], "Box");
        assert_eq!(objs[0]["faces"], 10);
    }
    if gpu {
        let r = s.call("render_overview", json!({"tile_size": 128}));
        let t: Value = serde_json::from_str(&text(&r)).unwrap();
        assert_eq!(t["tiles"].as_array().unwrap().len(), 9);
        std::fs::write(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("overview.png"), &images(&r)[0]).unwrap();
    }
}

/// Rig an arm, skin it, pose it, animate it, add a morph, look, export.
#[test]
fn rig_pose_animate_export() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "radius_bottom": 0.2, "radius_top": 0.2, "height": 2, "segments": 6, "height_segments": 4}, "name": "Arm"}));
    call(&s, "add_bone_chain", json!({"points": [[0, -1, 0], [0, 0, 0], [0, 1, 0]], "names": ["upper", "lower"]}));
    let e = call_err(&s, "add_bone", json!({"name": "upper", "head": [0, 0, 0], "tail": [1, 0, 0]}));
    assert!(e.contains("already exists"), "{e}");
    call(&s, "auto_weights", json!({"method": {"mode": "smooth", "falloff": 0.5}}));
    let rep = call(&s, "weight_report", json!({}));
    assert_eq!(rep["unweighted"]["count"], 0, "{rep}");
    assert_eq!(rep["unnormalized"]["count"], 0, "{rep}");
    let bones = call(&s, "list_bones", json!({}));
    assert_eq!(bones["bones"][1]["parent"], "upper");

    // pose + clip
    call(&s, "create_clip", json!({"name": "wave", "frames": 24}));
    call(&s, "key_pose", json!({"clip": "wave", "frame": 0, "bones": ["lower"]}));
    call(&s, "pose_bone", json!({"bone": "lower", "rotation": [0, 0, 90]}));
    call(&s, "key_pose", json!({"clip": "wave", "frame": 12}));
    call(&s, "update_clip", json!({"clip": "wave", "loop_start_frame": 6, "loop_end_frame": 18}));
    let d = call(&s, "describe_clip", json!({"clip": "wave"}));
    assert_eq!(d["loop_frames"], json!([6.0, 18.0]));
    assert_eq!(d["channels"][0]["frames"], json!([0.0, 12.0]));
    call(&s, "clear_pose", json!({}));

    // morph via begin/end
    call(&s, "begin_morph", json!({"name": "bulge"}));
    call(&s, "scale", json!({"elements": ["all"], "factor": [1.5, 1, 1.5]}));
    let m = call(&s, "end_morph", json!({}));
    assert!(m["moved_verts"].as_u64().unwrap() > 10, "{m}");
    let e = call_err(&s, "end_morph", json!({}));
    assert!(e.contains("no morph edit"), "{e}");
    call(&s, "set_key", json!({"clip": "wave", "frame": 12, "morph": "bulge", "value": [1.0]}));

    // render posed, weights heatmap, filmstrip (when a GPU is available)
    let r = s.call("render", json!({"pose": [{"clip": "wave", "frame": 12}], "overlays": {"bones": true, "bone_names": true}}));
    if r.is_error != Some(true) {
        let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
        std::fs::write(out.join("arm_posed.png"), &images(&r)[0]).unwrap();
        let w = s.call("render", json!({"show_weights": "lower", "view": "front"}));
        assert_ne!(w.is_error, Some(true), "{}", text(&w));
        std::fs::write(out.join("arm_weights.png"), &images(&w)[0]).unwrap();
        let f = s.call("render_animation", json!({"clips": [{"clip": "wave"}], "count": 4, "view": "front", "overlays": {"bones": true}}));
        assert_ne!(f.is_error, Some(true), "{}", text(&f));
        std::fs::write(out.join("arm_filmstrip.png"), &images(&f)[0]).unwrap();
        // Picking follows the pose of the last render: only the bent forearm
        // reaches x < -0.3 (at rest the arm stays within 0.2 of the axis).
        call(&s, "render", json!({"view": "front", "pose": [{"clip": "wave", "frame": 12}]}));
        let mut min_x = f64::MAX;
        for py in (16..512).step_by(32) {
            for px in (16..512).step_by(32) {
                let hit = call(&s, "pick", json!({"x": px, "y": py}));
                if hit["object"] == "Arm" {
                    min_x = min_x.min(hit["position"][0].as_f64().unwrap());
                }
            }
        }
        assert!(min_x < -0.3, "no hit on the bent forearm (min x {min_x})");
    }

    // export
    let dir = std::env::temp_dir().join(format!("polly-rig-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let glb = dir.join("arm.glb");
    call(&s, "export_glb", json!({"path": glb.to_str().unwrap()}));
    let (doc, _, _) = gltf::import(&glb).unwrap();
    assert_eq!(doc.skins().count(), 1);
    assert_eq!(doc.animations().count(), 1);
    std::fs::remove_dir_all(dir).ok();

    // deleting a bone removes its channels and weights
    call(&s, "delete_bone", json!({"bone": "lower"}));
    let d = call(&s, "describe_clip", json!({"clip": "wave"}));
    assert_eq!(d["channels"].as_array().unwrap().len(), 1, "{d}"); // morph channel remains
}

/// Agent-feedback batch: compact id ranges in and out, named parts that
/// survive joins and edits, match_transform.
#[test]
fn ranges_parts_and_match_transform() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "plane", "x_segments": 4, "z_segments": 4}, "name": "Body"}));
    // selecting everything reports a compact range, not 16 ids
    let sel = call(&s, "select", json!({"query": {"by": "all"}, "kind": "faces"}));
    assert_eq!(sel["selection"]["faces"], "f0..f15", "{sel}");
    // ranges and comma lists as input
    let r = call(&s, "extrude", json!({"elements": ["f0..f3, f8"], "distance": 0.2}));
    assert_eq!(r["result"]["faces"], "f0..f3, f8", "{r}");
    let e = call_err(&s, "move", json!({"elements": ["f200..f300"], "offset": [0, 1, 0]}));
    assert!(e.contains("f200..f300"), "{e}");

    // join keeps part names; later edits keep them on new faces
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.5, 0.5, 0.5]}, "name": "Head", "position": [0, 1, 0]}));
    let j = call(&s, "join_objects", json!({"objects": ["Body", "Head"]}));
    assert!(j["parts"]["Head"].as_str().unwrap().contains(".."), "{j}");
    let head = call(&s, "select", json!({"object": "Body", "query": {"by": "part", "name": "Head"}}));
    assert_eq!(head["selection"]["counts"]["faces"], 6);
    call(&s, "inset", json!({"object": "Body", "thickness": 0.05, "individual": true}));
    let d = call(&s, "describe_mesh", json!({"object": "Body"}));
    let head_after = call(&s, "select", json!({"object": "Body", "query": {"by": "part", "name": "Head"}}));
    assert_eq!(head_after["selection"]["counts"]["faces"], 6 * 5, "inset ring faces inherit the part: {d}");
    let e = call_err(&s, "select", json!({"object": "Body", "query": {"by": "part", "name": "Tail"}}));
    assert!(e.contains("Head") && e.contains("Body"), "{e}");
    call(&s, "set_part", json!({"object": "Body", "elements": ["f0"], "name": "Belt"}));
    let faces = call(&s, "list_faces", json!({"object": "Body", "elements": ["f0"]}));
    assert_eq!(faces["rows"][0]["part"], "Belt");

    // match_transform copies world placement, also across parents
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}, "name": "Arm", "position": [2, 1, 0], "rotation": [0, 0, 30]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}, "name": "Holder", "position": [-1, 0, 0]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.3, 0.3, 0.3]}, "name": "Bracer", "parent": "Holder"}));
    call(&s, "match_transform", json!({"object": "Bracer", "target": "Arm", "offset": [0, 0.5, 0]}));
    let sc = call(&s, "describe_scene", json!({}));
    let obj = |n: &str| sc["objects"].as_array().unwrap().iter().find(|o| o["name"] == n).unwrap().clone();
    // Arm's +Y rotated 30 degrees about Z: offset (0,0.5,0) -> (-0.25, 0.433, 0) from (2,1,0)
    let lo = obj("Bracer")["world_bounds"]["min"].clone();
    let hi = obj("Bracer")["world_bounds"]["max"].clone();
    let cx = (lo[0].as_f64().unwrap() + hi[0].as_f64().unwrap()) / 2.0;
    let cy = (lo[1].as_f64().unwrap() + hi[1].as_f64().unwrap()) / 2.0;
    assert!((cx - 1.75).abs() < 1e-3 && (cy - 1.433).abs() < 1e-3, "bracer centre {cx},{cy}");
    assert_eq!(obj("Bracer")["parent"], "Holder");
}
