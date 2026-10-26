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

/// Modelling batch: solidify, soft selection, mirror_object.
#[test]
fn solidify_soft_selection_mirror_object() {
    let s = server();
    // solidify: a skirt-like open sheet becomes a closed shell
    call(&s, "add_primitive", json!({"shape": {"kind": "plane", "x_segments": 2, "z_segments": 2}, "name": "Sheet"}));
    let r = call(&s, "solidify", json!({"elements": ["all"], "thickness": 0.05}));
    assert!(r.get("warnings").is_none(), "{r}");
    assert_eq!(call(&s, "describe_mesh", json!({}))["boundary_edges"], 0);

    // soft move: a strip lifted at its centre falls off linearly
    call(&s, "add_primitive", json!({"shape": {"kind": "plane", "size": [4, 1], "x_segments": 8, "z_segments": 1}, "name": "Strip"}));
    call(&s, "select", json!({"query": {"by": "box", "min": [-0.01, -1, -1], "max": [0.01, 1, 1]}, "kind": "verts"}));
    let r = call(&s, "move", json!({"offset": [0, 1, 0], "falloff": 1.0, "falloff_curve": "linear"}));
    assert_eq!(r["soft_selection"]["also_moved"], 2 * 2, "{r}"); // x = ±0.5 rows (x = ±1.0 is at weight 0)
    let verts = call(&s, "list_verts", json!({"elements": ["all"]}));
    for row in verts["rows"].as_array().unwrap() {
        let p = &row[1];
        let (x, y) = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
        let expect = (1.0 - x.abs()).max(0.0);
        assert!((y - expect).abs() < 1e-3, "x={x} y={y} expected {expect}");
    }

    // mirror_object: an off-centre, rotated, non-uniformly scaled part
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.3, 0.2, 0.6]}, "name": "Boot.L", "position": [0.3, 0.1, 0.2], "rotation": [0, 20, 10], "scale": [1, 2, 1]}));
    let r = call(&s, "mirror_object", json!({"object": "Boot.L"}));
    assert_eq!(r["name"], "Boot.R", "{r}");
    let ws = s.workspace.get("main").unwrap();
    let g = ws.lock().unwrap();
    let sc = &g.doc.scene;
    let find = |n: &str| *sc.objects.iter().find(|(_, o)| o.name == n).unwrap().0;
    let (l, rr) = (find("Boot.L"), find("Boot.R"));
    let world = |id: u32| -> Vec<polly_core::glam::Vec3> {
        let m = sc.world_matrix(id);
        sc.objects[&id].mesh.verts.values().map(|v| m.transform_point3(v.pos)).collect()
    };
    let (wl, wr) = (world(l), world(rr));
    for p in &wl {
        let q = polly_core::glam::Vec3::new(-p.x, p.y, p.z);
        assert!(wr.iter().any(|x| x.distance(q) < 1e-4), "no mirrored vertex for {p}");
    }
    let mesh = &sc.objects[&rr].mesh;
    assert!(mesh.validate().is_empty(), "{:?}", mesh.validate());
    let m = sc.world_matrix(rr);
    let centre = wr.iter().copied().sum::<polly_core::glam::Vec3>() / wr.len() as f32;
    for f in mesh.faces.values() {
        let pts: Vec<_> = f.verts.iter().map(|v| m.transform_point3(mesh.pos(*v))).collect();
        let n = polly_core::mesh::newell_normal(&pts);
        let c = pts.iter().copied().sum::<polly_core::glam::Vec3>() / pts.len() as f32;
        assert!(n.dot(c - centre) > 0.0, "mirrored copy has an inward-facing face");
    }
    assert!(sc.objects[&rr].transform.scale.min_element() > 0.0);
}

/// Texturing batch: per-part density, 3-D noise, baked shading, stamp,
/// dabs with mirrored painting.
#[test]
fn texturing_batch() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [1.0, 1.4, 0.7]}, "name": "Body"}));
    call(&s, "add_primitive", json!({"shape": {"kind": "uv_sphere", "radius": 0.45, "segments": 10, "rings": 7}, "name": "Head", "position": [0, 1.15, 0]}));
    call(&s, "join_objects", json!({"objects": ["Body", "Head"]}));
    call(&s, "auto_seams", json!({}));
    call(&s, "create_texture", json!({"name": "Skin", "width": 128, "height": 128, "palette": "db32", "material": "M", "object": "Body"}));
    call(&s, "unwrap", json!({"density": {"Head": 2.0}}));
    // the head got about twice the texels per unit (4x per area) of the body
    {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let m = &g.doc.scene.objects.values().next().unwrap().mesh;
        let density = |part: &str| {
            let (mut uv, mut a3) = (0.0, 0.0);
            for f in m.faces.values().filter(|f| f.part.as_deref() == Some(part)) {
                let p: Vec<_> = f.corners.iter().map(|c| c.uv.extend(0.0)).collect();
                uv += polly_core::mesh::newell_vector(&p).length() / 2.0;
                a3 += m.face_area(f);
            }
            uv / a3
        };
        let ratio = density("Head") / density("Body");
        assert!(ratio > 3.0 && ratio < 5.0, "head/body area density ratio {ratio}");
    }
    call(&s, "fill_faces", json!({"elements": ["all"], "fill": {"type": "solid", "color": "p21"}}));
    call(&s, "select", json!({"query": {"by": "part", "name": "Head"}}));
    let n = call(&s, "fill_faces", json!({"elements": ["selection"], "fill": {"type": "noise3d", "colors": ["p3", "p4", "p5"], "scale": 0.15, "contrast": 1.4, "dither": "bayer4"}}));
    assert!(n["pixels_changed"].as_u64().unwrap() > 50, "{n}");
    let b = call(&s, "bake_shading", json!({"ao": {"samples": 16}, "height": {"strength": 0.3}}));
    assert!(b["pixels_changed"].as_u64().unwrap() > 100, "{b}");
    let r = s.call("render", json!({"view": "front"}));
    if r.is_error == Some(true) {
        eprintln!("skipping GPU part: {}", text(&r));
        return;
    }
    // A button on the belly, then two eyes from one mirrored dab.
    let st = call(&s, "stamp", json!({"pixel": [256, 330], "radius": 0.08, "color": "p8", "outline": "p0"}));
    assert!(st["pixels_changed"].as_u64().unwrap() > 4, "{st}");
    let mut eye = None;
    for x in (200..256).rev().step_by(4) {
        let h = call(&s, "pick", json!({"x": x, "y": 150}));
        if h["object"] == "Body" && h["position"][0].as_f64().unwrap() < -0.08 {
            eye = Some(x);
            break;
        }
    }
    let eye = eye.expect("no head surface left of centre");
    let p = call(&s, "paint_stroke", json!({"points": [[eye, 150]], "mode": "dabs", "color": "p1", "radius_px": 2, "mirror": {}}));
    assert_eq!(p["mirrored_hits"], 1, "{p}");
    let stats = call(&s, "get_texture", json!({"texture": "Skin"}));
    assert_eq!(stats["stats"]["palette_violations"], 0, "{stats}");
    let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let r = s.call("render", json!({"view": "front", "overlays": {"wireframe": false, "grid": false, "selection": false}, "width": 400, "height": 400}));
    std::fs::write(out.join("texturing_front.png"), &images(&r)[0]).unwrap();
    let r = s.call("render", json!({"view": "persp", "overlays": {"wireframe": false, "selection": false}, "width": 400, "height": 400}));
    std::fs::write(out.join("texturing_persp.png"), &images(&r)[0]).unwrap();
    let t = s.call("get_texture", json!({"texture": "Skin", "scale": 3}));
    std::fs::write(out.join("texturing_tex.png"), &images(&t)[0]).unwrap();
}

/// Phase 1 of gear support: one armature deforms several bound meshes.
#[test]
fn shared_armature() {
    let s = server();
    call(&s, "add_mesh", json!({"name": "Rig", "vertices": [], "faces": []}));
    call(&s, "add_bone_chain", json!({"object": "Rig", "points": [[0, -1, 0], [0, 0, 0], [0, 1, 0]], "names": ["upper", "lower"]}));
    let cyl = |r: f32| json!({"kind": "cylinder", "radius_bottom": r, "radius_top": r, "height": 2, "segments": 6, "height_segments": 4});
    call(&s, "add_primitive", json!({"shape": cyl(0.2), "name": "Body"}));
    call(&s, "add_primitive", json!({"shape": cyl(0.25), "name": "Sleeve"}));
    for m in ["Body", "Sleeve"] {
        let r = call(&s, "bind_to_armature", json!({"object": m, "armature": "Rig"}));
        assert!(r["hint"].as_str().unwrap().contains("auto_weights"), "{r}");
        call(&s, "auto_weights", json!({"object": m, "method": {"mode": "smooth"}}));
        assert_eq!(call(&s, "weight_report", json!({"object": m}))["unweighted"]["count"], 0);
    }
    let bones = call(&s, "list_bones", json!({"object": "Sleeve"})); // a bound mesh resolves to its rig
    assert_eq!(bones["object"], "Rig");
    assert_eq!(bones["bound_meshes"], json!(["Body", "Sleeve"]));
    let e = call_err(&s, "bind_to_armature", json!({"object": "Body", "armature": "Sleeve"}));
    assert!(e.contains("no skeleton"), "{e}");

    // Posing "the body" poses the rig; a clip on the rig moves both meshes.
    call(&s, "create_clip", json!({"name": "bend", "frames": 12}));
    call(&s, "key_pose", json!({"clip": "bend", "frame": 0, "object": "Body", "bones": ["lower"]}));
    call(&s, "pose_bone", json!({"object": "Body", "bone": "lower", "rotation": [0, 0, 90]}));
    call(&s, "key_pose", json!({"clip": "bend", "frame": 12, "object": "Body"}));
    call(&s, "clear_pose", json!({"object": "Body"}));
    let d = call(&s, "describe_clip", json!({"clip": "bend"}));
    assert_eq!(d["channels"][0]["object"], "Rig", "{d}");
    {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let sc = &g.doc.scene;
        let cid = sc.resolve_clip("bend").unwrap();
        let posed = polly_core::anim::pose_scene(sc, &[polly_core::anim::ClipPlay { clip: cid, time: 0.5, weight: 1.0 }]);
        for name in ["Body", "Sleeve"] {
            let id = sc.resolve(Some(name)).unwrap();
            let (rest, now) = (&sc.objects[&id].mesh, &posed.objects[&id].mesh);
            let tip = rest.verts.iter().max_by(|a, b| a.1.pos.y.total_cmp(&b.1.pos.y)).map(|(v, _)| *v).unwrap();
            let w = posed.world_matrix(id).transform_point3(now.pos(tip));
            assert!(w.x < -0.7 && w.y.abs() < 0.35, "{name} tip should swing to -X, got {w}");
        }
    }

    // A mirrored glove moves its weights to the mirrored bones.
    call(&s, "add_bone", json!({"object": "Rig", "name": "hand.L", "head": [0.5, 0, 0], "tail": [0.8, 0, 0]}));
    call(&s, "add_bone", json!({"object": "Rig", "name": "hand.R", "head": [-0.5, 0, 0], "tail": [-0.8, 0, 0]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.2, 0.2, 0.2]}, "name": "Glove.L", "position": [0.65, 0, 0]}));
    call(&s, "bind_to_armature", json!({"object": "Glove.L", "armature": "Rig"}));
    call(&s, "assign_weights", json!({"object": "Glove.L", "elements": ["all"], "bone": "hand.L"}));
    call(&s, "mirror_object", json!({"object": "Glove.L"}));
    let rep = call(&s, "weight_report", json!({"object": "Glove.R"}));
    assert_eq!(rep["verts_per_bone"]["hand.R"], 8, "{rep}");
    assert_eq!(rep["verts_per_bone"]["hand.L"], 0, "{rep}");

    // Export says what isn't supported yet; deleting the rig unbinds.
    let dir = std::env::temp_dir().join(format!("polly-armature-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    call(&s, "export_glb", json!({"path": dir.join("a.glb").to_str().unwrap()}));
    std::fs::remove_dir_all(dir).ok();
    call(&s, "delete_object", json!({"object": "Rig"}));
    let sc = call(&s, "describe_scene", json!({}));
    assert!(sc["objects"].as_array().unwrap().iter().all(|o| o.get("skinned_to").is_none()), "{sc}");
}

/// Phase 2 of gear support: weight transfer and bone attachment.
#[test]
fn fitting_gear() {
    let s = server();
    let world_center = |s: &PollyServer, name: &str| -> [f64; 3] {
        let sc = call(s, "describe_scene", json!({}));
        let o = sc["objects"].as_array().unwrap().iter().find(|o| o["name"] == name).unwrap().clone();
        let (lo, hi) = (&o["world_bounds"]["min"], &o["world_bounds"]["max"]);
        [0, 1, 2].map(|i| (lo[i].as_f64().unwrap() + hi[i].as_f64().unwrap()) / 2.0)
    };
    call(&s, "add_mesh", json!({"name": "Rig", "vertices": [], "faces": []}));
    call(&s, "add_bone_chain", json!({"object": "Rig", "points": [[0, -1, 0], [0, 0, 0], [0, 1, 0]], "names": ["upper", "lower"]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "radius_bottom": 0.2, "radius_top": 0.2, "height": 2, "segments": 8, "height_segments": 6}, "name": "Body"}));
    call(&s, "bind_to_armature", json!({"object": "Body", "armature": "Rig"}));
    call(&s, "auto_weights", json!({"object": "Body", "method": {"mode": "smooth"}}));

    // A skirt around the lower body gets the body's weights and its armature.
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "radius_bottom": 0.35, "radius_top": 0.25, "height": 0.7, "segments": 8, "caps": false}, "name": "Skirt", "position": [0, -0.6, 0]}));
    let t = call(&s, "transfer_weights", json!({"object": "Skirt", "from": "Body"}));
    assert_eq!(t["armature"], "Rig", "{t}");
    assert_eq!(t["weighted_verts"], 16, "{t}");
    assert!(t.get("far_from_source").is_none(), "{t}");
    assert_eq!(call(&s, "describe_scene", json!({}))["objects"].as_array().unwrap().iter().find(|o| o["name"] == "Skirt").unwrap()["skinned_to"], "Rig");

    // A sword at the forearm tip, attached to the lower bone, keeps its place...
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.05, 0.6, 0.05]}, "name": "Sword", "position": [0, 1.3, 0]}));
    let before = world_center(&s, "Sword");
    let r = call(&s, "attach_to_bone", json!({"object": "Sword", "armature": "Body", "bone": "lower"}));
    assert_eq!(r["to"], "Rig / lower", "{r}");
    let after = world_center(&s, "Sword");
    assert!((0..3).all(|i| (before[i] - after[i]).abs() < 1e-4), "{before:?} -> {after:?}");
    let e = call_err(&s, "attach_to_bone", json!({"object": "Skirt", "armature": "Rig", "bone": "lower"}));
    assert!(e.contains("skinned"), "{e}");
    // ...and swings with the bone.
    call(&s, "pose_bone", json!({"object": "Rig", "bone": "lower", "rotation": [0, 0, 90]}));
    {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let sc = &g.doc.scene;
        let posed = polly_core::anim::pose_scene(sc, &[]);
        let id = sc.resolve(Some("Sword")).unwrap();
        let c = posed.world_matrix(id).transform_point3(polly_core::glam::Vec3::ZERO);
        assert!((c - polly_core::glam::Vec3::new(-1.3, 0.0, 0.0)).length() < 1e-3, "sword at {c}");
    }
    call(&s, "clear_pose", json!({"object": "Rig"}));

    // A bracer on arm.L, mirrored, ends up on arm.R; deleting arm.R leaves it in place.
    call(&s, "add_bone", json!({"object": "Rig", "name": "arm.L", "head": [0.3, 0.5, 0], "tail": [0.8, 0.5, 0]}));
    call(&s, "add_bone", json!({"object": "Rig", "name": "arm.R", "head": [-0.3, 0.5, 0], "tail": [-0.8, 0.5, 0]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "radius_bottom": 0.1, "radius_top": 0.1, "height": 0.2, "segments": 6}, "name": "Bracer.L", "position": [0.55, 0.5, 0], "rotation": [0, 0, 90]}));
    call(&s, "attach_to_bone", json!({"object": "Bracer.L", "armature": "Rig", "bone": "arm.L"}));
    call(&s, "mirror_object", json!({"object": "Bracer.L"}));
    let sc = call(&s, "describe_scene", json!({}));
    let br = sc["objects"].as_array().unwrap().iter().find(|o| o["name"] == "Bracer.R").unwrap().clone();
    assert_eq!(br["attached_to"], "Rig / arm.R", "{br}");
    let c = world_center(&s, "Bracer.R");
    assert!((c[0] + 0.55).abs() < 1e-3 && (c[1] - 0.5).abs() < 1e-3, "{c:?}");
    call(&s, "delete_bone", json!({"object": "Rig", "bone": "arm.R"}));
    let c2 = world_center(&s, "Bracer.R");
    assert!((0..3).all(|i| (c[i] - c2[i]).abs() < 1e-4), "{c:?} -> {c2:?}");

    // Export: the sword node sits at its rest place; the limitation is reported.
    let dir = std::env::temp_dir().join(format!("polly-gear-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("g.glb");
    call(&s, "export_glb", json!({"path": path.to_str().unwrap()}));
    let (doc, _, _) = gltf::import(&path).unwrap();
    let mut world = std::collections::HashMap::new();
    fn walk(n: gltf::Node, parent: polly_core::glam::Mat4, out: &mut std::collections::HashMap<String, polly_core::glam::Mat4>) {
        let m = parent * polly_core::glam::Mat4::from_cols_array_2d(&n.transform().matrix());
        out.insert(n.name().unwrap_or("").to_string(), m);
        for c in n.children() {
            walk(c, m, out);
        }
    }
    for n in doc.scenes().next().unwrap().nodes() {
        walk(n, polly_core::glam::Mat4::IDENTITY, &mut world);
    }
    let p = world["Sword"].transform_point3(polly_core::glam::Vec3::ZERO);
    assert!((p - polly_core::glam::Vec3::new(0.0, 1.3, 0.0)).length() < 1e-3, "sword exported at {p}");
    std::fs::remove_dir_all(dir).ok();
}

/// Phase 3 of gear support: slots, variants and hide rules.
#[test]
fn slots_and_variants() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.6, 1.0, 0.4]}, "name": "Body"}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.4, 0.2, 0.4]}, "name": "HairShort", "position": [0, 0.7, 0]}));
    call(&s, "select", json!({"object": "HairShort", "query": {"by": "normal", "direction": [0, 1, 0]}}));
    call(&s, "set_part", json!({"object": "HairShort", "name": "hair_top"}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.45, 0.6, 0.45]}, "name": "HairLong", "position": [0, 0.5, 0]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "radius_bottom": 0.3, "radius_top": 0.0, "height": 0.4}, "name": "Helmet", "position": [0, 0.95, 0]}));

    let r = call(&s, "add_variant", json!({"slot": "hair", "variant": "short", "objects": ["HairShort"]}));
    assert_eq!(r["note"], "created slot 'hair'", "{r}");
    call(&s, "add_variant", json!({"slot": "hair", "variant": "long", "objects": ["HairLong"]}));
    call(&s, "add_variant", json!({"slot": "head", "variant": "none"}));
    call(&s, "add_variant", json!({"slot": "head", "variant": "helmet", "objects": ["Helmet"], "hides": [{"object": "HairShort", "part": "hair_top"}]}));
    let slots = call(&s, "list_slots", json!({}));
    assert_eq!(slots["slots"][0]["default"], "short");
    assert_eq!(slots["slots"][1]["variants"][1]["hides"][0]["part"], "hair_top");

    // mistakes are explained
    let e = call_err(&s, "add_variant", json!({"slot": "head", "variant": "wig", "objects": ["HairLong"]}));
    assert!(e.contains("already belongs to slot 'hair'"), "{e}");
    let e = call_err(&s, "add_variant", json!({"slot": "head", "variant": "cap", "hides": [{"object": "Body", "part": "scalp"}]}));
    assert!(e.contains("no part 'scalp'"), "{e}");
    let e = call_err(&s, "render", json!({"variants": {"hair": "mohawk"}}));
    assert!(e.contains("mohawk") && e.contains("short"), "{e}");

    // what is shown follows the active variants and hide rules
    let shown = |s: &PollyServer, choice: serde_json::Value| -> std::collections::BTreeMap<String, usize> {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let c: std::collections::BTreeMap<String, String> = serde_json::from_value(choice).unwrap();
        let v = polly_core::variants::variant_view(&g.doc.scene, &c);
        v.objects.values().filter(|o| o.visible).map(|o| (o.name.clone(), o.mesh.faces.len())).collect()
    };
    assert_eq!(shown(&s, json!({})), [("Body".into(), 6), ("HairShort".into(), 6)].into());
    assert_eq!(shown(&s, json!({"head": "helmet"})), [("Body".into(), 6), ("HairShort".into(), 5), ("Helmet".into(), 9)].into());
    call(&s, "set_variant", json!({"slot": "hair", "variant": "long"}));
    assert!(shown(&s, json!({})).contains_key("HairLong"));
    call(&s, "set_default_variant", json!({"slot": "head", "variant": "helmet"}));
    assert_eq!(call(&s, "describe_scene", json!({}))["slots"]["head"]["default"], "helmet");

    // renders and picks use the chosen variants
    let r = s.call("render", json!({"view": "front", "variants": {"head": "helmet", "hair": "short"}}));
    if r.is_error != Some(true) {
        let hit = call(&s, "pick", json!({"x": 256, "y": 120}));
        assert_eq!(hit["object"], "Helmet", "{hit}");
        let v = s.call("render_variants", json!({"slot": "hair", "variants": {"head": "none"}, "view": "front"}));
        assert_ne!(v.is_error, Some(true), "{}", text(&v));
        std::fs::write(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("variants.png"), &images(&v)[0]).unwrap();
    }
    // deleting an object takes it out of its variant
    call(&s, "delete_object", json!({"object": "HairLong"}));
    let slots = call(&s, "list_slots", json!({}));
    assert_eq!(slots["slots"][0]["variants"][1]["objects"], json!([]), "{slots}");
}

/// Phase 4 of gear support: the runtime-swappable GLB.
#[test]
fn runtime_gear_export() {
    use polly_core::glam::{Mat4, Vec3};
    let s = server();
    // Rig away from the origin so wrong spaces would show.
    call(&s, "add_mesh", json!({"name": "Rig", "vertices": [], "faces": [], "position": [2.0, 0.5, -1.0]}));
    call(&s, "set_transform", json!({"object": "Rig", "rotation": [0, 30, 0]}));
    call(&s, "add_bone_chain", json!({"object": "Rig", "points": [[0, -1, 0], [0, 0.5, 0], [0, 1, 0]], "names": ["spine", "head"]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.6, 1.5, 0.4]}, "name": "Body", "position": [2.0, 0.25, -1.0]}));
    call(&s, "bind_to_armature", json!({"object": "Body", "armature": "Rig"}));
    call(&s, "auto_weights", json!({"object": "Body"}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "radius_bottom": 0.5, "radius_top": 0.35, "height": 0.6, "segments": 8, "caps": false}, "name": "Skirt", "position": [2.0, -0.2, -1.0]}));
    call(&s, "transfer_weights", json!({"object": "Skirt", "from": "Body"}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.4, 0.2, 0.4]}, "name": "HairShort", "position": [2.0, 1.15, -1.0]}));
    call(&s, "select", json!({"object": "HairShort", "query": {"by": "normal", "direction": [0, 1, 0]}}));
    call(&s, "set_part", json!({"object": "HairShort", "name": "hair_top"}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.45, 0.7, 0.45]}, "name": "HairLong", "position": [2.0, 0.9, -1.0]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "radius_bottom": 0.3, "radius_top": 0.0, "height": 0.4}, "name": "Helmet", "position": [2.0, 1.35, -1.0]}));
    call(&s, "attach_to_bone", json!({"object": "Helmet", "armature": "Rig", "bone": "head"}));
    for m in ["HairShort", "HairLong"] {
        call(&s, "attach_to_bone", json!({"object": m, "armature": "Rig", "bone": "head"}));
    }
    call(&s, "add_variant", json!({"slot": "hair", "variant": "short", "objects": ["HairShort"]}));
    call(&s, "add_variant", json!({"slot": "hair", "variant": "long", "objects": ["HairLong"]}));
    call(&s, "add_variant", json!({"slot": "head", "variant": "helmet", "objects": ["Helmet"], "hides": [{"object": "HairShort", "part": "hair_top"}]}));
    call(&s, "add_variant", json!({"slot": "head", "variant": "none"}));
    call(&s, "create_clip", json!({"name": "nod", "frames": 12}));
    call(&s, "set_key", json!({"clip": "nod", "frame": 0, "object": "Rig", "bone": "head", "value": [0, 0, 0]}));
    call(&s, "set_key", json!({"clip": "nod", "frame": 12, "object": "Rig", "bone": "head", "value": [30, 0, 0]}));

    // Polly's world positions, for comparison.
    let (world_of, bounds_of) = {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let sc = &g.doc.scene;
        let ids: Vec<(String, u32)> = sc.objects.iter().map(|(i, o)| (o.name.clone(), *i)).collect();
        let world: std::collections::HashMap<String, Mat4> = ids.iter().map(|(n, i)| (n.clone(), sc.world_matrix(*i))).collect();
        let bounds: std::collections::HashMap<String, (Vec3, Vec3)> = ids
            .iter()
            .filter(|(_, i)| !sc.objects[i].mesh.verts.is_empty())
            .map(|(n, i)| {
                let m = sc.world_matrix(*i);
                let pts: Vec<Vec3> = sc.objects[i].mesh.verts.values().map(|v| m.transform_point3(v.pos)).collect();
                (n.clone(), (pts.iter().fold(Vec3::MAX, |a, b| a.min(*b)), pts.iter().fold(Vec3::MIN, |a, b| a.max(*b))))
            })
            .collect();
        (world, bounds)
    };

    let dir = std::env::temp_dir().join(format!("polly-knight-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("knight.glb");
    call(&s, "export_glb", json!({"path": path.to_str().unwrap()}));
    let bytes = std::fs::read(&path).unwrap();
    let (doc, buffers, _) = gltf::import(&path).unwrap();
    let json: Value = {
        let len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        serde_json::from_slice(&bytes[20..20 + len]).unwrap()
    };
    std::fs::remove_dir_all(&dir).ok();

    // Node world matrices (rest pose).
    let mut world = std::collections::HashMap::new();
    let mut parent_name = std::collections::HashMap::new();
    fn walk(n: gltf::Node, parent: (Mat4, String), w: &mut std::collections::HashMap<String, Mat4>, p: &mut std::collections::HashMap<String, String>) {
        let m = parent.0 * Mat4::from_cols_array_2d(&n.transform().matrix());
        let name = n.name().unwrap_or("").to_string();
        w.insert(name.clone(), m);
        p.insert(name.clone(), parent.1.clone());
        for c in n.children() {
            walk(c, (m, name.clone()), w, p);
        }
    }
    for n in doc.scenes().next().unwrap().nodes() {
        walk(n, (Mat4::IDENTITY, String::new()), &mut world, &mut parent_name);
    }

    // 1. One skin, shared by Body and Skirt, which hang under the rig.
    assert_eq!(doc.skins().count(), 1);
    let skinned: Vec<String> = doc.nodes().filter(|n| n.skin().is_some()).map(|n| n.name().unwrap().to_string()).collect();
    assert!(skinned.contains(&"Body".into()) && skinned.contains(&"Skirt".into()), "{skinned:?}");
    assert_eq!(parent_name["Skirt"], "Rig");
    // 2. Rest-pose positions match Polly: skinned vertices are in rig space.
    for name in ["Body", "Skirt"] {
        let node = doc.nodes().find(|n| n.name() == Some(name)).unwrap();
        let mut lo = Vec3::MAX;
        let mut hi = Vec3::MIN;
        for p in node.mesh().unwrap().primitives() {
            for v in p.reader(|b| Some(&buffers[b.index()])).read_positions().unwrap() {
                let w = world_of["Rig"].transform_point3(Vec3::from(v));
                lo = lo.min(w);
                hi = hi.max(w);
            }
        }
        let (plo, phi) = bounds_of[name];
        assert!((lo - plo).length() < 1e-3 && (hi - phi).length() < 1e-3, "{name}: {lo},{hi} vs {plo},{phi}");
    }
    // 3. Bone-attached gear is a child of its joint, at Polly's place.
    assert_eq!(parent_name["Helmet"], "head");
    assert!(world["Helmet"].abs_diff_eq(world_of["Helmet"], 1e-4), "{} vs {}", world["Helmet"], world_of["Helmet"]);
    // 4. Variants: non-defaults hidden and tagged.
    let node_j = |name: &str| json["nodes"].as_array().unwrap().iter().find(|n| n["name"] == name).unwrap().clone();
    assert_eq!(node_j("HairLong")["extensions"]["KHR_node_visibility"]["visible"], false);
    assert_eq!(node_j("HairLong")["extras"]["polly"], json!({"slot": "hair", "variant": "long", "default": false}));
    assert!(node_j("HairShort").get("extensions").is_none());
    assert!(node_j("Helmet").get("extensions").is_none(), "helmet is the default head variant");
    // 5. The hideable part is its own node, hidden by default because the default (helmet) hides it.
    let part = node_j("HairShort.hair_top");
    assert_eq!(part["extensions"]["KHR_node_visibility"]["visible"], false, "{part}");
    assert_eq!(part["extras"]["polly"], json!({"object": "HairShort", "part": "hair_top"}));
    assert_eq!(parent_name["HairShort.hair_top"], "HairShort");
    // 6. Slot table on the scene; 7. extension used, never required.
    let slots = &json["scenes"][0]["extras"]["polly"];
    assert_eq!(slots["version"], 1);
    assert_eq!(slots["slots"]["head"]["default"], "helmet");
    assert_eq!(slots["slots"]["head"]["hides"]["helmet"][0], json!(["HairShort", "hair_top", "HairShort.hair_top"]));
    assert!(json["extensionsUsed"].as_array().unwrap().contains(&json!("KHR_node_visibility")));
    assert!(json.get("extensionsRequired").is_none());
    // The nod animation targets the head joint.
    let anim = doc.animations().next().unwrap();
    assert!(anim.channels().any(|c| c.target().node().name() == Some("head")));
}

/// Reference images: placement, crop, layer, opacity, views, undo.
#[test]
fn reference_images() {
    let s = server();
    let dir = std::env::temp_dir().join(format!("polly-ref-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // 64x32 sheet: red left half, green right half. The crop uses the green half.
    let img = image::RgbaImage::from_fn(64, 32, |x, _| if x < 32 { image::Rgba([255, 0, 0, 255]) } else { image::Rgba([0, 255, 0, 255]) });
    let path = dir.join("sheet.png");
    img.save(&path).unwrap();
    let r = call(&s, "add_reference", json!({"path": path.to_str().unwrap(), "view": "front", "crop": [32, 0, 32, 32]}));
    std::fs::remove_dir_all(&dir).ok();
    // Defaults: 2 units high, standing on the ground, square from the crop.
    assert_eq!(r["corners"]["top_left"], json!([-1.0, 2.0, 0.0]), "{r}");
    assert_eq!(r["corners"]["bottom_right"], json!([1.0, 0.0, 0.0]));
    assert_eq!(r["layer"], "behind");
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.4, 0.4, 0.4]}, "name": "Box", "position": [0, 1, 0]}));
    call(&s, "create_material", json!({"name": "Blue", "color": "#2040ff"}));
    call(&s, "assign_material", json!({"object": "Box", "elements": ["all"], "material": "Blue"}));

    let render = |view: &str| -> Option<image::RgbaImage> {
        let r = s.call("render", json!({"view": view, "width": 260, "height": 260, "overlays": {"wireframe": false, "grid": false, "selection": false}, "legend": false}));
        if r.is_error == Some(true) {
            eprintln!("skipping: {}", text(&r));
            return None;
        }
        Some(image::load_from_memory(&images(&r)[0]).unwrap().to_rgba8())
    };
    let Some(front) = render("front") else { return };
    // The view frames the reference: 2 units * 1.3 padding = 260 px, so 100 px per unit, centred on (0, 1).
    let at = |img: &image::RgbaImage, x: f32, y: f32| *img.get_pixel((130.0 + x * 100.0) as u32, (130.0 - (y - 1.0) * 100.0) as u32);
    let near_corner = at(&front, 0.9, 1.9);
    assert!(near_corner[1] > 100 && near_corner[0] < 80, "half-transparent green behind: {near_corner:?}");
    let outside = at(&front, 1.1, 1.0);
    assert!(outside[1] < 80, "nothing outside the image: {outside:?}");
    let model = at(&front, 0.0, 1.0);
    assert!(model[2] > 150 && model[1] < 120, "the model covers a reference behind it: {model:?}");
    // On top at full opacity it covers the model.
    call(&s, "update_reference", json!({"name": "sheet", "layer": "front", "opacity": 1.0}));
    let front = render("front").unwrap();
    let model = at(&front, 0.0, 1.0);
    assert!(model[1] > 240 && model[2] < 20, "reference over the model: {model:?}");
    // Other views don't show it.
    for v in ["right", "persp"] {
        let img = render(v).unwrap();
        assert!(!img.pixels().any(|p| p[1] > 200 && p[0] < 50 && p[2] < 50), "{v} shows no reference");
    }
    // Undoable; removable; listed.
    call(&s, "undo", json!({}));
    assert_eq!(call(&s, "list_references", json!({}))["references"][0]["layer"], "behind");
    call_err(&s, "update_reference", json!({"name": "sheet", "crop": [60, 0, 10, 10]}));
    call(&s, "remove_reference", json!({"name": "sheet"}));
    assert_eq!(call(&s, "list_references", json!({}))["references"], json!([]));
}

/// Feedback batch 1: mesh files, region select, seam paths/planes, variant-aware baking.
#[test]
fn files_seams_and_baking() {
    let s = server();
    let dir = std::env::temp_dir().join(format!("polly-batch1-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = |name: &str| dir.join(name).to_str().unwrap().to_string();

    // 1. OBJ in, JSON out (with ids, world space), JSON back in.
    std::fs::write(dir.join("quad.obj"), "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\ng panel\nf 1/1 2/2 3/3 4/4\n").unwrap();
    let r = call(&s, "add_mesh", json!({"name": "Quad", "path": p("quad.obj"), "position": [0, 0, 2]}));
    assert_eq!(r["faces"], 1, "{r}");
    call(&s, "export_mesh", json!({"object": "Quad", "path": p("quad.json"), "space": "world"}));
    let j: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("quad.json")).unwrap()).unwrap();
    assert_eq!(j["vertices"][2], json!([1.0, 1.0, 2.0]));
    assert_eq!(j["parts"], json!(["panel"]));
    assert_eq!(j["uvs"][0][0], json!([0.0, 1.0]), "OBJ v is flipped to Polly's downward v");
    assert_eq!(j["vertex_ids"].as_array().unwrap().len(), 4);
    call(&s, "add_mesh", json!({"name": "Quad2", "path": p("quad.json")}));
    call(&s, "export_mesh", json!({"object": "Quad2", "path": p("quad2.obj")}));
    assert!(std::fs::read_to_string(dir.join("quad2.obj")).unwrap().contains("g panel"));
    call_err(&s, "add_mesh", json!({"name": "X", "path": p("quad.json"), "vertices": [[0, 0, 0]]}));

    // 2. A box selects the vertices inside it, not every vertex of faces centred in it.
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [2, 2, 2]}, "name": "Cube"}));
    let r = call(&s, "select", json!({"object": "Cube", "query": {"by": "box", "min": [0.9, -0.5, -2], "max": [2, 2, 2]}, "kind": "verts"}));
    assert_eq!(r["selection"]["counts"]["verts"], 2, "{r}");
    let sel_verts = |s: &polly_mcp::PollyServer| {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let o = g.doc.scene.objects.values().find(|o| o.name == "Cube").unwrap();
        o.selection.verts.iter().map(|v| o.mesh.verts[v].pos).collect::<Vec<_>>()
    };
    let vs = sel_verts(&s);
    assert_eq!(vs.len(), 2, "{vs:?}");
    assert!(vs.iter().all(|p| p.x > 0.9 && p.y > 0.9));

    // 3. Seam along the shortest path between opposite cube corners: 3 edges.
    call(&s, "export_mesh", json!({"object": "Cube", "path": p("cube.json")}));
    let c: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("cube.json")).unwrap()).unwrap();
    let id_at = |want: [f64; 3]| {
        let i = c["vertices"].as_array().unwrap().iter().position(|v| (0..3).all(|k| (v[k].as_f64().unwrap() - want[k]).abs() < 1e-5)).unwrap();
        format!("v{}", c["vertex_ids"][i])
    };
    let r = call(&s, "seam_path", json!({"object": "Cube", "verts": [id_at([-1.0, -1.0, -1.0]), id_at([1.0, 1.0, 1.0])]}));
    assert_eq!(r["seam_edges"], 3, "{r}");

    // 4. Plane seam down the back of an open tube.
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "segments": 8, "caps": false}, "name": "Tube"}));
    let r = call(&s, "seam_plane", json!({"object": "Tube", "point": [0, 0, 0], "normal": [1, 0, 0], "within": {"point": [0, 0, 0], "normal": [0, 0, -1]}}));
    assert_eq!(r["seam_edges"], 1, "{r}");
    let r = call(&s, "seam_plane", json!({"object": "Tube", "point": [0, 0, 0], "normal": [1, 0, 0]}));
    assert_eq!(r["seam_edges"], 2, "front and back: {r}");

    // 5. Baking: "scene" occluders use the default loadout; swappable occluders warn.
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.6, 1.0, 0.4]}, "name": "Body", "position": [5, 0, 0]}));
    call(&s, "auto_seams", json!({"object": "Body"}));
    call(&s, "unwrap", json!({"object": "Body"}));
    call(&s, "create_texture", json!({"name": "BodyTex", "width": 32, "height": 32, "fill": "#c0a080", "material": "BodyMat", "object": "Body"}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.8, 0.6, 0.6]}, "name": "Blazer", "position": [5, 0.1, 0]}));
    call(&s, "add_variant", json!({"slot": "top", "variant": "none"}));
    call(&s, "add_variant", json!({"slot": "top", "variant": "blazer", "objects": ["Blazer"]}));
    call(&s, "set_variant", json!({"slot": "top", "variant": "blazer"}));
    let r = call(&s, "bake_shading", json!({"object": "Body", "ao": {"samples": 4}, "occluders": "scene"}));
    let occ = r["occluders"].to_string();
    assert!(!occ.contains("Blazer") && r.get("warning").is_none(), "default loadout has no blazer, even though it is active: {r}");
    let r = call(&s, "bake_shading", json!({"object": "Body", "ao": {"samples": 4}, "occluders": "scene", "variants": {"top": "blazer"}}));
    assert!(r["occluders"].to_string().contains("Blazer") && r["warning"].as_str().unwrap().contains("swappable"), "{r}");
    let r = call(&s, "bake_shading", json!({"object": "Body", "ao": {"samples": 4}, "occluders": ["Tube"]}));
    assert_eq!(r["occluders"], json!(["Tube"]));
    call_err(&s, "bake_shading", json!({"object": "Body", "ao": {}, "occluders": "self", "variants": {"top": "none"}}));
    std::fs::remove_dir_all(&dir).ok();
}
