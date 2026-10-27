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
    assert_eq!(node_j("HairLong")["extras"]["polly"], json!({"slot": "hair", "variants": ["long"], "default": false}));
    assert!(node_j("HairShort").get("extensions").is_none());
    assert!(node_j("Helmet").get("extensions").is_none(), "helmet is the default head variant");
    // 5. The hideable part is its own node, hidden by default because the default (helmet) hides it.
    let part = node_j("HairShort.hair_top");
    assert_eq!(part["extensions"]["KHR_node_visibility"]["visible"], false, "{part}");
    assert_eq!(part["extras"]["polly"], json!({"object": "HairShort", "parts": ["hair_top"]}));
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

/// Feedback batch 2: smooth clips, bone axes, mirrored poses and keys.
#[test]
fn animation_batch() {
    let s = server();
    call(&s, "add_mesh", json!({"name": "Rig", "vertices": [], "faces": []}));
    call(&s, "add_bone", json!({"object": "Rig", "name": "spine", "head": [0, 0.8, 0], "tail": [0, 1.4, 0]}));
    call(&s, "add_bone", json!({"object": "Rig", "name": "arm.L", "head": [0.2, 1.3, 0], "tail": [0.8, 1.3, 0], "parent": "spine", "roll": 25}));
    call(&s, "add_bone", json!({"object": "Rig", "name": "arm.R", "head": [-0.2, 1.3, 0], "tail": [-0.8, 1.3, 0], "parent": "spine", "roll": -60}));

    // Axes in world terms, and which way a positive turn moves the tail.
    let bones = call(&s, "list_bones", json!({"object": "Rig"}));
    let arm = bones["bones"].as_array().unwrap().iter().find(|b| b["name"] == "arm.L").unwrap().clone();
    assert_eq!(arm["axes"]["y"], "+X", "{arm}");
    // Check the reported direction against an actual posed render-free computation.
    let tail_after = |s: &polly_mcp::PollyServer, bone: &str| {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let shown = polly_core::anim::pose_scene(&g.doc.scene, &[]);
        let o = shown.objects.values().find(|o| o.name == "Rig").unwrap();
        let sk = o.skeleton.as_ref().unwrap();
        sk.bones[&sk.find(bone).unwrap()].tail
    };
    call(&s, "pose_bone", json!({"object": "Rig", "bone": "arm.L", "rotation": [10, 0, 0]}));
    let moved = tail_after(&s, "arm.L") - polly_core::glam::Vec3::new(0.8, 1.3, 0.0);
    // An axis name, or a vector when the roll puts it between axes (here 25 degrees).
    let v = &arm["axes"]["positive_x_turn_moves_tail"];
    let dir = match v.as_str() {
        Some("+Y") => polly_core::glam::Vec3::Y,
        Some("-Y") => polly_core::glam::Vec3::NEG_Y,
        Some("+Z") => polly_core::glam::Vec3::Z,
        Some("-Z") => polly_core::glam::Vec3::NEG_Z,
        Some(d) => panic!("{d}"),
        None => polly_core::glam::Vec3::new(v[0].as_f64().unwrap() as f32, v[1].as_f64().unwrap() as f32, v[2].as_f64().unwrap() as f32),
    };
    assert!(moved.normalize().dot(dir) > 0.95, "reported {dir}, moved {moved}");

    // Mirror the pose: arm.R's tail ends up at the mirror image of arm.L's.
    call(&s, "mirror_pose", json!({"object": "Rig", "mode": "left_to_right"}));
    let (l, r) = (tail_after(&s, "arm.L"), tail_after(&s, "arm.R"));
    assert!(l.abs_diff_eq(r * polly_core::glam::Vec3::new(-1.0, 1.0, 1.0), 1e-4), "{l} vs {r}");
    call(&s, "mirror_pose", json!({"object": "Rig"}));
    assert!(tail_after(&s, "arm.L").abs_diff_eq(l, 1e-4), "flip keeps a symmetric pose symmetric");

    // Clips are smooth by default; copy_keys mirrors one step into the next.
    let c = call(&s, "create_clip", json!({"name": "walk", "frames": 24}));
    assert!(c.to_string().contains("walk"));
    assert_eq!(call(&s, "list_clips", json!({}))["clips"][0]["interpolation"], "smooth");
    call(&s, "clear_pose", json!({"object": "Rig"}));
    call(&s, "set_key", json!({"clip": "walk", "frame": 0, "object": "Rig", "bone": "arm.L", "value": [30, 0, 0]}));
    call(&s, "set_key", json!({"clip": "walk", "frame": 0, "object": "Rig", "bone": "arm.R", "value": [-30, 0, 0]}));
    let r = call(&s, "copy_keys", json!({"clip": "walk", "object": "Rig", "frames": [0, 0], "to_frame": 12, "mirror": true}));
    assert_eq!(r["keys_written"], 2, "{r}");
    call(&s, "copy_keys", json!({"clip": "walk", "object": "Rig", "frames": [0, 0], "to_frame": 24}));
    let d = call(&s, "describe_clip", json!({"clip": "walk"}));
    assert!(d["channels"].as_array().unwrap().iter().all(|c| c["interpolation"] == "smooth" && c["frames"] == json!([0.0, 12.0, 24.0])), "{d}");
    // At frame 12 arm.R holds arm.L's frame-0 swing, mirrored.
    let tails_at = |frame: f32| {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let sc = &g.doc.scene;
        let cid = *sc.clips.keys().next().unwrap();
        let shown = polly_core::anim::pose_scene(sc, &[polly_core::anim::ClipPlay { clip: cid, time: frame / 24.0, weight: 1.0 }]);
        let o = shown.objects.values().find(|o| o.name == "Rig").unwrap();
        let sk = o.skeleton.as_ref().unwrap();
        (sk.bones[&sk.find("arm.L").unwrap()].tail, sk.bones[&sk.find("arm.R").unwrap()].tail)
    };
    let ((l0, _), (_, r12)) = (tails_at(0.0), tails_at(12.0));
    assert!(l0.abs_diff_eq(r12 * polly_core::glam::Vec3::new(-1.0, 1.0, 1.0), 1e-4), "{l0} vs {r12}");
    // The bone_axes overlay renders (GPU permitting).
    let r = s.call("render", json!({"view": "front", "overlays": {"bones": true, "bone_axes": true}}));
    if r.is_error != Some(true) {
        assert_eq!(images(&r).len(), 1);
    }
}

/// Feedback batch 3: clothes from the body, auto-hidden skin, shared variants.
#[test]
fn clothing_batch() {
    let s = server();
    call(&s, "add_mesh", json!({"name": "Rig", "vertices": [], "faces": []}));
    call(&s, "add_bone_chain", json!({"object": "Rig", "points": [[0, -1, 0], [0, 0, 0], [0, 1, 0]], "names": ["hips", "chest"]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "segments": 8, "height_segments": 4, "height": 2.0, "radius_bottom": 0.4, "radius_top": 0.4}, "name": "Body"}));
    call(&s, "bind_to_armature", json!({"object": "Body", "armature": "Rig"}));
    call(&s, "auto_weights", json!({"object": "Body"}));
    // Blouse: the upper half of the sides; blazer: from y = -0.5 up, further out.
    call(&s, "select", json!({"object": "Body", "query": {"by": "box", "min": [-1, 0, -1], "max": [1, 0.99, 1]}, "kind": "faces"}));
    let r = call(&s, "shell", json!({"object": "Body", "name": "Blouse", "offset": 0.02}));
    assert_eq!(r["from_faces"], 16, "{r}");
    assert_eq!(r["skinned_to"], "Rig", "the shell is rigged like the body: {r}");
    call(&s, "select", json!({"object": "Body", "query": {"by": "box", "min": [-1, -0.5, -1], "max": [1, 0.99, 1]}, "kind": "faces"}));
    call(&s, "shell", json!({"object": "Body", "name": "Blazer", "offset": 0.04}));
    call(&s, "inflate", json!({"object": "Blazer", "elements": ["all"], "distance": 0.005, "keep_plane": {"point": [0, 0, 0], "normal": [1, 0, 0]}}));
    // One object in several variants of a slot.
    call(&s, "add_variant", json!({"slot": "top", "variant": "suit", "objects": ["Blouse", "Blazer"]}));
    call(&s, "add_variant", json!({"slot": "top", "variant": "blouse", "objects": ["Blouse"]}));
    call(&s, "add_variant", json!({"slot": "top", "variant": "blazer", "objects": ["Blazer"]}));
    call(&s, "add_variant", json!({"slot": "top", "variant": "none"}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}, "name": "Hat"}));
    call(&s, "add_variant", json!({"slot": "head", "variant": "hat", "objects": ["Hat"]}));
    call_err(&s, "add_variant", json!({"slot": "head", "variant": "blouse_hat", "objects": ["Blouse"]}));
    // Covered skin.
    let r = call(&s, "auto_hide", json!({"garment": "Blouse", "body": "Body"}));
    assert_eq!(r["hidden_faces"], 16, "{r}");
    let r = call(&s, "auto_hide", json!({"garment": "Blazer", "body": "Body"}));
    assert_eq!(r["hidden_faces"], 24, "{r}");
    let visible_faces = |choice: Value| {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let c: std::collections::BTreeMap<String, String> = serde_json::from_value(choice).unwrap();
        let v = polly_core::variants::variant_view(&g.doc.scene, &c);
        let o = v.objects.values().find(|o| o.name == "Body").unwrap();
        let shown = |n: &str| v.objects.values().find(|o| o.name == n).unwrap().visible;
        (o.mesh.faces.len(), shown("Blouse"), shown("Blazer"))
    };
    // 8 sides x 4 rows + 2 caps = 34 faces.
    assert_eq!(visible_faces(json!({"top": "none"})), (34, false, false));
    assert_eq!(visible_faces(json!({"top": "blouse"})), (34 - 16, true, false));
    assert_eq!(visible_faces(json!({"top": "blazer"})), (34 - 24, false, true));
    assert_eq!(visible_faces(json!({"top": "suit"})), (34 - 24, true, true));
    // Re-adding a variant keeps its auto_hide rules.
    call(&s, "add_variant", json!({"slot": "top", "variant": "blouse", "objects": ["Blouse"]}));
    assert_eq!(visible_faces(json!({"top": "blouse"})).0, 34 - 16);

    // Export: the body's hidden faces split by which garments cover them.
    let dir = std::env::temp_dir().join(format!("polly-clothes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("clothes.glb");
    call(&s, "export_glb", json!({"path": path.to_str().unwrap()}));
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let j: Value = serde_json::from_slice(&bytes[20..20 + len]).unwrap();
    let node = |name: &str| j["nodes"].as_array().unwrap().iter().find(|n| n["name"] == name).cloned().unwrap_or_else(|| panic!("no node {name}"));
    let both = node("Body.under_Blazer+under_Blouse");
    assert_eq!(both["extras"]["polly"]["parts"], json!(["under_Blazer", "under_Blouse"]));
    assert!(node("Body.under_Blazer").get("extras").is_some());
    // Default "suit" shows both garments and hides everything they cover.
    assert_eq!(node("Blouse")["extras"]["polly"], json!({"slot": "top", "variants": ["suit", "blouse"], "default": true}));
    assert!(node("Blouse").get("extensions").is_none());
    assert_eq!(both["extensions"]["KHR_node_visibility"]["visible"], false);
    let hides = &j["scenes"][0]["extras"]["polly"]["slots"]["top"]["hides"];
    let nodes_of = |v: &str| hides[v].as_array().unwrap().iter().map(|e| e[2].as_str().unwrap().to_string()).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(nodes_of("blouse"), ["Body.under_Blazer+under_Blouse".to_string()].into());
    assert_eq!(nodes_of("blazer"), ["Body.under_Blazer".to_string(), "Body.under_Blazer+under_Blouse".to_string()].into());
}

/// Back/left/bottom references: each shows only in its own view, reading as drawn.
#[test]
fn opposite_view_references() {
    let s = server();
    let dir = std::env::temp_dir().join(format!("polly-ref6-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Red left half, green right half (as drawn).
    let img = image::RgbaImage::from_fn(64, 32, |x, _| if x < 32 { image::Rgba([255, 0, 0, 255]) } else { image::Rgba([0, 255, 0, 255]) });
    let path = dir.join("lr.png");
    img.save(&path).unwrap();
    for v in ["back", "left", "bottom"] {
        call(&s, "add_reference", json!({"path": path.to_str().unwrap(), "view": v, "name": v, "opacity": 1.0}));
    }
    std::fs::remove_dir_all(&dir).ok();
    let render = |view: &str| -> Option<image::RgbaImage> {
        let r = s.call("render", json!({"view": view, "width": 200, "height": 200, "overlays": {"wireframe": false, "grid": false}, "legend": false}));
        if r.is_error == Some(true) {
            eprintln!("skipping: {}", text(&r));
            return None;
        }
        Some(image::load_from_memory(&images(&r)[0]).unwrap().to_rgba8())
    };
    let Some(_) = render("back") else { return };
    for v in ["back", "left", "bottom"] {
        let img = render(v).unwrap();
        // Red on the screen's left, green on its right: not mirrored.
        let (l, r) = (img.get_pixel(60, 100), img.get_pixel(140, 100));
        assert!(l[0] > 200 && l[1] < 50 && r[1] > 200 && r[0] < 50, "{v}: {l:?} {r:?}");
    }
    // Only one reference shows per view: the front view has none.
    let front = render("front").unwrap();
    assert!(!front.pixels().any(|p| p[0] > 200 && p[1] < 50 && p[2] < 50), "front shows no reference");
}

/// paint_line: world-unit lines on the surface, from points or render pixels.
#[test]
fn surface_lines() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "uv_sphere", "radius": 0.5, "segments": 16, "rings": 10}, "name": "Ball"}));
    call(&s, "auto_seams", json!({}));
    call(&s, "unwrap", json!({"texture_size": [128, 128]}));
    call(&s, "create_texture", json!({"name": "T", "width": 128, "height": 128, "fill": "#000000", "material": "M", "object": "Ball"}));
    let solid = call(&s, "paint_line", json!({"object": "Ball", "points": [[0.5, 0, 0], [0, 0, 0.5], [-0.5, 0, 0], [0, 0, -0.5]], "closed": true, "width": 0.05, "color": "#ffffff"}));
    let n = solid["pixels_changed"].as_u64().unwrap();
    assert!(n > 100, "{solid}");
    assert!((solid["length"].as_f64().unwrap() - 4.0 * 0.5 * 2f64.sqrt()).abs() < 0.01, "waypoint polyline length: {solid}");
    call(&s, "undo", json!({}));
    let dashed = call(&s, "paint_line", json!({"object": "Ball", "points": [[0.5, 0, 0], [0, 0, 0.5], [-0.5, 0, 0], [0, 0, -0.5]], "closed": true, "width": 0.05, "color": "#ffffff", "dashes": [0.05, 0.05]}));
    let d = dashed["pixels_changed"].as_u64().unwrap();
    assert!(d > n / 4 && d < n * 3 / 4, "dashes paint about half: {d} of {n}");
    call_err(&s, "paint_line", json!({"object": "Ball", "points": [[0, 0, 0.5]], "width": 0.05, "color": "#ffffff"}));
    // From render pixels across the middle of a front view.
    let r = s.call("render", json!({"view": "front"}));
    if r.is_error == Some(true) {
        return;
    }
    let px = call(&s, "paint_line", json!({"pixels": [[180, 256], [330, 256]], "width": 0.04, "color": "#ff0000"}));
    assert!(px["pixels_changed"].as_u64().unwrap() > 20, "{px}");
}

/// Two-bone IK: reaching a world target, and foot locking baked into a clip.
#[test]
fn two_bone_ik_tools() {
    use polly_core::glam::Vec3;
    let s = server();
    call(&s, "add_mesh", json!({"name": "Rig", "vertices": [], "faces": [], "position": [2, 0, 0]}));
    call(&s, "add_bone", json!({"object": "Rig", "name": "hips", "head": [0, 1, 0], "tail": [0, 1.2, 0]}));
    call(&s, "add_bone", json!({"object": "Rig", "name": "thigh.L", "head": [0.1, 1, 0], "tail": [0.1, 0.55, 0.02], "parent": "hips"}));
    call(&s, "add_bone", json!({"object": "Rig", "name": "shin.L", "head": [0.1, 0.55, 0.02], "tail": [0.1, 0.1, 0], "parent": "thigh.L"}));
    // World position of the foot (shin tail) at a clip time (None = static pose).
    let foot = |s: &polly_mcp::PollyServer, t: Option<f32>| {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let sc = &g.doc.scene;
        let plays: Vec<_> = t.map(|t| polly_core::anim::ClipPlay { clip: *sc.clips.keys().next().unwrap(), time: t, weight: 1.0 }).into_iter().collect();
        let shown = polly_core::anim::pose_scene(sc, &plays);
        let (id, o) = shown.objects.iter().find(|(_, o)| o.name == "Rig").unwrap();
        let sk = o.skeleton.as_ref().unwrap();
        shown.world_matrix(*id).transform_point3(sk.bones[&sk.find("shin.L").unwrap()].tail)
    };
    let target = Vec3::new(2.25, 0.4, 0.2);
    let r = call(&s, "solve_ik", json!({"object": "Rig", "bone": "shin.L", "target": target.to_array(), "pole": [2.1, 0.6, 1.0]}));
    assert_eq!(r["reached"], true, "{r}");
    assert!(foot(&s, None).distance(target) < 1e-3, "{}", foot(&s, None));
    call_err(&s, "solve_ik", json!({"object": "Rig", "bone": "hips", "target": [0, 0, 0]}));

    // The hips move forward through a step; the planted foot must not slide.
    call(&s, "clear_pose", json!({"object": "Rig"}));
    call(&s, "create_clip", json!({"name": "step", "frames": 12}));
    call(&s, "set_key", json!({"clip": "step", "frame": 0, "object": "Rig", "bone": "hips", "property": "translation", "value": [0, 0, -0.15]}));
    call(&s, "set_key", json!({"clip": "step", "frame": 12, "object": "Rig", "bone": "hips", "property": "translation", "value": [0, 0, 0.15]}));
    let before: Vec<Vec3> = [0.0, 0.25, 0.5].iter().map(|t| foot(&s, Some(*t))).collect();
    assert!(before[0].distance(before[2]) > 0.25, "without IK the foot slides with the hips");
    // A straight leg can't reach once the hips move away: warned.
    let r = call(&s, "bake_ik", json!({"clip": "step", "object": "Rig", "bone": "shin.L", "frames": [0, 12], "lock": true}));
    assert_eq!(r["keys_written"], 26, "{r}");
    assert!(r["warning"].as_str().unwrap().contains("out of reach"), "{r}");
    // Lower the hips so the knee bends, plant the foot, then lock it there.
    call(&s, "set_key", json!({"clip": "step", "frame": 0, "object": "Rig", "bone": "hips", "property": "translation", "value": [0, -0.2, -0.15]}));
    call(&s, "set_key", json!({"clip": "step", "frame": 12, "object": "Rig", "bone": "hips", "property": "translation", "value": [0, -0.2, 0.15]}));
    let r = call(&s, "bake_ik", json!({"clip": "step", "object": "Rig", "bone": "shin.L", "frames": [0, 12], "target": [2.1, 0.0, 0.0], "pole": [2.1, 0.5, 1.0]}));
    assert!(r.get("warning").is_none(), "{r}");
    let planted = foot(&s, Some(0.0));
    assert!(planted.distance(Vec3::new(2.1, 0.0, 0.0)) < 1e-3, "{planted}");
    let r = call(&s, "bake_ik", json!({"clip": "step", "object": "Rig", "bone": "shin.L", "frames": [0, 12], "lock": true}));
    assert!(r.get("warning").is_none(), "{r}");
    let before = vec![planted];
    for f in [0.0f32, 3.0, 6.0, 9.0, 12.0] {
        let p = foot(&s, Some(f / 24.0));
        assert!(p.distance(before[0]) < 2e-3, "frame {f}: foot at {p}, locked at {}", before[0]);
    }
}

/// Reference measuring: two-point calibration, flip, measure and compare.
#[test]
fn reference_measure_and_compare() {
    let s = server();
    let dir = std::env::temp_dir().join(format!("polly-refm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // White sheet; a black figure 40 px wide (columns 80..120) from row 40 to 360, and a
    // mark to its right (columns 150..170, rows 100..120).
    let img = image::RgbaImage::from_fn(200, 400, |x, y| {
        let body = (80..120).contains(&x) && (40..360).contains(&y);
        let mark = (150..170).contains(&x) && (100..120).contains(&y);
        if body || mark { image::Rgba([20, 20, 30, 255]) } else { image::Rgba([250, 248, 240, 255]) }
    });
    let path = dir.join("sheet.png");
    img.save(&path).unwrap();
    // Row 360 is the ground and row 40 is 1.6 m up, both on column 100 (x = 0).
    let r = call(&s, "add_reference", json!({"path": path.to_str().unwrap(), "view": "front", "name": "sheet",
        "calibrate": [{"pixel": [100, 360], "at": [0, 0]}, {"pixel": [100, 40], "at": [0, 1.6]}]}));
    assert!((r["world_per_pixel"].as_f64().unwrap() - 0.005).abs() < 1e-6, "{r}");
    std::fs::remove_dir_all(&dir).ok();

    let m = call(&s, "measure_reference", json!({"name": "sheet", "heights": [0.5, 1.25]}));
    assert_eq!(m["drawing"]["y_range"], json!([0.0, 1.6]), "{m}");
    assert_eq!(m["rows"][0]["x"], json!([-0.1, 0.1]), "{m}");
    assert_eq!(m["rows"][1]["x"], json!([-0.1, 0.35]), "the mark on the right: {m}");
    call(&s, "update_reference", json!({"name": "sheet", "flip": true}));
    let m = call(&s, "measure_reference", json!({"name": "sheet", "heights": [1.25]}));
    // Flipping mirrors about the image centre (column 100 is the centre, x = 0).
    assert_eq!(m["rows"][0]["x"], json!([-0.35, 0.1]), "flipped: {m}");
    call(&s, "update_reference", json!({"name": "sheet", "flip": false}));

    // A 0.3 m wide model: 5 cm too wide on each side where only the figure is drawn.
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.3, 1.6, 0.3]}, "name": "Body", "position": [0, 0.8, 0]}));
    let r = s.call("compare_reference", json!({"name": "sheet", "heights": [0.5, 1.25, 1.7]}));
    assert_ne!(r.is_error, Some(true), "{}", text(&r));
    assert_eq!(images(&r).len(), 1);
    let c: Value = serde_json::from_str(&text(&r)).unwrap();
    assert_eq!(c["rows"][0]["sticks_out"], json!({"-x": 0.05, "+x": 0.05}), "{c}");
    assert_eq!(c["rows"][1]["sticks_out"]["+x"], json!(-0.2), "the model falls short of the mark: {c}");
    assert!(c["rows"][2].get("sticks_out").is_none(), "above both");
    // Region render with the reference on top, at full opacity, this render only.
    let r = s.call("render", json!({"view": "front", "region": [[-0.2, 1.1, -0.2], [0.4, 1.4, 0.2]], "reference_opacity": 1.0, "reference_layer": "front", "width": 200, "height": 200, "overlays": {"grid": false}}));
    if r.is_error != Some(true) {
        let img = image::load_from_memory(&images(&r)[0]).unwrap().to_rgba8();
        // The close-up is filled by the sheet drawn on top (white paper or dark ink, not the grey model).
        let p = img.get_pixel(100, 100);
        assert!(p[0] > 230 || p[0] < 40, "{p:?}");
    }
    assert_eq!(call(&s, "list_references", json!({}))["references"][0]["opacity"], 0.5, "the scene's own opacity is unchanged");
}

/// Editing batch: replace_geometry, set_positions, mirrored edits, new queries, warnings.
#[test]
fn editing_batch() {
    let s = server();
    let dir = std::env::temp_dir().join(format!("polly-edit2-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = |n: &str| dir.join(n).to_str().unwrap().to_string();
    let mesh = |s: &polly_mcp::PollyServer, name: &str| {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        g.doc.scene.objects.values().find(|o| o.name == name).unwrap().mesh.clone()
    };

    // --- replace_geometry keeps the object's setup ---
    call(&s, "add_mesh", json!({"name": "Rig", "vertices": [], "faces": []}));
    call(&s, "add_bone_chain", json!({"object": "Rig", "points": [[0, 0, 0], [0, 1, 0], [0, 2, 0]], "names": ["low", "high"]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "segments": 8, "height_segments": 4, "height": 2.0}, "name": "Body", "position": [0, 1, 0]}));
    call(&s, "create_material", json!({"name": "Skin", "color": "#e0a080"}));
    call(&s, "assign_material", json!({"object": "Body", "elements": ["all"], "material": "Skin"}));
    call(&s, "bind_to_armature", json!({"object": "Body", "armature": "Rig"}));
    call(&s, "auto_weights", json!({"object": "Body"}));
    call(&s, "add_variant", json!({"slot": "body", "variant": "base", "objects": ["Body"]}));
    // Same layout: export, scale the x coordinates in the "script", replace. Ids survive.
    call(&s, "export_mesh", json!({"object": "Body", "path": p("body.json")}));
    let mut j: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("body.json")).unwrap()).unwrap();
    for v in j["vertices"].as_array_mut().unwrap() {
        v[0] = json!(v[0].as_f64().unwrap() * 1.5);
    }
    std::fs::write(dir.join("body.json"), j.to_string()).unwrap();
    let before = mesh(&s, "Body");
    let r = call(&s, "replace_geometry", json!({"object": "Body", "path": p("body.json")}));
    assert_eq!(r["kept_ids"], true, "{r}");
    let after = mesh(&s, "Body");
    assert_eq!(before.verts.keys().collect::<Vec<_>>(), after.verts.keys().collect::<Vec<_>>());
    assert_eq!(before.weights, after.weights);
    assert!(r.to_string().contains("body: base"), "slot kept: {r}");
    // Different layout: a fresh mesh, weights transferred, the main material kept.
    call(&s, "add_primitive", json!({"shape": {"kind": "cylinder", "segments": 12, "height_segments": 6, "height": 2.0}, "name": "Tmp", "position": [0, 1, 0]}));
    call(&s, "export_mesh", json!({"object": "Tmp", "path": p("tmp.obj")}));
    call(&s, "delete_object", json!({"object": "Tmp"}));
    let r = call(&s, "replace_geometry", json!({"object": "Body", "path": p("tmp.obj")}));
    assert_eq!(r["kept_ids"], false);
    assert!(r["notes"].to_string().contains("weights transferred"), "{r}");
    let m = mesh(&s, "Body");
    assert_eq!(m.faces.len(), 12 * 6 + 2);
    assert!(!m.weights.is_empty());
    assert!(m.faces.values().all(|f| f.material == 1), "the main material");
    // delete_object says what it removed from slots.
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}, "name": "Hat"}));
    call(&s, "add_variant", json!({"slot": "head", "variant": "hat", "objects": ["Hat"]}));
    let r = call(&s, "delete_object", json!({"object": "Hat"}));
    assert!(r["note"].as_str().unwrap().contains("slot 'head'"), "{r}");

    // --- set_positions and mirrored edits ---
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [2, 2, 2]}, "name": "Box"}));
    let pos = |s: &polly_mcp::PollyServer, at: [f32; 3]| {
        let m = mesh(s, "Box");
        *m.verts.iter().find(|(_, v)| v.pos.distance(polly_core::glam::Vec3::from(at)) < 1e-4).map(|(i, _)| i).unwrap_or_else(|| panic!("no vertex at {at:?}"))
    };
    let (a, b) = (pos(&s, [1.0, 1.0, 1.0]), pos(&s, [-1.0, 1.0, 1.0]));
    let r = call(&s, "move", json!({"object": "Box", "elements": [format!("v{a}")], "offset": [0.5, 0.2, 0], "mirror": true}));
    assert_eq!(r["mirror"]["partners_moved"], 1, "{r}");
    pos(&s, [1.5, 1.2, 1.0]);
    pos(&s, [-1.5, 1.2, 1.0]);
    call(&s, "set_positions", json!({"object": "Box", "positions": {format!("v{a}"): [1.1, 1.0, 1.0]}, "mirror": true}));
    pos(&s, [-1.1, 1.0, 1.0]);
    assert_eq!(pos(&s, [-1.1, 1.0, 1.0]), b);
    std::fs::write(dir.join("pos.json"), json!({"vertex_ids": [a], "vertices": [[1.0, 1.0, 1.0]]}).to_string()).unwrap();
    call(&s, "set_positions", json!({"object": "Box", "path": p("pos.json"), "mirror": true}));
    pos(&s, [-1.0, 1.0, 1.0]);
    call_err(&s, "set_positions", json!({"object": "Box", "positions": {"v99999": [0, 0, 0]}}));

    // --- queries ---
    let r = call(&s, "select", json!({"object": "Box", "query": {"by": "facing", "direction": [0, 1, 0], "max_angle": 60}}));
    assert_eq!(r["selection"]["counts"]["verts"], 4, "the top corners (normals tilt 54.7 deg): {r}");
    let r = call(&s, "select", json!({"object": "Box", "query": {"by": "outline", "view": [0, 0, -1]}}));
    assert_eq!(r["selection"]["counts"]["edges"], 4, "a cube's front silhouette is the 4 side edges: {r}");

    // --- warnings: only new problems ---
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}, "name": "Warp"}));
    let w = mesh(&s, "Warp");
    let v = *w.verts.iter().find(|(_, v)| v.pos.x > 0.4 && v.pos.y > 0.4 && v.pos.z > 0.4).unwrap().0;
    let r = call(&s, "move", json!({"object": "Warp", "elements": [format!("v{v}")], "offset": [0, 0.4, 0]}));
    assert!(r["warnings"].to_string().contains("non-planar"), "{r}");
    let r = call(&s, "move", json!({"object": "Warp", "elements": ["all"], "offset": [0, 0.1, 0]}));
    assert!(r.get("warnings").is_none() && r["earlier_warnings"].is_string(), "old problems are not repeated: {r}");

    // --- the render explains the orange tint ---
    let r = s.call("render", json!({"view": "front"}));
    if r.is_error != Some(true) {
        let j: Value = serde_json::from_str(&text(&r)).unwrap();
        assert!(j["selection"].as_str().unwrap().contains("Warp"), "{j}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Shape tools: loft, bridge_loops (different sizes), intersections, push_out.
#[test]
fn loft_bridge_and_collisions() {
    let s = server();
    let ring = |n: usize, r: f32, y: f32| -> Vec<[f32; 3]> {
        (0..n).map(|i| {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            [r * t.cos(), y, r * t.sin()]
        }).collect()
    };
    // A torso tube (8 around) and a neck tube (5 around), joined.
    let r = call(&s, "loft", json!({"name": "Torso", "sections": [ring(8, 0.4, 0.0), ring(8, 0.45, 0.5), ring(8, 0.35, 1.0)]}));
    assert_eq!(r["faces"], 16, "{r}");
    call(&s, "loft", json!({"name": "Neck", "sections": [ring(5, 0.12, 1.2), ring(5, 0.11, 1.4)]}));
    call(&s, "join_objects", json!({"objects": ["Torso", "Neck"]}));
    call(&s, "rename_object", json!({"object": "Torso", "name": "Figure"}));
    let border_edge = |y: f32| {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        let m = &g.doc.scene.objects.values().find(|o| o.name == "Figure").unwrap().mesh;
        let ef = m.edge_faces();
        let e = ef.iter().find(|(e, f)| f.len() == 1 && (m.verts[&e.0].pos.y - y).abs() < 1e-4 && (m.verts[&e.1].pos.y - y).abs() < 1e-4).unwrap().0;
        format!("e{}-{}", e.0, e.1)
    };
    let r = call(&s, "bridge_loops", json!({"object": "Figure", "a": [border_edge(1.0)], "b": [border_edge(1.2)]}));
    assert_eq!(r["result"]["counts"]["faces"], 13, "8 + 5 triangles: {r}");
    assert!(r.get("warnings").is_none(), "consistent winding, no new problems: {r}");
    call_err(&s, "bridge_loops", json!({"object": "Figure", "a": ["v1", "v2"], "b": ["v3", "v4", "v5"]}));

    // Hair poking into a head.
    call(&s, "add_primitive", json!({"shape": {"kind": "uv_sphere", "radius": 0.5, "segments": 12, "rings": 8}, "name": "Head", "position": [3, 1, 0]}));
    call(&s, "add_primitive", json!({"shape": {"kind": "uv_sphere", "radius": 0.5, "segments": 12, "rings": 8}, "name": "Hair", "position": [3, 1.08, 0]}));
    let r = call(&s, "intersections", json!({"object": "Hair", "select": true}));
    let hit = &r["intersections"][0];
    assert_eq!(hit["with"], "Head", "{r}");
    assert!(hit["crossings"].as_u64().unwrap() > 0);
    // Clearance above the faces' sag between pushed vertices (edges ~0.26 on r = 0.5).
    let r = call(&s, "push_out", json!({"object": "Hair", "target": "Head", "clearance": 0.04}));
    assert!(r["pushed"].as_u64().unwrap() >= 4, "{r}");
    let r = call(&s, "intersections", json!({"object": "Hair", "with": ["Head"]}));
    assert_eq!(r["result"], "no crossings", "{r}");
    // The torso and the hair are far apart.
    assert_eq!(call(&s, "intersections", json!({"object": "Figure"}))["result"], "no crossings");
}

/// Follow-ups from the agent run: push_out hint, select without undo,
/// mirror tolerance, compare picture size.
#[test]
fn agent_followups() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "uv_sphere", "radius": 0.5, "segments": 12, "rings": 8}, "name": "Head"}));
    // A coarse slab through the head: every vertex outside, faces crossing.
    call(&s, "add_primitive", json!({"shape": {"kind": "cube", "size": [0.9, 0.3, 0.9]}, "name": "Slab", "position": [0, 0.4, 0]}));
    let r = call(&s, "push_out", json!({"object": "Slab", "target": "Head", "clearance": 0.01}));
    assert_eq!(r["pushed"], 0, "{r}");
    assert!(r["still_crossing"]["crossings"].as_u64().unwrap() > 0, "{r}");

    // intersections select: like select, not an undo step.
    let undo_steps = |s: &polly_mcp::PollyServer| call(s, "history", json!({})).to_string();
    let before = undo_steps(&s);
    call(&s, "intersections", json!({"object": "Slab", "select": true}));
    assert_eq!(undo_steps(&s), before);

    // Mirror tolerance: a vertex 1 cm off its mirror position pairs only with a larger tolerance.
    call(&s, "add_mesh", json!({"name": "Strip", "vertices": [[0.5, 0, 0], [-0.51, 0, 0], [0.5, 1, 0], [-0.5, 1, 0]], "faces": [[0, 2, 3, 1]]}));
    let ids: Vec<String> = {
        let ws = s.workspace.get("main").unwrap();
        let g = ws.lock().unwrap();
        g.doc.scene.objects.values().find(|o| o.name == "Strip").unwrap().mesh.verts.keys().map(|v| format!("v{v}")).collect()
    };
    let r = call(&s, "move", json!({"object": "Strip", "elements": [ids[0].clone()], "offset": [0.1, 0, 0], "mirror": true}));
    assert!(r["mirror"]["unpaired"].is_string(), "{r}");
    let r = call(&s, "move", json!({"object": "Strip", "elements": [ids[0].clone()], "offset": [0.1, 0, 0], "mirror": true, "mirror_tolerance": 0.2}));
    assert_eq!(r["mirror"]["partners_moved"], 1, "{r}");
}

fn glb_json(path: &std::path::Path) -> Value {
    let glb = std::fs::read(path).unwrap();
    let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&glb[20..20 + len]).unwrap()
}

#[test]
fn pbr_materials() {
    let s = server();
    call(&s, "add_primitive", json!({"shape": {"kind": "cube"}, "name": "Box"}));
    call(&s, "auto_seams", json!({}));
    call(&s, "unwrap", json!({"texture_size": [32, 32]}));
    let t = call(&s, "create_texture", json!({"name": "N", "width": 32, "height": 32, "kind": "normal"}));
    assert_eq!(t["linear"], true);
    let tex = call(&s, "get_texture", json!({"texture": "N"}));
    assert!(tex.to_string().contains("80"), "flat normal fill: {tex}");
    call_err(&s, "create_texture", json!({"name": "Big", "width": 8192, "height": 8}));
    call(&s, "create_texture", json!({"name": "Big", "width": 4096, "height": 8}));
    call_err(&s, "create_texture", json!({"name": "X", "kind": "normal", "material": "M", "slot": "base"}));
    let m = call(
        &s,
        "create_material",
        json!({
            "name": "Paint", "color": "#3366cc", "roughness": 0.3, "metallic": 0.8,
            "normal_texture": "N", "emissive": "#ff6600", "emissive_strength": 2.0,
            "clearcoat": {"factor": 1.0, "roughness": 0.05}
        }),
    );
    assert_eq!(m["index"], 1);
    call_err(&s, "update_material", json!({"material": "Paint", "blend": true, "alpha_cutoff": 0.5}));
    call_err(&s, "update_material", json!({"material": "Paint", "roughness": 2.0}));
    call(&s, "assign_material", json!({"elements": ["all"], "material": "Paint"}));
    let d = call(&s, "describe_scene", json!({}));
    let pm = &d["materials"][1];
    assert_eq!(pm["normal_texture"], "N");
    assert_eq!(pm["clearcoat"]["factor"], 1.0);
    assert!(pm.get("occlusion_texture").is_none() && d["materials"][0].get("roughness").is_none(), "{d}");

    let dir = std::env::temp_dir().join(format!("polly-pbr-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let glb = dir.join("pbr.glb");
    call(&s, "export_glb", json!({"path": glb.to_str().unwrap()}));
    let j = glb_json(&glb);
    let mat = j["materials"].as_array().unwrap().iter().find(|m| m["name"] == "Paint").unwrap().clone();
    assert!((mat["pbrMetallicRoughness"]["roughnessFactor"].as_f64().unwrap() - 0.3).abs() < 1e-6);
    assert!((mat["pbrMetallicRoughness"]["metallicFactor"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    assert!(mat["normalTexture"].is_object());
    assert_eq!(mat["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"], 2.0);
    assert!(mat["extensions"]["KHR_materials_clearcoat"].is_object());
    assert!(j["meshes"][0]["primitives"][0]["attributes"]["TANGENT"].is_number());

    // The render shows the emissive glow (orange added to blue) and says PBR channels are not shown yet.
    let r = s.call("render", json!({"view": "front", "overlays": {"wireframe": false, "grid": false}, "legend": false}));
    if r.is_error != Some(true) {
        assert!(text(&r).contains("PBR channels not shown"), "{}", text(&r));
        std::fs::write(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("session_pbr.png"), &images(&r)[0]).unwrap();
        let img = image::load_from_memory(&images(&r)[0]).unwrap().to_rgba8();
        let p = img.get_pixel(256, 256);
        assert!(p[0] > 200 && p[1] > 100, "emissive orange should show at the centre, got {p:?}");
    } else {
        eprintln!("skipping render: {}", text(&r));
    }

    // Clearing the normal map removes tangents; the texture stays linear.
    call(&s, "update_material", json!({"material": "Paint", "normal_texture": ""}));
    call(&s, "update_material", json!({"material": "Paint", "clearcoat": null, "emissive": ""}));
    call(&s, "export_glb", json!({"path": glb.to_str().unwrap()}));
    let j = glb_json(&glb);
    assert!(j["meshes"][0]["primitives"][0]["attributes"].get("TANGENT").is_none());
    let mat = &j["materials"].as_array().unwrap().iter().find(|m| m["name"] == "Paint").unwrap().clone();
    assert!(mat.get("extensions").is_none() && mat.get("emissiveFactor").is_none(), "{mat}");

    // A colour texture moved into a data slot becomes linear, with a note.
    call(&s, "create_texture", json!({"name": "C", "width": 8, "height": 8}));
    let u = call(&s, "update_material", json!({"material": "Paint", "occlusion_texture": "C"}));
    assert!(u["notes"][0].as_str().unwrap().contains("linear"), "{u}");
    let uv = s.call("render_uv_layout", json!({"slot": "occlusion"}));
    assert!(text(&uv).contains("\"C\""), "{}", text(&uv));
    std::fs::remove_dir_all(dir).ok();
}
