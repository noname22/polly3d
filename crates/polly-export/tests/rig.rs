//! Rigging / morph / animation export tests.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use gltf::animation::Property;
use polly_core::anim::{AutoWeights, Bone, Clip, Interpolation, Morph, Skeleton, Target, auto_weights};
use polly_core::glam::{Mat4, Quat, Vec3, Vec4};
use polly_core::mesh::VertId;
use polly_core::primitives::Primitive;
use polly_core::scene::{ObjectId, Scene, Transform};
use polly_export::{ExportOptions, export_glb};

fn rigged() -> (Scene, ObjectId) {
    let mut s = Scene::default();
    let mesh = Primitive::Cylinder { radius_bottom: 0.2, radius_top: 0.2, height: 2.0, segments: 6, height_segments: 4, caps: true }
        .build()
        .unwrap();
    let id = s.add_object("Arm", mesh);
    let o = s.objects.get_mut(&id).unwrap();
    o.transform = Transform { translation: Vec3::new(1.0, 2.0, 0.0), ..Default::default() };
    let mut sk = Skeleton::default();
    let upper = sk.add(Bone { name: "upper".into(), parent: None, head: Vec3::new(0.0, -1.0, 0.0), tail: Vec3::ZERO, roll: 0.0 }).unwrap();
    let lower = sk.add(Bone { name: "lower".into(), parent: Some(upper), head: Vec3::ZERO, tail: Vec3::new(0.3, 1.0, 0.0), roll: 20.0 }).unwrap();
    let mesh = Arc::make_mut(&mut o.mesh);
    let all: BTreeSet<VertId> = mesh.verts.keys().copied().collect();
    auto_weights(mesh, &sk, &all, &BTreeSet::from([upper, lower]), AutoWeights::Smooth { falloff: 1.0 }).unwrap();
    let offsets = mesh.verts.iter().filter(|(_, v)| v.pos.y > 0.99).map(|(i, _)| (*i, Vec3::new(0.0, 0.5, 0.0))).collect();
    mesh.morphs.push(Morph { name: "stretch".into(), offsets });
    o.skeleton = Some(sk);
    o.morph_weights.insert("stretch".into(), 0.25);
    let mut c = Clip::new("wave", 2.0, 24.0);
    let ch = c.channel_mut(Target::BoneRotation { object: id, bone: lower });
    ch.set_key(0.0, Vec4::from(Quat::IDENTITY));
    ch.set_key(1.0, Vec4::from(Quat::from_rotation_z(90f32.to_radians())));
    let ch = c.channel_mut(Target::MorphWeight { object: id, morph: "stretch".into() });
    ch.interpolation = Interpolation::Step;
    ch.set_key(0.5, Vec4::new(1.0, 0.0, 0.0, 0.0));
    ch.set_key(1.5, Vec4::ZERO);
    let ch = c.channel_mut(Target::ObjectTranslation { object: id });
    ch.set_key(0.0, Vec4::new(1.0, 2.0, 0.0, 0.0));
    ch.set_key(2.0, Vec4::new(3.0, 2.0, 0.0, 0.0));
    c.loop_start = 0.5;
    c.loop_end = Some(1.5);
    s.add_clip(c);
    (s, id)
}

fn json_chunk(glb: &[u8]) -> serde_json::Value {
    let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&glb[20..20 + len]).unwrap()
}

fn mat(m: [[f32; 4]; 4]) -> Mat4 {
    Mat4::from_cols_array_2d(&m)
}

#[test]
fn exports_skin_morphs_and_animation() {
    let (s, id) = rigged();
    let bytes = export_glb(&s, &ExportOptions { objects: None }).unwrap();
    std::fs::write(concat!(env!("CARGO_TARGET_TMPDIR"), "/rigged.glb"), &bytes).unwrap();
    let (doc, buffers, _) = gltf::import_slice(&bytes).unwrap();
    let get = |b: gltf::Buffer| Some(&buffers[b.index()][..]);
    let sk = s.objects[&id].skeleton.as_ref().unwrap();

    // Skin.
    let skin = doc.skins().next().unwrap();
    let joints: Vec<gltf::Node> = skin.joints().collect();
    assert_eq!(joints.len(), 2);
    let names: Vec<&str> = joints.iter().map(|j| j.name().unwrap()).collect();
    assert_eq!(names, ["upper", "lower"]);
    let ibm: Vec<[[f32; 4]; 4]> = skin.reader(get).read_inverse_bind_matrices().unwrap().collect();
    assert_eq!(ibm.len(), 2);
    let obj_node = doc.nodes().find(|n| n.mesh().is_some()).unwrap();
    assert_eq!(obj_node.skin().unwrap().index(), skin.index());

    // Rest skeleton: composing node locals below the object node reproduces rest_matrix().
    let mut parent: HashMap<usize, usize> = HashMap::new();
    for n in doc.nodes() {
        for c in n.children() {
            parent.insert(c.index(), n.index());
        }
    }
    let nodes: Vec<gltf::Node> = doc.nodes().collect();
    for (j, bid) in joints.iter().zip(sk.ordered()) {
        let mut m = Mat4::IDENTITY;
        let mut cur = j.index();
        while cur != obj_node.index() {
            m = mat(nodes[cur].transform().matrix()) * m;
            cur = parent[&cur];
        }
        let rest = sk.bones[&bid].rest_matrix();
        assert!(m.abs_diff_eq(rest, 1e-5), "{m:?} vs {rest:?}");
        let ibm_m = mat(ibm[joints.iter().position(|x| x.index() == j.index()).unwrap()]);
        assert!((ibm_m * m).abs_diff_eq(Mat4::IDENTITY, 1e-5));
    }

    // Vertex skin data and morph targets.
    let mesh = obj_node.mesh().unwrap();
    for p in mesh.primitives() {
        let r = p.reader(get);
        let n = r.read_positions().unwrap().count();
        let js: Vec<[u16; 4]> = r.read_joints(0).unwrap().into_u16().collect();
        let ws: Vec<[f32; 4]> = r.read_weights(0).unwrap().into_f32().collect();
        assert_eq!((js.len(), ws.len()), (n, n));
        for (j, w) in js.iter().zip(&ws) {
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(j.iter().all(|i| *i < 2));
        }
        let targets: Vec<_> = r.read_morph_targets().collect();
        assert_eq!(targets.len(), 1);
        let deltas: Vec<[f32; 3]> = targets[0].0.clone().unwrap().collect();
        assert_eq!(deltas.len(), n);
        assert!(deltas.iter().any(|d| d[1] > 0.49));
    }
    assert_eq!(mesh.weights(), Some(&[0.25][..]));
    let js = json_chunk(&bytes);
    assert_eq!(js["meshes"][0]["extras"]["targetNames"], serde_json::json!(["stretch"]));

    // Animation.
    assert_eq!(doc.animations().count(), 1);
    let anim = doc.animations().next().unwrap();
    assert_eq!(anim.name(), Some("wave"));
    let mut paths: Vec<(usize, String)> =
        anim.channels().map(|c| (c.target().node().index(), format!("{:?}", c.target().property()))).collect();
    paths.sort();
    let lower_node = joints[1].index();
    let mut expected = vec![
        (obj_node.index(), "Translation".to_string()),
        (obj_node.index(), "MorphTargetWeights".to_string()),
        (lower_node, "Rotation".to_string()),
    ];
    expected.sort();
    assert_eq!(paths, expected);
    for c in anim.channels() {
        let r = c.reader(get);
        let times: Vec<f32> = r.read_inputs().unwrap().collect();
        match c.target().property() {
            Property::Rotation => {
                let gltf::animation::util::ReadOutputs::Rotations(q) = r.read_outputs().unwrap() else { panic!() };
                let q: Vec<[f32; 4]> = q.into_f32().collect();
                // Key 0 = identity pose = rest local rotation.
                let rest_local = Quat::from_array(nodes[lower_node].transform().decomposed().1);
                assert!(Quat::from_array(q[0]).abs_diff_eq(rest_local, 1e-5));
                let want = rest_local * Quat::from_rotation_z(90f32.to_radians());
                assert!(Quat::from_array(q[1]).abs_diff_eq(want, 1e-5));
            }
            Property::MorphTargetWeights => {
                assert_eq!(c.sampler().interpolation(), gltf::animation::Interpolation::Step);
                assert_eq!(times, [0.5, 1.5]);
                let gltf::animation::util::ReadOutputs::MorphTargetWeights(w) = r.read_outputs().unwrap() else { panic!() };
                assert_eq!(w.into_f32().collect::<Vec<_>>(), [1.0, 0.0]);
            }
            Property::Translation => assert_eq!(times, [0.0, 2.0]),
            p => panic!("unexpected {p:?}"),
        }
    }
    let extras = &js["animations"][0]["extras"]["polly"];
    assert_eq!(extras["looping"], serde_json::json!(true));
    assert_eq!(extras["loop_end"], serde_json::json!(1.5));
    assert_eq!(extras["fps"], serde_json::json!(24.0));
}

#[test]
fn unweighted_vertices_bind_to_root_joint() {
    let (mut s, id) = rigged();
    let o = s.objects.get_mut(&id).unwrap();
    let mesh = Arc::make_mut(&mut o.mesh);
    let v = *mesh.weights.keys().next().unwrap();
    mesh.weights.remove(&v);
    let bytes = export_glb(&s, &ExportOptions { objects: None }).unwrap();
    let (doc, buffers, _) = gltf::import_slice(&bytes).unwrap();
    let skin = doc.skins().next().unwrap();
    let joints: Vec<gltf::Node> = skin.joints().collect();
    assert_eq!(joints.len(), 3);
    assert_eq!(joints[2].name(), Some("Arm_root"));
    let get = |b: gltf::Buffer| Some(&buffers[b.index()][..]);
    let uses_root = doc.meshes().next().unwrap().primitives().any(|p| {
        let js: Vec<[u16; 4]> = p.reader(get).read_joints(0).unwrap().into_u16().collect();
        js.iter().any(|j| j[0] == 2)
    });
    assert!(uses_root);
}

#[test]
fn dropped_bone_channels_are_skipped() {
    let (mut s, id) = rigged();
    let sk = s.objects.get_mut(&id).unwrap().skeleton.as_mut().unwrap();
    let lower = sk.find("lower").unwrap();
    sk.remove(lower).unwrap();
    let bytes = export_glb(&s, &ExportOptions { objects: None }).unwrap();
    let (doc, ..) = gltf::import_slice(&bytes).unwrap();
    assert_eq!(doc.animations().next().unwrap().channels().count(), 2);
}
