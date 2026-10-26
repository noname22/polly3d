//! CPU annotation after readback: element-id labels, object names, legend and
//! axis gizmo.

use glam::Vec3;
use image::RgbaImage;
use polly_core::ids;
use polly_core::scene::Scene;

use crate::camera::CameraInfo;
use crate::canvas::{self, Rect, Rgba, WHITE};
use crate::gpu::Frame;
use crate::options::RenderOptions;

pub const MAX_LABELS: usize = 300;

const FACE_BG: Rgba = [28, 52, 110, 215];
const VERT_BG: Rgba = [22, 90, 40, 215];
const EDGE_BG: Rgba = [95, 35, 100, 215];
const SEL_BG: Rgba = [190, 85, 10, 235];
const NAME_BG: Rgba = [120, 95, 10, 225];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Face,
    Edge,
    Vert,
    Name,
}

struct Cand {
    text: String,
    at: [f32; 2],
    depth: f32,
    kind: Kind,
    selected: bool,
}

pub(crate) fn label_scale(o: &RenderOptions, size: [u32; 2]) -> u32 {
    o.label_scale.unwrap_or(if size[0].min(size[1]) >= 480 { 2 } else { 1 }).clamp(1, 4)
}

/// Draw labels + legend. Returns the number of labels drawn.
pub(crate) fn annotate(img: &mut RgbaImage, scene: &Scene, o: &RenderOptions, frame: &Frame, notes: &mut Vec<String>) -> usize {
    let size = [img.width(), img.height()];
    let cam = &frame.cam;
    let prep = &frame.prep;
    let ov = &o.overlays;
    let scale = label_scale(o, size);
    let eps = frame.frame_radius * 2e-3 + 1e-5;
    let mut cands: Vec<Cand> = Vec::new();

    // Which objects get element labels: focus objects, else the only/active object.
    let label_objs: Vec<usize> = {
        let all: Vec<usize> = (0..prep.objects.len()).collect();
        match o.focus.as_ref().filter(|f| !f.is_empty()) {
            Some(f) => all.into_iter().filter(|&i| f.contains(&prep.objects[i].id)).collect(),
            None if prep.objects.len() <= 1 => all,
            None => match scene.active.and_then(|a| all.iter().copied().find(|&i| prep.objects[i].id == a)) {
                Some(i) => vec![i],
                None => all,
            },
        }
    };
    let element_labels = ov.face_ids || ov.vert_ids || ov.edge_ids;
    if element_labels && !label_objs.is_empty() && prep.objects.len() > 1 {
        let names: Vec<String> = label_objs.iter().map(|&i| format!("'{}'", prep.objects[i].name)).collect();
        let hint = if o.focus.is_none() { " (pass focus to choose)" } else { "" };
        notes.push(format!("element ids are labelled for {}{hint}", names.join(", ")));
    }

    let on_screen = |p: [f32; 2]| p[0] >= 0.0 && p[1] >= 0.0 && p[0] < size[0] as f32 && p[1] < size[1] as f32;
    let facing = |p: Vec3, n: Vec3| n.dot(cam.to_eye(p)) > 1e-3;

    for &oi in &label_objs {
        let po = &prep.objects[oi];
        let mesh = &po.mesh;
        let sel = &scene.objects[&po.id].selection;
        if ov.face_ids {
            for (&fid, f) in &mesh.faces {
                let Some(&n) = po.face_normal.get(&fid) else { continue };
                let c = po.face_center(f);
                if !ov.backface_labels && (!facing(c, n) || !prep.visible(cam, c, Some((oi, fid)), eps)) {
                    continue;
                }
                let Some((at, depth)) = cam.project_depth(c) else { continue };
                if on_screen(at) {
                    cands.push(Cand { text: ids::face(fid), at, depth, kind: Kind::Face, selected: sel.faces.contains(&fid) });
                }
            }
        }
        if ov.edge_ids {
            let ef = mesh.edge_faces();
            for (e, fs) in &ef {
                let (Some(&a), Some(&b)) = (po.pos.get(&e.0), po.pos.get(&e.1)) else { continue };
                let m = (a + b) * 0.5;
                if !ov.backface_labels {
                    let any_front = fs.iter().any(|f| po.face_normal.get(f).is_some_and(|n| facing(m, *n)));
                    if !any_front || !prep.visible(cam, m, None, eps) {
                        continue;
                    }
                }
                let Some((at, depth)) = cam.project_depth(m) else { continue };
                if on_screen(at) {
                    cands.push(Cand { text: ids::edge(*e), at, depth, kind: Kind::Edge, selected: sel.edges.contains(e) });
                }
            }
        }
        if ov.vert_ids {
            let vf = mesh.vert_faces();
            for (&v, &p) in &po.pos {
                if !ov.backface_labels {
                    let any_front = vf.get(&v).is_none_or(|fs| fs.iter().any(|f| po.face_normal.get(f).is_some_and(|n| facing(p, *n))));
                    if !any_front || !prep.visible(cam, p, None, eps) {
                        continue;
                    }
                }
                let Some((at, depth)) = cam.project_depth(p) else { continue };
                if on_screen(at) {
                    cands.push(Cand { text: ids::vert(v), at, depth, kind: Kind::Vert, selected: sel.verts.contains(&v) });
                }
            }
        }
    }
    if ov.object_names {
        for po in &prep.objects {
            let Some((lo, hi)) = po.bounds() else { continue };
            let top = Vec3::new((lo.x + hi.x) * 0.5, hi.y, (lo.z + hi.z) * 0.5);
            if let Some(at) = cam.project(top) {
                let at = [at[0], at[1] - 10.0 * scale as f32];
                cands.push(Cand { text: format!("{} (o{})", po.name, po.id), at, depth: -1.0, kind: Kind::Name, selected: false });
            }
        }
    }

    // Names first, then selected elements, then nearest first; faces before edges before verts.
    let rank = |k: Kind| match k {
        Kind::Name => 0,
        Kind::Face => 1,
        Kind::Edge => 2,
        Kind::Vert => 3,
    };
    cands.sort_by(|a, b| {
        (a.kind != Kind::Name)
            .cmp(&(b.kind != Kind::Name))
            .then(b.selected.cmp(&a.selected))
            .then(a.depth.total_cmp(&b.depth))
            .then(rank(a.kind).cmp(&rank(b.kind)))
    });

    let mut placed: Vec<Rect> = Vec::new();
    // Reserve the legend area.
    let legend_rect = if o.legend { draw_legend(img, cam, scale, true) } else { None };
    if let Some(r) = legend_rect {
        placed.push(r);
    }
    // The gizmo is drawn underneath labels, so it does not block label space.
    if o.legend {
        draw_gizmo(img, cam, scale);
    }
    let mut skipped_overlap = 0;
    let mut capped = 0;
    let mut drawn: Vec<(Rect, &Cand)> = Vec::new();
    for c in &cands {
        if drawn.len() >= MAX_LABELS {
            capped += 1;
            continue;
        }
        let (tw, th) = canvas::text_size(&c.text, scale);
        let (bw, bh) = ((tw + 2 * scale as i32 + 2) as f32, (th + 2 * scale as i32 + 2) as f32);
        let (x, y) = (c.at[0], c.at[1]);
        let offs: Vec<[f32; 2]> = match c.kind {
            Kind::Vert => {
                let (dx, dy) = (bw * 0.5 + 3.0, bh * 0.5 + 1.0);
                vec![[dx, -dy], [-dx, -dy], [dx, dy], [-dx, dy], [0.0, -dy - 3.0], [0.0, dy + 3.0]]
            }
            _ => vec![[0.0, 0.0], [0.0, -bh * 0.6], [0.0, bh * 0.6], [bw * 0.6, 0.0], [-bw * 0.6, 0.0]],
        };
        let mut ok = None;
        for d in offs {
            let r = canvas::label_rect(&c.text, scale, x + d[0], y + d[1]);
            if r.inside(size[0], size[1]) && !placed.iter().any(|p| p.overlaps(&r)) {
                ok = Some(r);
                break;
            }
        }
        match ok {
            Some(r) => {
                placed.push(r);
                drawn.push((r, c));
            }
            None => skipped_overlap += 1,
        }
    }
    // Vertex labels get a tick mark towards their vertex.
    for (r, c) in &drawn {
        if c.kind == Kind::Vert {
            let cx = (r.x + r.w / 2) as f32;
            let cy = (r.y + r.h / 2) as f32;
            let ex = cx.clamp(r.x as f32, (r.x + r.w) as f32);
            let ey = if c.at[1] < cy { r.y as f32 } else { (r.y + r.h) as f32 };
            canvas::line(img, [ex, ey], c.at, [10, 10, 14, 200], 1.0);
        }
    }
    for (r, c) in &drawn {
        let bg = if c.selected {
            SEL_BG
        } else {
            match c.kind {
                Kind::Face => FACE_BG,
                Kind::Edge => EDGE_BG,
                Kind::Vert => VERT_BG,
                Kind::Name => NAME_BG,
            }
        };
        canvas::boxed_label(img, *r, &c.text, scale, bg, WHITE);
    }
    if capped > 0 {
        notes.push(format!(
            "label limit {MAX_LABELS} reached; {capped} more labels not drawn (zoom in, use focus, or fewer id overlays)"
        ));
    }
    if skipped_overlap > 0 {
        notes.push(format!("{skipped_overlap} labels skipped to avoid overlap (zoom in or render larger to see them)"));
    }
    if o.legend {
        draw_legend(img, cam, scale, false);
    }
    drawn.len()
}

fn legend_lines(cam: &CameraInfo) -> Vec<String> {
    let mut l = vec![if cam.view == "persp" {
        "PERSP".to_string()
    } else {
        format!("{} ({})", cam.view.to_uppercase(), if cam.ortho { "ortho" } else { "persp" })
    }];
    if let Some(h) = cam.axis_hint() {
        l.push(h);
    }
    l
}

/// Legend box in the top-left corner. With `measure` only the rect is returned.
fn draw_legend(img: &mut RgbaImage, cam: &CameraInfo, scale: u32, measure: bool) -> Option<Rect> {
    let lines = legend_lines(cam);
    let ls = scale.max(1);
    let pad = 3 * ls as i32;
    let lh = 10 * ls as i32;
    let w = lines.iter().map(|s| canvas::text_size(s, ls).0).max().unwrap_or(0) + pad * 2;
    let h = lh * lines.len() as i32 + pad * 2 - 2 * ls as i32;
    let r = Rect { x: 4, y: 4, w, h };
    if !measure {
        canvas::fill_rect(img, r, [0, 0, 0, 150]);
        for (i, s) in lines.iter().enumerate() {
            let col = if i == 0 { [255, 255, 255, 255] } else { [200, 205, 215, 255] };
            canvas::text(img, r.x + pad, r.y + pad + i as i32 * lh, s, ls, col, None);
        }
    }
    Some(r)
}

fn gizmo_rect(size: [u32; 2], scale: u32) -> Rect {
    let s = 16 * scale as i32 + 8;
    Rect { x: 4, y: size[1] as i32 - s * 2 - 4, w: s * 2, h: s * 2 }
}

/// Small XYZ axis gizmo in the bottom-left corner.
fn draw_gizmo(img: &mut RgbaImage, cam: &CameraInfo, scale: u32) {
    let r = gizmo_rect([img.width(), img.height()], scale);
    let c = [(r.x + r.w / 2) as f32, (r.y + r.h / 2) as f32];
    let len = 13.0 * scale as f32;
    canvas::disc(img, c, len + 7.0 * scale as f32, [0, 0, 0, 90]);
    let (right, up, fwd) = (cam.right(), cam.screen_up(), cam.forward());
    let mut axes = [(Vec3::X, "X", [235, 70, 70, 255]), (Vec3::Y, "Y", [90, 215, 90, 255]), (Vec3::Z, "Z", [90, 140, 255, 255])];
    // Draw the farthest axis first.
    axes.sort_by(|a, b| b.0.dot(fwd).total_cmp(&a.0.dot(fwd)));
    for (a, name, col) in axes {
        let d = [a.dot(right), -a.dot(up)];
        let end = [c[0] + d[0] * len, c[1] + d[1] * len];
        let away = a.dot(fwd) > 0.3;
        let col = if away { [col[0] / 2 + 40, col[1] / 2 + 40, col[2] / 2 + 40, 255] } else { col };
        canvas::line(img, c, end, col, 1.5 * scale as f32);
        let (tw, th) = canvas::text_size(name, scale);
        let lp = [c[0] + d[0] * (len + 5.0 * scale as f32), c[1] + d[1] * (len + 5.0 * scale as f32)];
        canvas::text(img, (lp[0] - tw as f32 / 2.0) as i32, (lp[1] - th as f32 / 2.0) as i32, name, scale, col, Some([0, 0, 0, 255]));
    }
}
