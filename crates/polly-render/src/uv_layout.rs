//! CPU UV-layout view: texture (or checker) with the object's UV wireframe.

use image::RgbaImage;
use polly_core::ids;
use polly_core::mesh::Edge;
use polly_core::scene::{ObjectId, Scene, TextureId};

use crate::canvas::{self, Rect, WHITE};

/// Draw the UV layout of `object` over `texture` (or a checker when `None` /
/// missing). Only faces whose material uses that texture are drawn, unless no
/// face does, in which case all faces are drawn. The longer image side is
/// `size` pixels (64..2048). UV (0,0) is the top-left of the image.
pub fn render_uv_layout(scene: &Scene, object: ObjectId, texture: Option<TextureId>, size: u32, show_ids: bool) -> RgbaImage {
    let size = size.clamp(64, 2048);
    let tex = texture.and_then(|t| scene.textures.get(&t)).filter(|t| t.width > 0 && t.height > 0 && t.pixels.len() == (t.width * t.height * 4) as usize);
    let (tw, th) = tex.map_or((8, 8), |t| (t.width, t.height));
    let (ow, oh) = if tw >= th {
        (size, ((size as f32 * th as f32 / tw as f32).round() as u32).max(1))
    } else {
        (((size as f32 * tw as f32 / th as f32).round() as u32).max(1), size)
    };
    let mut img = RgbaImage::new(ow, oh);
    for y in 0..oh {
        let sy = (y as u64 * th as u64 / oh as u64) as u32;
        for x in 0..ow {
            let sx = (x as u64 * tw as u64 / ow as u64) as u32;
            let px = match tex {
                Some(t) => {
                    let p = t.get(sx, sy);
                    // Composite transparency over a neutral grey checker.
                    let bg = if ((x / 8) + (y / 8)) % 2 == 0 { 70.0 } else { 95.0 };
                    let a = p[3] as f32 / 255.0;
                    let mix = |c: u8| (c as f32 * a + bg * (1.0 - a)).round() as u8;
                    [mix(p[0]), mix(p[1]), mix(p[2]), 255]
                }
                None => {
                    if (sx + sy) % 2 == 0 { [92, 94, 104, 255] } else { [62, 64, 72, 255] }
                }
            };
            img.put_pixel(x, y, image::Rgba(px));
        }
    }
    let factor = ow as f32 / tw as f32;
    if tex.is_some() && factor >= 8.0 {
        for i in 1..tw {
            let x = (i as f32 * factor).round() as i32;
            for y in 0..oh as i32 {
                canvas::blend(&mut img, x, y, [0, 0, 0, 255], 0.18);
            }
        }
        for j in 1..th {
            let y = (j as f32 * oh as f32 / th as f32).round() as i32;
            for x in 0..ow as i32 {
                canvas::blend(&mut img, x, y, [0, 0, 0, 255], 0.18);
            }
        }
    }

    let Some(obj) = scene.objects.get(&object) else { return img };
    let mesh = &obj.mesh;
    let uses_tex = |m: u32| texture.is_some() && scene.materials.get(&m).and_then(|m| m.texture) == texture;
    let any = mesh.faces.values().any(|f| uses_tex(f.material));
    let faces: Vec<_> = mesh.faces.iter().filter(|(_, f)| !any || uses_tex(f.material)).collect();
    let to_px = |uv: glam::Vec2| [uv.x * ow as f32, uv.y * oh as f32];
    let lw = (size as f32 / 512.0).clamp(1.0, 2.5);

    // Dark halo first, then bright lines, then seams on top.
    for pass in 0..3 {
        for (_, f) in &faces {
            let n = f.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let a = to_px(f.corners[i].uv);
                let b = to_px(f.corners[j].uv);
                let seam = mesh.seams.contains(&Edge::new(f.verts[i], f.verts[j]));
                match pass {
                    0 => canvas::line(&mut img, a, b, [0, 0, 0, 200], lw + 2.0),
                    1 if !seam => canvas::line(&mut img, a, b, [240, 240, 245, 255], lw),
                    2 if seam => canvas::line(&mut img, a, b, [255, 60, 60, 255], lw + 0.8),
                    _ => {}
                }
            }
        }
    }
    let uv_area: f32 = faces
        .iter()
        .map(|(_, f)| {
            let n = f.len();
            (0..n).map(|i| f.corners[i].uv.perp_dot(f.corners[(i + 1) % n].uv)).sum::<f32>().abs() * 0.5
        })
        .sum();
    if !faces.is_empty() && uv_area < 1e-9 {
        let msg = "no UVs: zero UV area";
        let scale = if size >= 480 { 2 } else { 1 };
        let r = canvas::label_rect(msg, scale, ow as f32 * 0.5, oh as f32 * 0.5);
        canvas::boxed_label(&mut img, r, msg, scale, [150, 30, 30, 230], WHITE);
    }
    if show_ids {
        let scale = if size >= 480 { 2 } else { 1 };
        // Faces whose UV centroids coincide (stacked / overlapping islands) share one label.
        let mut groups: Vec<([f32; 2], Vec<u32>)> = Vec::new();
        for (fid, f) in &faces {
            let c = f.corners.iter().map(|c| c.uv).sum::<glam::Vec2>() / f.len() as f32;
            let p = to_px(c);
            match groups.iter_mut().find(|(q, _)| (q[0] - p[0]).abs() < 2.0 && (q[1] - p[1]).abs() < 2.0) {
                Some((_, ids)) => ids.push(**fid),
                None => groups.push((p, vec![**fid])),
            }
        }
        let mut placed: Vec<Rect> = Vec::new();
        for (p, fids) in groups {
            let mut text = fids.iter().take(6).map(|f| ids::face(*f)).collect::<Vec<_>>().join(" ");
            if fids.len() > 6 {
                text.push_str(&format!(" +{}", fids.len() - 6));
            }
            for d in [[0.0, 0.0], [0.0, -10.0 * scale as f32], [0.0, 10.0 * scale as f32]] {
                let mut r = canvas::label_rect(&text, scale, p[0] + d[0], p[1] + d[1]);
                // Keep labels of faces at the border (or with degenerate UVs) inside the image.
                r.x = r.x.clamp(0, (ow as i32 - r.w).max(0));
                r.y = r.y.clamp(0, (oh as i32 - r.h).max(0));
                if r.inside(ow, oh) && !placed.iter().any(|q| q.overlaps(&r)) {
                    canvas::boxed_label(&mut img, r, &text, scale, [28, 52, 110, 215], WHITE);
                    placed.push(r);
                    break;
                }
            }
        }
    }
    img
}
