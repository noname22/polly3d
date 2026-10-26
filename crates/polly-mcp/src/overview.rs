//! Overview of all scenes as a captioned grid, shared by the
//! `render_overview` tool and the live viewer.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;
use image::RgbaImage;
use polly_render::canvas::{self, Rect};
use polly_render::{Overlays, RenderOptions, Renderer, View};
use serde::Serialize;

use crate::workspace::Snapshot;

#[derive(Debug, Clone, Serialize)]
pub struct TileRect {
    pub scene: String,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl TileRect {
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x as f64 && y >= self.y as f64 && x < (self.x + self.w) as f64 && y < (self.y + self.h) as f64
    }
}

/// Per-scene tile renders, reused while a scene's version and tile size are unchanged.
#[derive(Default)]
pub struct TileCache(Mutex<HashMap<String, (u64, u32, RgbaImage)>>);

const GAP: u32 = 4;
const BG: [u8; 4] = [18, 19, 23, 255];

/// How a scene thumbnail is rendered (shared with the viewer's overview).
pub fn tile_options(size: u32) -> RenderOptions {
    RenderOptions {
        view: View::Persp,
        width: size,
        height: size,
        overlays: Overlays { selection: false, ..Overlays::default() },
        legend: false,
        ..RenderOptions::default()
    }
}

/// Grid geometry for `n` tiles that best fills `width` x `height`.
/// Returns (columns, tile size).
pub fn layout(n: usize, width: u32, height: u32) -> (u32, u32) {
    let n = n.max(1) as u32;
    (1..=n)
        .map(|cols| {
            let rows = n.div_ceil(cols);
            let size = ((width.saturating_sub(GAP * (cols + 1))) / cols).min(height.saturating_sub(GAP * (rows + 1)) / rows);
            (cols, size)
        })
        .max_by_key(|&(_, size)| size)
        .unwrap()
}

/// Render `scenes` as a grid of `tile`-sized squares with `cols` columns.
pub fn render_grid(
    renderer: &mut Renderer,
    cache: &TileCache,
    scenes: &[Snapshot],
    cols: u32,
    tile: u32,
    canvas_size: Option<[u32; 2]>,
) -> Result<(RgbaImage, Vec<TileRect>)> {
    let tile = tile.clamp(64, 1024);
    let cols = cols.max(1);
    let rows = (scenes.len() as u32).div_ceil(cols).max(1);
    let [w, h] = canvas_size.unwrap_or([cols * tile + GAP * (cols + 1), rows * tile + GAP * (rows + 1)]);
    let mut img = RgbaImage::from_pixel(w, h, image::Rgba(BG));
    // Center the grid in the canvas.
    let (gw, gh) = (cols * tile + GAP * (cols + 1), rows * tile + GAP * (rows + 1));
    let (ox, oy) = (w.saturating_sub(gw) / 2, h.saturating_sub(gh) / 2);
    let mut rects = Vec::new();
    let mut cache = cache.0.lock().unwrap();
    cache.retain(|name, _| scenes.iter().any(|s| &s.name == name));
    for (i, s) in scenes.iter().enumerate() {
        let (c, r) = (i as u32 % cols, i as u32 / cols);
        let (x, y) = (ox + GAP + c * (tile + GAP), oy + GAP + r * (tile + GAP));
        let fresh = matches!(cache.get(&s.name), Some((v, t, _)) if *v == s.version && *t == tile);
        if !fresh {
            let shown = polly_core::variants::display(&s.scene, &[], &Default::default());
            let rendered = renderer.render(&shown, &tile_options(tile))?.image;
            cache.insert(s.name.clone(), (s.version, tile, rendered));
        }
        let tile_img = &cache[&s.name].2;
        image::imageops::overlay(&mut img, tile_img, x as i64, y as i64);
        caption(&mut img, s, x as i32, y as i32, tile as i32);
        rects.push(TileRect { scene: s.name.clone(), x, y, w: tile, h: tile });
    }
    Ok((img, rects))
}

fn caption(img: &mut RgbaImage, s: &Snapshot, x: i32, y: i32, tile: i32) {
    let scale = if tile >= 320 { 2 } else { 1 };
    let objects = s.scene.objects.len();
    let tris: usize = s.scene.objects.values().map(|o| o.mesh.triangle_count()).sum();
    let detail = if objects == 0 { "(empty)".to_string() } else { format!("{objects} obj, {tris} tris") };
    let (_, th) = canvas::text_size("X", scale);
    let bar = th * 2 + 6 * scale as i32;
    canvas::fill_rect(img, Rect { x, y: y + tile - bar, w: tile, h: bar }, [0, 0, 0, 170]);
    canvas::text(img, x + 4, y + tile - bar + 2 * scale as i32, &s.name, scale, canvas::WHITE, None);
    canvas::text(img, x + 4, y + tile - th - 2 * scale as i32, &detail, scale, [180, 185, 195, 255], None);
}
