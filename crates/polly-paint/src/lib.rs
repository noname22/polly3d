//! Texture painting for Polly3D: palettes, crisp pixel primitives, ordered
//! dithering, procedural retro patterns, surface-space (UV) face fills and
//! brush strokes, quantization and PNG I/O.
//!
//! Pixel coordinates are integers with (0,0) at the top-left. UVs follow glTF:
//! (0,0) is the top-left of the image, v points down, and texel (x, y) has its
//! centre at ((x+0.5)/w, (y+0.5)/h). All drawing clips silently to the texture.

pub mod canvas;
pub mod color;
pub mod faces;
pub mod io;
pub mod mask;
pub mod palette;
pub mod quantize;

pub use canvas::{
    Blend, Dither, GradientDir, Paint, adjust, border, dither_fill, ellipse, flood_fill, gradient, line, noise, outline,
    polygon, polyline, rect, replace_colors, set_pixels,
};
pub use color::{Rgba, parse_color, to_hex};
pub use faces::{Fill, bake_vertex_colors, face_mask, face_uv_polys, fill_faces, paint_at_uv, stroke_uv};
pub use io::{TextureStats, from_png, stats, to_png};
pub use mask::{FaceUv, Mask, Region};
pub use quantize::{quantize, resize_nearest};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_are_mcp_friendly() {
        let fill: Fill = serde_json::from_str(r#"{"type":"bricks","mortar":[0,0,0,255],"brick":[[200,80,60,255]],"size_px":[8,4]}"#).unwrap();
        assert!(matches!(fill, Fill::Bricks { mortar_px: 1, .. }));
        let r: Region = serde_json::from_str(r#"{"type":"faces","polys":[[[0,0],[1,0],[0,1]]]}"#).unwrap();
        assert!(matches!(r, Region::Faces { grow: 0, .. }));
        let p: Paint = serde_json::from_str(r#"{"color":[1,2,3,255],"blend":"alpha"}"#).unwrap();
        assert_eq!(p.blend, Blend::Alpha);
        for s in [
            schemars::schema_for!(Fill),
            schemars::schema_for!(Region),
            schemars::schema_for!(Paint),
            schemars::schema_for!(GradientDir),
            schemars::schema_for!(TextureStats),
        ] {
            assert!(serde_json::to_string(&s).unwrap().len() > 10);
        }
    }
}
