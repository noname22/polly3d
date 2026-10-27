//! Overlay lines: a custom material that expands segments into screen-space
//! quads (pixel widths, depth pull toward the eye, optional "on top").

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError, VertexFormat};
use bevy::shader::ShaderRef;
use polly_render::lines::LineInst;

pub const LINE_SHADER: &str = include_str!("line.wesl");

const ATTRIBUTE_B: MeshVertexAttribute = MeshVertexAttribute::new("PollyLineB", 917_340_001, VertexFormat::Float32x3);
const ATTRIBUTE_PARAMS: MeshVertexAttribute = MeshVertexAttribute::new("PollyLineParams", 917_340_002, VertexFormat::Float32x4);
const ATTRIBUTE_CORNER: MeshVertexAttribute = MeshVertexAttribute::new("PollyLineCorner", 917_340_003, VertexFormat::Float32x2);

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
pub struct LineMaterial {}

/// Shader handle, set once when the service starts.
static SHADER: std::sync::OnceLock<Handle<Shader>> = std::sync::OnceLock::new();

pub fn set_shader(h: Handle<Shader>) {
    let _ = SHADER.set(h);
}

fn shader() -> ShaderRef {
    SHADER.get().cloned().map_or(ShaderRef::Default, ShaderRef::Handle)
}

impl Material for LineMaterial {
    fn vertex_shader() -> ShaderRef {
        shader()
    }
    fn fragment_shader() -> ShaderRef {
        shader()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
    /// Sorts the overlay after other blended geometry (selection tint) so
    /// lines stay crisp on top of it.
    fn depth_bias(&self) -> f32 {
        1.0e6
    }
    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            ATTRIBUTE_B.at_shader_location(1),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(2),
            ATTRIBUTE_PARAMS.at_shader_location(3),
            ATTRIBUTE_CORNER.at_shader_location(4),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// A mesh of quads, four vertices per line or point. `bias` scales each
/// line's depth-pull multiplier (see `Frame::line_bias`).
pub fn mesh(lines: &[LineInst], bias: f32) -> Mesh {
    let n = lines.len() * 4;
    let (mut a, mut b, mut color, mut params, mut corner) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut idx = Vec::with_capacity(lines.len() * 6);
    for (i, l) in lines.iter().enumerate() {
        let p = [l.params[0], l.params[1], l.params[2] * bias, l.params[3]];
        for c in [[0.0, -1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 1.0]] {
            a.push(l.a);
            b.push(l.b);
            color.push(l.color);
            params.push(p);
            corner.push(c);
        }
        let base = (i * 4) as u32;
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, a)
        .with_inserted_attribute(ATTRIBUTE_B, b)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, color)
        .with_inserted_attribute(ATTRIBUTE_PARAMS, params)
        .with_inserted_attribute(ATTRIBUTE_CORNER, corner)
        .with_inserted_indices(Indices::U32(idx))
}
