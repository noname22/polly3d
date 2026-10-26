//! Render options. These are deserialized directly from MCP tool arguments, so
//! every field has a serde default.

use polly_core::scene::{Filter, ObjectId};
use serde::{Deserialize, Serialize};

/// Camera view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum View {
    /// Orthographic, looking down -Z (camera on +Z). Screen: +X right, +Y up.
    Front,
    /// Orthographic, looking down +Z (camera on -Z). Screen: -X right, +Y up.
    Back,
    /// Orthographic, camera on -X looking +X. Screen: +Z right, +Y up.
    Left,
    /// Orthographic, camera on +X looking -X. Screen: -Z right, +Y up.
    Right,
    /// Orthographic, camera above looking down -Y. Screen: +X right, -Z up.
    Top,
    /// Orthographic, camera below looking up +Y. Screen: +X right, +Z up.
    Bottom,
    /// Perspective 3/4 view from front-right-above (yaw 35, pitch 25).
    #[default]
    Persp,
    /// Orthographic isometric view from front-right-above.
    Iso,
    /// Explicit camera. Ortho custom views are auto-sized to fit the focus bounds.
    Custom {
        eye: [f32; 3],
        target: [f32; 3],
        #[serde(default)]
        up: Option<[f32; 3]>,
        #[serde(default)]
        fov_deg: Option<f32>,
        #[serde(default)]
        ortho: bool,
    },
    /// Perspective orbit around the focus centre. yaw 0 = camera on +Z (front),
    /// yaw 90 = camera on +X (right); pitch > 0 = above. When `distance` is omitted
    /// the bounding sphere is framed, so the distance is the same at every angle.
    Orbit {
        #[serde(default)]
        yaw_deg: f32,
        #[serde(default)]
        pitch_deg: f32,
        #[serde(default)]
        distance: Option<f32>,
    },
}

impl View {
    pub fn name(&self) -> &'static str {
        match self {
            View::Front => "front",
            View::Back => "back",
            View::Left => "left",
            View::Right => "right",
            View::Top => "top",
            View::Bottom => "bottom",
            View::Persp => "persp",
            View::Iso => "iso",
            View::Custom { .. } => "custom",
            View::Orbit { .. } => "orbit",
        }
    }
}

fn t() -> bool {
    true
}
fn default_size() -> u32 {
    512
}
fn one() -> f32 {
    1.0
}

/// Overlays drawn on top of the shaded model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct Overlays {
    /// Polygon edges (hidden edges are not drawn). Default true.
    pub wireframe: bool,
    /// Vertex markers.
    pub verts: bool,
    /// `f<id>` labels at face centres.
    pub face_ids: bool,
    /// `v<id>` labels next to vertices.
    pub vert_ids: bool,
    /// `e<a>-<b>` labels at edge midpoints.
    pub edge_ids: bool,
    /// Highlight each object's selection (faces tinted orange, edges/verts bright). Default true.
    pub selection: bool,
    /// UV seam edges (red).
    pub seams: bool,
    /// Sharp edges (cyan).
    pub sharp: bool,
    /// Face normals as short lines from face centres.
    pub normals: bool,
    /// Ground grid on y=0 with X (red) and Z (blue) axes. Default true.
    pub grid: bool,
    /// Per-object bounding boxes.
    pub bounds: bool,
    /// Object name labels.
    pub object_names: bool,
    /// Skeleton bones (drawn on top of the mesh, head as a dot).
    pub bones: bool,
    /// Bone name labels (implies nothing else; combine with `bones`).
    pub bone_names: bool,
    /// Also label elements that face away from / are hidden from the camera.
    pub backface_labels: bool,
    /// Reference images, in the front/right/top view they belong to. Default true.
    pub references: bool,
}

impl Default for Overlays {
    fn default() -> Self {
        Overlays {
            wireframe: true,
            verts: false,
            face_ids: false,
            vert_ids: false,
            edge_ids: false,
            selection: true,
            seams: false,
            sharp: false,
            normals: false,
            grid: true,
            bounds: false,
            object_names: false,
            bones: false,
            bone_names: false,
            backface_labels: false,
            references: true,
        }
    }
}

impl Overlays {
    /// No overlays at all (plain shaded image).
    pub fn none() -> Self {
        Overlays { wireframe: false, selection: false, grid: false, ..Default::default() }
    }
    pub fn any_labels(&self) -> bool {
        self.face_ids || self.vert_ids || self.edge_ids || self.object_names || self.bone_names
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct RenderOptions {
    pub view: View,
    /// Image width in pixels (64..2048, default 512).
    #[serde(default = "default_size")]
    pub width: u32,
    /// Image height in pixels (64..2048, default 512).
    #[serde(default = "default_size")]
    pub height: u32,
    /// Objects to frame (default: all visible objects).
    pub focus: Option<Vec<ObjectId>>,
    /// Zoom factor relative to the auto-framed view (default 1; 2 = twice as close).
    #[serde(default = "one")]
    pub zoom: f32,
    pub overlays: Overlays,
    /// Override every texture's own filter (nearest / linear).
    pub filter: Option<Filter>,
    /// Render only the focus objects.
    pub isolate: bool,
    /// Background colour override (sRGB, 0..1 RGBA).
    pub background: Option<[f32; 4]>,
    /// Label text scale (1 = 8px font, 2 = 16px). Default: chosen from image size.
    pub label_scale: Option<u32>,
    /// Draw the corner legend (view name, axis hints, axis gizmo). Default true.
    #[serde(default = "t")]
    pub legend: bool,
    /// Frame these world bounds instead of the scene's (e.g. the rest pose, so
    /// the camera holds still while an animation plays). Not an agent option.
    #[serde(skip)]
    #[schemars(skip)]
    pub frame_bounds: Option<[[f32; 3]; 2]>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            view: View::default(),
            width: 512,
            height: 512,
            focus: None,
            zoom: 1.0,
            overlays: Overlays::default(),
            filter: None,
            isolate: false,
            background: None,
            label_scale: None,
            legend: true,
            frame_bounds: None,
        }
    }
}

impl RenderOptions {
    pub fn with_view(mut self, view: View) -> Self {
        self.view = view;
        self
    }
    pub fn with_size(mut self, w: u32, h: u32) -> Self {
        self.width = w;
        self.height = h;
        self
    }
    pub(crate) fn clamped_size(&self) -> [u32; 2] {
        [self.width.clamp(64, 2048), self.height.clamp(64, 2048)]
    }
}
