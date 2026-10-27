// Overlay lines and points (wireframe, grid, selection, bones...), expanded
// into screen-space quads. Port of the old polly renderer's `vs_line`.
// Positions are world space; Bevy's depth is reverse-Z (near = 1).
import bevy_pbr::render::mesh_view_bindings::view;

struct Vertex {
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
    @location(2) color: vec4<f32>,
    // x = width px, y = 1 for a point (square), z = depth pull toward the eye, w = 1 to draw on top
    @location(3) params: vec4<f32>,
    // x: 0 at a, 1 at b; y: -1 / +1 across the line
    @location(4) corner: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn is_ortho() -> bool {
    return view.clip_from_view[3][3] == 1.0;
}

fn pull(p: vec3<f32>, k: f32) -> vec3<f32> {
    if (is_ortho()) {
        let forward = -normalize(view.world_from_view[2].xyz);
        return p - forward * k;
    }
    return p + (view.world_position - p) * k;
}

@vertex
fn vertex(l: Vertex) -> VertexOutput {
    var o: VertexOutput;
    o.color = l.color;
    var ca = view.clip_from_world * vec4<f32>(pull(l.a, l.params.z), 1.0);
    var cb = view.clip_from_world * vec4<f32>(pull(l.b, l.params.z), 1.0);
    let eps = 1e-5;
    if (ca.w < eps && cb.w < eps) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    if (ca.w < eps) {
        ca = mix(ca, cb, (eps - ca.w) / (cb.w - ca.w));
    }
    if (cb.w < eps) {
        cb = mix(cb, ca, (eps - cb.w) / (ca.w - cb.w));
    }
    let na = ca.xyz / ca.w;
    let nb = cb.xyz / cb.w;
    let half_px = view.viewport.zw * 0.5;
    let c = l.corner;
    let hw = l.params.x * 0.5;
    var ndc: vec2<f32>;
    var z: f32;
    if (l.params.y > 0.5) {
        ndc = na.xy + vec2<f32>(c.x * 2.0 - 1.0, c.y) * hw / half_px;
        z = na.z;
    } else {
        let d = (nb.xy - na.xy) * half_px;
        var dir = vec2<f32>(1.0, 0.0);
        if (length(d) > 1e-6) {
            dir = normalize(d);
        }
        let nrm = vec2<f32>(-dir.y, dir.x);
        let base = mix(na, nb, c.x);
        ndc = base.xy + (nrm * c.y * hw + dir * (c.x * 2.0 - 1.0) * hw) / half_px;
        z = base.z;
    }
    // On top of geometry (bones inside a mesh), keeping the relative order of such lines.
    if (l.params.w > 0.5) {
        z = 1.0 - (1.0 - z) * 0.001;
    }
    o.pos = vec4<f32>(ndc, clamp(z, 0.0, 1.0), 1.0);
    return o;
}

@fragment
fn fragment(i: VertexOutput) -> @location(0) vec4<f32> {
    return i.color;
}
