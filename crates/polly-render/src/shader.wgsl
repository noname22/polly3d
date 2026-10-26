struct Globals {
    view_proj: mat4x4<f32>,
    // xyz = eye, w = 1 for orthographic
    eye: vec4<f32>,
    // xyz = camera forward, w = line depth bias (fraction of eye distance, or world units for ortho)
    forward: vec4<f32>,
    // xyz = direction towards key light, w = intensity
    key: vec4<f32>,
    fill: vec4<f32>,
    // x, y = viewport size, z = 1 to encode sRGB in the shader (non-sRGB target)
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;

struct Material {
    base: vec4<f32>,
    // x = unlit, y = alpha cutoff (< 0: none)
    params: vec4<f32>,
};

@group(1) @binding(0) var<uniform> mat: Material;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn encode(c: vec4<f32>) -> vec4<f32> {
    if (g.viewport.z > 0.5) {
        return vec4<f32>(to_srgb(clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0))), c.a);
    }
    return c;
}

struct MeshIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) sel: f32,
};

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) sel: f32,
    @location(4) world: vec3<f32>,
};

@vertex
fn vs_mesh(v: MeshIn) -> MeshOut {
    var o: MeshOut;
    o.clip = g.view_proj * vec4<f32>(v.pos, 1.0);
    o.normal = v.normal;
    o.uv = v.uv;
    o.color = v.color;
    o.sel = v.sel;
    o.world = v.pos;
    return o;
}

@fragment
fn fs_mesh(i: MeshOut, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
    let t = textureSample(tex, samp, i.uv);
    let c = mat.base * t * i.color;
    if (mat.params.y >= 0.0 && c.a < mat.params.y) {
        discard;
    }
    var rgb = c.rgb;
    if (mat.params.x < 0.5) {
        var n = normalize(i.normal);
        if (!ff) {
            n = -n;
        }
        var v = normalize(g.eye.xyz - i.world);
        if (g.eye.w > 0.5) {
            v = -g.forward.xyz;
        }
        // Sky/ground hemisphere ambient keeps up- and down-facing faces distinct.
        let hemi = mix(0.12, 0.30, n.y * 0.5 + 0.5);
        let key = max(dot(n, g.key.xyz), 0.0) * g.key.w;
        let fill = max(dot(n, g.fill.xyz), 0.0) * g.fill.w;
        let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.2;
        rgb = rgb * (hemi + key + fill + rim);
    }
    if (i.sel > 0.5) {
        rgb = mix(rgb, vec3<f32>(1.0, 0.33, 0.02), 0.5);
    }
    return encode(vec4<f32>(rgb, 1.0));
}

struct LineIn {
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
    @location(2) color: vec4<f32>,
    // x = width px, y = 1 for a point (square), z = depth bias multiplier, w = 1 to draw on top
    @location(3) params: vec4<f32>,
};

struct LineOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn bias_point(p: vec3<f32>, k: f32) -> vec3<f32> {
    if (g.eye.w > 0.5) {
        return p - g.forward.xyz * g.forward.w * k;
    }
    return p + (g.eye.xyz - p) * g.forward.w * k;
}

@vertex
fn vs_line(@builtin(vertex_index) vi: u32, l: LineIn) -> LineOut {
    var o: LineOut;
    o.color = l.color;
    var ca = g.view_proj * vec4<f32>(bias_point(l.a, l.params.z), 1.0);
    var cb = g.view_proj * vec4<f32>(bias_point(l.b, l.params.z), 1.0);
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
    let half_px = g.viewport.xy * 0.5;
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let c = corners[vi];
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
    // params.w > 0.5: draw on top of geometry (bones inside a mesh), keeping
    // the relative depth order between such lines.
    if (l.params.w > 0.5) {
        z = z * 0.001;
    }
    o.pos = vec4<f32>(ndc, clamp(z, 0.0, 1.0), 1.0);
    return o;
}

@fragment
fn fs_line(i: LineOut) -> @location(0) vec4<f32> {
    return encode(i.color);
}

// ---- reference images ----
// Uses the material group: mat.base.a = opacity.

struct RefIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct RefOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_ref(v: RefIn) -> RefOut {
    var o: RefOut;
    o.clip = g.view_proj * vec4<f32>(v.pos, 1.0);
    // Never clipped by near/far: draw order alone decides behind / on top.
    o.clip.z = 0.5 * o.clip.w;
    o.uv = v.uv;
    return o;
}

@fragment
fn fs_ref(i: RefOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, i.uv);
    return encode(vec4<f32>(c.rgb, c.a * mat.base.a));
}
