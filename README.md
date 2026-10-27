# Polly3D

A low-poly 3-D modeller with no editing UI: an AI agent drives it entirely
over [MCP](https://modelcontextprotocol.io). The agent builds and edits meshes,
queries geometry, looks at annotated screenshots, unwraps UVs, paints small
palette textures and exports glTF. The target look is "modern retro". The
renderer has one shading model. PS1, N64 or clean low-poly looks come from the
assets themselves: low poly counts, small nearest-filtered palette textures,
vertex colours, and flat or smooth shading.

## Build & run

```sh
cargo build --release
./target/release/polly              # MCP server on stdio
./target/release/polly --viewer     # also open the live window at startup
./target/release/polly --no-viewer  # never open a window (no display connection)
./target/release/polly --project scene.polly
```

The live viewer is a read-only window that mirrors every edit. Nothing you do
in it changes the scene: its controls are previews that affect only that
window, so you can inspect the agent's work without ever interfering with it. It is not shown
by default: ask the agent to open it (`show_viewer`) when you want to watch,
and to close it (`hide_viewer`) when you don't. Closing the window yourself has
the same effect as `hide_viewer`. The window is built with egui. With several
scenes it opens on an overview of thumbnails: click one to view that scene
alone, and use `< Back` (or Backspace/Esc) to return. In the single-scene view,
the toolbar switches views, overlays and texture filtering. Left-drag orbits,
the mouse wheel zooms, and the shortcuts are `1`–`4` (front/right/top/persp; Ctrl+`1`–`3` for back/left/bottom),
`W` (wireframe), `G` (grid), `S` (selection highlight), `F` (cycle texture
filter) and `R` (reset view).

When a scene has animation clips or morph targets, an animation panel appears
below the view:
- Play/Pause (Space) and Stop.
- A timeline of the selected clip: click or drag to scrub. Drag the blue
  triangles to try another loop region; "Reset loop" goes back to the clip's.
- One row per clip, with a checkbox to play it and a blend weight slider.
  Several checked clips blend. Click a clip's name to show it on the timeline.
- A weight slider per morph target.

Playback, clip weights, morph sliders and loop regions are previews and don't
change the scene. The camera frames the rest pose, so it holds still while
clips play.

Register it with Claude Code:

```sh
claude mcp add polly -- /path/to/polly3d/target/release/polly
```

Logs go to stderr (`RUST_LOG=polly=debug`), because stdout carries the MCP
protocol. Rendering is done by one headless Bevy app running on a background
thread; it never opens a window, and the viewer only displays its images. Set
`POLLY_ADAPTER=software` to force the software (llvmpipe) adapter.

## Crates

| crate | purpose |
|---|---|
| `polly-core` | n-gon mesh with stable ids, primitives, modelling ops (extrude, inset, bevel, loop cut, subdivide, dissolve, bridge, mirror…), selection queries, scene, undo history, skeletons, skinning, morphs, clips |
| `polly-uv` | seams, projections, LSCM unwrap, pixel-snapped packing, UV reports |
| `polly-paint` | palettes, crisp pixel drawing, ordered dithering, procedural fills (bricks, planks, noise…), face fills, quantize |
| `polly-render` | CPU render helpers: cameras and framing, picking, overlay line lists, ID labels and legend, contact sheets, UV layout view |
| `polly-bevy` | the renderer: a headless Bevy app on a worker thread (`StandardMaterial` shading, overlay lines, readback), shared by the agent's renders and the viewer |
| `polly-export` | GLB export (textures, sampler filter, skins, morph targets, animations), `.polly` project files |
| `polly-mcp` | MCP tool layer (rmcp) |
| `polly` | binary: stdio server + optional egui viewer |

## Scenes and subagents

Subagents share the session's single `polly` process, but it holds any number
of named **scenes**. Each scene has its own objects, selection, undo history and
render camera, and calls in different scenes run in parallel. Every tool takes
an optional `scene` argument; the default is `main`.

To build unrelated models at once, for example "make 10 props with
subagents", the agent calls `create_scene` once per model and each subagent
passes its `scene` on every call. `list_scenes` and `render_overview` (a
captioned grid of all scenes) show the results. `delete_scene` removes a
scene, and `clear_scene` empties one.

## Rigging and animation

Objects can have a skeleton: bones given by head/tail points with parents. Each
vertex has up to 4 skin weights, set by hand or computed automatically (rigid,
the classic one-bone-per-part look, or smooth blending at joints). Morph targets
(shape keys) are authored by moving vertices between `begin_morph` and
`end_morph`. Clips hold keyframes for bones, object transforms and morph
weights, with linear or step interpolation and a loop region. Any number of
clips can be blended by weight. Renders can show bones, a weight heatmap for one
bone, a pose at any frame, or a filmstrip of frames. GLB export includes skins,
morph targets and animations; loop regions go in the animation's extras.
Deformation happens on the CPU and produces a display copy of the scene, so
element ids, picking and painting all work on the posed model.

A skeleton can live on one object, an armature, with other meshes bound to it
(`bind_to_armature`). One pose and one set of clips then deform the body and
all its gear together. Bone, pose and clip tools accept a bound mesh and act
on its armature, while skin weights stay on each mesh. Rigid gear is attached
to a bone (`attach_to_bone`) and follows it; fitted gear gets the body's
weights with `transfer_weights`.

## Gear slots and variants

Swappable gear is organised in slots (`hair`, `head`, `weapon`...), each with
variants: sets of objects, possibly empty (`none`). An object can be in
several variants of one slot (`suit` = blouse + blazer). One variant per slot
is active in Polly, and one is the default, meant to be visible when the
exported model is loaded. A variant can hide parts of other objects while
active, such as a helmet hiding `hair_top`, to avoid poke-through.

Clothing is made from the body: `shell` copies body faces into a new object
pushed out along the normals (rigged like the body), `inflate` fattens or
shrinks a surface, and `auto_hide` finds the skin a garment covers and hides
it whenever the garment is shown. Overlapping garments each keep their own
rule.
Renders accept a variant choice, `render_variants` shows all options of a
slot, and the viewer has a dropdown per slot (a preview that doesn't change
the scene).

The GLB export is made for games that swap gear at runtime. The file contains
one shared skin per armature, every variant as its own node (non-defaults
hidden with `KHR_node_visibility`), hideable parts as separate nodes, and the
slot table in `extras`. See [docs/gltf-extras.md](docs/gltf-extras.md) for the
format and [docs/bevy.md](docs/bevy.md) for using it in Bevy.

## Materials

Materials follow glTF's metallic-roughness model. A retro asset uses only a
base colour and a small nearest-filtered texture (optionally unlit); a modern
one adds any of:
- `metallic`, `roughness` and a `metallic_roughness_texture` (G = roughness, B = metallic);
- a `normal_texture` (with `normal_scale`) and an `occlusion_texture` (with `occlusion_strength`);
- `emissive` colour, `emissive_strength` and an `emissive_texture`;
- `blend` (alpha blending; exclusive with `alpha_cutoff`), `double_sided`, `unlit`;
- `clearcoat`, `anisotropy`, `transmission` (with `ior` and `thickness`) and `specular`.

Textures are sRGB colour by default. Normal, metallic-roughness and occlusion
maps are linear data: `create_texture` with `kind: "normal"` or `"data"` makes
one (and `slot` wires it into a material), `import_texture` takes `linear`, and
a texture placed in a data slot becomes linear. Textures can be up to 4096 px
per side. Export writes every channel and the matching `KHR_materials_*`
extensions, and adds MikkTSpace tangents to normal-mapped meshes. Renders use
Bevy's `StandardMaterial`, so every channel shows up in the agent's images and
the viewer just as it will in the game.

## Lighting and effects

Lighting is a scene setting (`set_lighting`), saved with the project; a
`render` can override it for one image. The default is polly's classic
camera-relative key, fill and ambient lights with no effects, which suits
modelling and retro assets. For modern assets:
- `environment`: image-based lighting and reflections from a built-in `"studio"` or `"sky"`, or an equirectangular `.hdr` file; `show_environment` uses it as the background.
- `sun`: a world-fixed directional light; `shadows` makes it and the key light cast shadows.
- `ambient_occlusion` (screen-space), `tonemapping` (`agx`, `tony_mc_mapface`, `aces_fitted`, `reinhard`, `blender_filmic`), `exposure` in stops and `bloom`.

These change the light and the camera, not the shading model: materials look
the same as in Bevy under the same light. The viewer shows the scene's lighting.

## Reference images

A reference image (a character sheet, a blueprint) is shown behind or over the
model in one of the six axis views (front, back, left, right, top, bottom). It is placed in world units, so it zooms
with the model and stays put while the camera re-frames. `crop` picks one view
out of a multi-view sheet, and `height` scales it to the subject's real size,
or `calibrate` places it by two known points (the ground row, the top of the
head). `measure_reference` reads the drawing's edges in metres at any height,
and `compare_reference` diffs the model's outline against the drawing.
The agent manages references with `add_reference`, `update_reference`,
`remove_reference` and `list_references`; they appear in its `render` of that
view and are saved in the project file.

In the viewer:
- The References panel shows or hides each of the agent's images, switches it between Behind and On top, and sets its opacity. These settings affect only the viewer; "Reset" returns to the agent's.
- "Add reference…" in the toolbar loads your own image for the current axis view. Dropping an image file on the window does the same on X11, macOS and Windows; winit doesn't support file drops on Wayland. Your images are listed under "Mine": the agent never sees them, and they aren't saved in the project.

## Conventions the agent sees

- Y up, right-handed. The "front" view looks from +Z.
- Element ids are `v12` (vertex), `f3` (face) and `e3-9` (edge). Ids are never reused.
- Element arguments also take ranges (`f400..f507`, meaning the faces that exist in that range) and comma lists in one string. Results report ids the same compact way, so they can be pasted back.
- Faces can carry a part name. `join_objects` names parts after the joined objects, `set_part` labels faces, and `select {"by":"part"}` finds them. Faces created by edits inherit their source face's part.
- Rotations are Euler degrees with R = Rx·Ry·Rz: Z first, then Y, then X about the parent's axes.
- Edit tools act on the object's selection when `elements` is omitted. Tools that create geometry select their result.
- UVs follow glTF: (0,0) is the top-left of the texture.
- Colours can be `"#rrggbb"`, `"#rrggbbaa"`, a palette index such as `"p3"`, a name, or `[r,g,b,a]`.

## Tests

```sh
cargo test --workspace
```
