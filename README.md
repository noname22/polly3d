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

The live viewer is a read-only window that mirrors every edit. It is not shown
by default: ask the agent to open it (`show_viewer`) when you want to watch,
and to close it (`hide_viewer`) when you don't. Closing the window yourself has
the same effect as `hide_viewer`. The window is built with egui. With several
scenes it opens on an overview of thumbnails: click one to view that scene
alone, and use `< Back` (or Backspace/Esc) to return. In the single-scene view,
the toolbar switches views, overlays and texture filtering. Left-drag orbits,
the mouse wheel zooms, and the shortcuts are `1`–`4` (front/right/top/persp),
`W` (wireframe), `G` (grid), `S` (selection highlight), `F` (cycle texture
filter) and `R` (reset view).

When a scene has animation clips or morph targets, an animation panel appears
below the view:
- Play/Pause (Space) and Stop.
- A timeline of the selected clip: click or drag to scrub. Drag the blue
  triangles to move the loop region.
- One row per clip, with a checkbox to play it and a blend weight slider.
  Several checked clips blend. Click a clip's name to show it on the timeline.
- A weight slider per morph target.

Playback, clip weights and morph sliders are a preview and don't change the
scene. Moving a loop handle edits the clip, so the agent sees it and it can be
undone. The camera frames the rest pose, so it holds still while clips play.

Register it with Claude Code:

```sh
claude mcp add polly -- /path/to/polly3d/target/release/polly
```

Logs go to stderr (`RUST_LOG=polly=debug`), because stdout carries the MCP
protocol. Set `POLLY_ADAPTER=software` to force the software (llvmpipe)
renderer.

## Crates

| crate | purpose |
|---|---|
| `polly-core` | n-gon mesh with stable ids, primitives, modelling ops (extrude, inset, bevel, loop cut, subdivide, dissolve, bridge, mirror…), selection queries, scene, undo history, skeletons, skinning, morphs, clips |
| `polly-uv` | seams, projections, LSCM unwrap, pixel-snapped packing, UV reports |
| `polly-paint` | palettes, crisp pixel drawing, ordered dithering, procedural fills (bricks, planks, noise…), face fills, quantize |
| `polly-render` | wgpu offscreen renderer, ID-labelled overlays, contact sheets, CPU picking, UV layout view |
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
variants: sets of objects, possibly empty (`none`). One variant per slot is
active in Polly, and one is the default, meant to be visible when the
exported model is loaded. A variant can hide named parts of other objects
while active, such as a helmet hiding `hair_top`, to avoid poke-through.
Renders accept a variant choice, `render_variants` shows all options of a
slot, and the viewer has a dropdown per slot (a preview that doesn't change
the scene).

The GLB export is made for games that swap gear at runtime. The file contains
one shared skin per armature, every variant as its own node (non-defaults
hidden with `KHR_node_visibility`), hideable parts as separate nodes, and the
slot table in `extras`. See [docs/gltf-extras.md](docs/gltf-extras.md) for the
format and [docs/bevy.md](docs/bevy.md) for using it in Bevy.

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
