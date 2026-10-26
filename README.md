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
the same effect as `hide_viewer`. Controls: left-drag to
orbit, mouse wheel to zoom, `1`–`4` for the front/right/top/persp views, `W`
toggles the wireframe, `G` the grid, `S` the selection highlight, `F` cycles the
texture filter (per-texture / nearest / linear), and `R` resets the camera.

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
| `polly-core` | n-gon mesh with stable ids, primitives, modelling ops (extrude, inset, bevel, loop cut, subdivide, dissolve, bridge, mirror…), selection queries, scene, undo history |
| `polly-uv` | seams, projections, LSCM unwrap, pixel-snapped packing, UV reports |
| `polly-paint` | palettes, crisp pixel drawing, ordered dithering, procedural fills (bricks, planks, noise…), face fills, quantize |
| `polly-render` | wgpu offscreen renderer, ID-labelled overlays, contact sheets, CPU picking, UV layout view |
| `polly-export` | GLB export (textures embedded, sampler filter preserved), `.polly` project files |
| `polly-mcp` | MCP tool layer (rmcp) |
| `polly` | binary: stdio server + optional viewer |

Subagents share the session's single `polly` process, so they all see one
scene. Each tool call is atomic, but the active object, the selection, the undo
history and the last render camera are shared too. Parallel subagents should
each work on their own object and pass explicit `object` and element ids.

## Conventions the agent sees

- Y up, right-handed. The "front" view looks from +Z.
- Element ids are `v12` (vertex), `f3` (face) and `e3-9` (edge). Ids are never reused.
- Edit tools act on the object's selection when `elements` is omitted. Tools that create geometry select their result.
- UVs follow glTF: (0,0) is the top-left of the texture.
- Colours can be `"#rrggbb"`, `"#rrggbbaa"`, a palette index such as `"p3"`, a name, or `[r,g,b,a]`.

## Tests

```sh
cargo test --workspace
```
