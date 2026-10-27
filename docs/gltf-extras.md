# Polly's glTF export: structure and `extras` (schema version 1)

Polly exports characters with swappable gear as **one GLB**. Every variant is
in the file, the defaults are visible, and the alternatives are hidden. An
engine swaps gear by toggling node visibility. This page describes the node
layout and the metadata Polly writes, so any engine or tool can use it.

## Node layout

```
Rig                      armature: the object with the skeleton
├── spine                joint nodes, one per bone (rest pose)
│   └── head
│       ├── Helmet       rigid gear attached to a bone: child of its joint
│       └── HairShort
│           └── HairShort.hair_top   a hideable part: its own node and mesh
├── Body                 skinned meshes: children of the armature, identity transform
│   ├── Body.under_Blouse+under_Blazer   skin covered by both garments
│   └── Body.under_Blazer                skin covered only by the blazer
└── Skirt
```

- **Skins:**
  - There is **one skin per armature**, shared by every mesh skinned to it.
  - The skin's `skeleton` is the armature node.
  - Skinned meshes store their vertices in the armature's space. Per the glTF spec, a skinned mesh node's own transform doesn't affect it; Polly places those nodes under the armature with an identity transform anyway.
  - Vertices without usable weights are bound to an extra joint, `<armature>_root`, which is never animated.
- **Bone-attached gear** (helmets, weapons) is a child of its joint node, so it follows the bone when animated.
- **Hideable parts:** faces that some variant hides become separate child nodes, with their own mesh, the same skin and the same morph targets. Hiding such a node hides just those faces.
  - Faces are grouped by *which hide rules* cover them. A node is named `<object>.<rule>`, or `<object>.<rule1>+<rule2>` for faces that several rules hide (skin under both a blouse and a blazer).
  - A rule is a labelled part (`hair_top`) or a face set found by `auto_hide` (`under_Blouse`).
- **Animations:** every clip is a glTF animation.
  - Bone channels target joint nodes.
  - Morph-weight channels target the object's node and its part nodes.
  - Clip loop settings are in `animations[i].extras.polly`: `length`, `fps`, `looping`, `loop_start`, `loop_end` (in seconds).

## Default visibility

The nodes of **non-default variants**, and parts **hidden by a default
variant**, carry the Khronos extension `KHR_node_visibility`:

```json
"extensions": { "KHR_node_visibility": { "visible": false } }
```

The extension is listed in `extensionsUsed` and never in
`extensionsRequired`, so importers that don't support it still load the file.
Engines that do support it, such as Godot 4.5+, show only the default loadout.
Engines that don't, such as Bevy 0.20, show everything until the metadata
below is applied; see [bevy.md](bevy.md).

## `extras`

All Polly metadata sits under a `"polly"` key, so it can share `extras` with
other tools.

### On each variant's node

```json
"extras": { "polly": { "slot": "top", "variants": ["blouse", "suit"], "default": true } }
```

An object can belong to several variants of one slot (a blouse is in both `blouse` and `suit`). `variants` lists them all, and `default` says whether the slot's default variant includes the object.

### On each hideable part's node

```json
"extras": { "polly": { "object": "Body", "parts": ["under_Blazer", "under_Blouse"] } }
```

`parts` lists the hide rules that cover this node's faces.

### On a material with a custom shader

```json
"extras": { "polly": { "shader": { "name": "toon", "params": { "bands": [3], "tint": [1, 0.8, 0.6, 1] } } } }
```

The material is a normal glTF PBR material; `shader` names a fragment shader
(`shaders/<name>.wesl` in the game's assets) that runs on top of it. Each
parameter is 1–4 numbers. The `bevy_polly` crate turns these into
`PollyMaterial`s; see [bevy.md](bevy.md#custom-shaders).

### On the scene (`scenes[0].extras`)

```json
"extras": { "polly": {
  "version": 1,
  "slots": {
    "hair": {
      "default": "short",
      "variants": { "short": ["HairShort"], "long": ["HairLong"] }
    },
    "head": {
      "default": "helmet",
      "variants": { "helmet": ["Helmet"], "none": [] },
      "hides": { "helmet": [["HairShort", "hair_top", "HairShort.hair_top"]] }
    }
  }
} }
```

- **`variants`:** maps each variant to the **node names** of its objects. A variant can be empty (`none`).
- **`hides`:** maps a variant to the nodes it hides while chosen, as `[object, rule, node name]`. A rule whose faces overlap other rules appears once per node.

### Showing a loadout

To show a choice of one variant per slot:

1. For every slot, show the nodes of the chosen variant's objects and hide the slot's other variant nodes. A node with `variants` is shown when that list contains the slot's choice.
2. Collect the `hides` of all chosen variants. Hide those part nodes, and show every other part node.

Object and part node names are unique within the file.

## Versioning

`version` changes only when the meaning of existing fields changes. New
fields can appear in version 1, so readers should ignore keys they don't know.
