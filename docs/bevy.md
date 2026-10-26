# Using Polly models in Bevy

This page covers loading a Polly GLB with swappable gear in **Bevy 0.20**,
showing only the default loadout, and swapping variants at runtime. For the
file layout and metadata, see [gltf-extras.md](gltf-extras.md).

> **Status:** these snippets were written against the `bevy_gltf` /
> `bevy_world_serialization` 0.20.0 sources. They have not been compiled or
> run. Treat them as a starting point; a ready-made `bevy_polly` crate may
> follow.

## What Bevy does with the file

- **Nodes:** every glTF node becomes an entity with a `Transform`, a `Visibility` and a `Name` (the node name). Its `extras` arrive as a `GltfExtras` component, holding the JSON as a string.
- **Skins and animation:** skins are supported, so one skeleton drives the body and its skinned gear, and gear attached to bones follows them. Clips load as `AnimationClip`s, named after the clip.
- **Visibility:** Bevy 0.20 does **not** support `KHR_node_visibility`, so all variants are visible after loading until you hide them. Its own supported-extensions table lists it as ❌.

## Loading and hiding the alternatives

Spawn the scene and observe `WorldInstanceReady`. The instance is complete at
that point, so the observer can hide non-default variants and default-hidden
parts before anything is shown.

```rust
use bevy::prelude::*;
use bevy::gltf::{GltfAssetLabel, GltfExtras};
use bevy::world_serialization::{WorldAssetRoot, WorldInstanceReady};
use serde_json::Value;

fn spawn_knight(mut commands: Commands, assets: Res<AssetServer>) {
    commands
        .spawn(WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset("knight.glb"))))
        .observe(apply_default_loadout);
}

/// Hide everything Polly marked as hidden by default.
fn apply_default_loadout(
    ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    mut nodes: Query<(&GltfExtras, &mut Visibility)>,
) {
    for e in children.iter_descendants(ready.entity) {
        let Ok((extras, mut vis)) = nodes.get_mut(e) else { continue };
        let Ok(json) = serde_json::from_str::<Value>(&extras.value) else { continue };
        if json["polly"]["default"] == Value::Bool(false) {
            *vis = Visibility::Hidden;
        }
    }
    // Parts hidden by default variants: see `apply_loadout` below, called with
    // each slot's default from the scene extras.
}
```

The scene-level slot table (`scenes[0].extras.polly`) arrives as a
`GltfSceneExtras` component. You can read it there, or keep your own list of
slots in game code; the variant node names are stable.

## Swapping variants

A loadout is one variant per slot. To apply it, show the chosen variants'
nodes, hide the rest, then hide the parts the chosen variants hide:

```rust
use std::collections::{HashMap, HashSet};

/// slot -> chosen variant
#[derive(Component, Default)]
pub struct Loadout(pub HashMap<String, String>);

/// Show `loadout` on the character spawned at `root`. `slots` is the parsed
/// `scenes[0].extras.polly.slots` object.
pub fn apply_loadout(
    root: Entity,
    loadout: &Loadout,
    slots: &Value,
    children: &Query<&Children>,
    nodes: &mut Query<(&Name, Option<&GltfExtras>, &mut Visibility)>,
) {
    // Part nodes hidden by the chosen variants.
    let mut hidden_parts = HashSet::new();
    for (slot, variant) in &loadout.0 {
        for h in slots[slot]["hides"][variant].as_array().into_iter().flatten() {
            hidden_parts.insert(h[2].as_str().unwrap_or_default().to_string());
        }
    }
    for e in children.iter_descendants(root) {
        let Ok((name, extras, mut vis)) = nodes.get_mut(e) else { continue };
        let polly = extras.and_then(|x| serde_json::from_str::<Value>(&x.value).ok()).map(|j| j["polly"].clone());
        match polly {
            // A variant's node: visible when its variant is chosen.
            Some(p) if p["slot"].is_string() => {
                let chosen = loadout.0.get(p["slot"].as_str().unwrap()).map(String::as_str);
                *vis = if chosen == p["variant"].as_str() { Visibility::Inherited } else { Visibility::Hidden };
            }
            // A hideable part's node.
            Some(p) if p["part"].is_string() => {
                *vis = if hidden_parts.contains(name.as_str()) { Visibility::Hidden } else { Visibility::Inherited };
            }
            _ => {}
        }
    }
}
```

Use `Visibility::Inherited`, not `Visible`, when showing a node, so hiding the
whole character still works.

## Alternative: hide during loading

Bevy 0.20 has a `GltfExtensionHandler` trait (in `bevy::gltf::extensions`). Its
`on_gltf_node(&mut self, load_context, gltf_node, entity: &mut EntityWorldMut)`
hook runs while each node's entity is built. A handler can read
`gltf_node.extras()` or the `KHR_node_visibility` extension and insert
`Visibility::Hidden` right there. Handlers are registered through the
`GltfExtensionHandlers` resource. This avoids the observer, at the cost of
working with a lower-level API.

## Animation

Animations are plain glTF animations. Play them with `AnimationPlayer` and
`AnimationGraph`, as in Bevy's `animated_mesh` example. One clip drives the
body, skinned gear and attached gear together. Polly's loop region is not
applied by Bevy; read it from the clip's extras (`animations[i].extras.polly`,
in seconds) and seek back to `loop_start` when playback passes `loop_end`.
