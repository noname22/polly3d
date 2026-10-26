//! Agent-friendly colours: tool args may give colours as "#rrggbb",
//! "#rrggbbaa", names ("black"), or palette indices ("p3"); they are turned
//! into the `[r, g, b, a]` arrays the paint crate expects.

use polly_core::scene::Scene;
use polly_paint::Rgba;
use serde_json::{Map, Value, json};

/// Keys whose string values are colours.
const COLOR_KEYS: &[&str] =
    &["color", "colors", "a", "b", "from", "to", "mortar", "brick", "gap", "grain", "fill", "background", "outline_color"];

fn looks_like_color(s: &str) -> bool {
    s.starts_with('#') || s.starts_with("palette:")
}

/// Recursively convert colour strings to arrays. Errors name the bad value.
pub fn normalize(v: &mut Value, palette: Option<&[Rgba]>) -> anyhow::Result<()> {
    fn walk(v: &mut Value, key: Option<&str>, palette: Option<&[Rgba]>) -> anyhow::Result<()> {
        match v {
            Value::String(s) => {
                let keyed = key.is_some_and(|k| COLOR_KEYS.contains(&k));
                if looks_like_color(s) || keyed {
                    match polly_paint::parse_color(s, palette) {
                        Ok(c) => *v = json!(c),
                        Err(e) if keyed || looks_like_color(s) => anyhow::bail!("bad colour '{s}': {e}"),
                        Err(_) => {}
                    }
                }
            }
            Value::Array(a) => {
                for x in a {
                    walk(x, key, palette)?;
                }
            }
            Value::Object(m) => {
                for (k, x) in m.iter_mut() {
                    walk(x, Some(k), palette)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    walk(v, None, palette)
}

pub fn palette_named(scene: &Scene, name: &str) -> Option<Vec<Rgba>> {
    scene.palettes.get(name).cloned().or_else(|| polly_paint::palette::builtin(name))
}

/// Make `[u8; 4]` colour arrays in a schema also accept strings.
pub fn widen_schema(v: &mut Value) {
    match v {
        Value::Object(m) => {
            if is_rgba_schema(m) {
                let orig = Value::Object(std::mem::take(m));
                *m = json!({
                    "description": "Colour: \"#rrggbb\", \"#rrggbbaa\", a palette index \"p3\" (texture's palette), a name like \"black\", or [r,g,b,a] 0-255",
                    "anyOf": [{"type": "string"}, orig],
                })
                .as_object()
                .unwrap()
                .clone();
                return;
            }
            for x in m.values_mut() {
                widen_schema(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(widen_schema),
        _ => {}
    }
}

fn is_rgba_schema(m: &Map<String, Value>) -> bool {
    m.get("type") == Some(&json!("array"))
        && m.get("minItems") == Some(&json!(4))
        && m.get("maxItems") == Some(&json!(4))
        && m.get("items").and_then(|i| i.get("format")) == Some(&json!("uint8"))
}
