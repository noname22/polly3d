//! Agent-friendly colours: tool args may give colours as "#rrggbb",
//! "#rrggbbaa", names ("black"), or palette indices ("p3"); they are turned
//! into the `[r, g, b, a]` arrays the paint crate expects.

use polly_core::scene::Scene;
use polly_paint::Rgba;
use serde_json::{Map, Value, json};

/// Convert colour strings to `[r, g, b, a]` arrays wherever the tool's
/// (unwidened) input schema expects an RGBA colour. Errors name the bad value.
pub fn normalize(v: &mut Value, schema: &Value, palette: Option<&[Rgba]>) -> anyhow::Result<()> {
    let defs = schema.get("$defs").cloned().unwrap_or(Value::Null);
    walk(v, schema, &defs, palette, 0)
}

fn walk(v: &mut Value, schema: &Value, defs: &Value, palette: Option<&[Rgba]>, depth: u32) -> anyhow::Result<()> {
    if depth > 32 {
        return Ok(());
    }
    let Some(sm) = schema.as_object() else { return Ok(()) };
    if let Some(r) = sm.get("$ref").and_then(Value::as_str) {
        let name = r.rsplit('/').next().unwrap_or_default();
        if let Some(d) = defs.get(name) {
            walk(v, d, defs, palette, depth + 1)?;
        }
    }
    if is_rgba_schema(sm) {
        // "" is left alone: tools use it to remove an optional colour.
        if let Some(s) = v.as_str().filter(|s| !s.is_empty()) {
            let c = polly_paint::parse_color(s, palette).map_err(|e| anyhow::anyhow!("bad colour '{s}': {e}"))?;
            *v = json!(c);
        }
        return Ok(());
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        for branch in sm.get(key).and_then(Value::as_array).into_iter().flatten() {
            walk(v, branch, defs, palette, depth + 1)?;
        }
    }
    match v {
        Value::Object(m) => {
            if let Some(props) = sm.get("properties").and_then(Value::as_object) {
                for (k, x) in m.iter_mut() {
                    if let Some(ps) = props.get(k) {
                        walk(x, ps, defs, palette, depth + 1)?;
                    }
                }
            }
            if let Some(ap) = sm.get("additionalProperties").filter(|a| a.is_object()) {
                for x in m.values_mut() {
                    walk(x, ap, defs, palette, depth + 1)?;
                }
            }
        }
        Value::Array(a) => {
            let prefix = sm.get("prefixItems").and_then(Value::as_array);
            for (i, x) in a.iter_mut().enumerate() {
                let item = match (prefix.and_then(|p| p.get(i)), sm.get("items")) {
                    (Some(p), _) => Some(p),
                    (None, Some(Value::Array(items))) => items.get(i),
                    (None, Some(it)) => Some(it),
                    _ => None,
                };
                if let Some(it) = item {
                    walk(x, it, defs, palette, depth + 1)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
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
    let is_array = match m.get("type") {
        Some(Value::String(t)) => t == "array",
        Some(Value::Array(ts)) => ts.contains(&json!("array")),
        _ => false,
    };
    is_array
        && m.get("minItems") == Some(&json!(4))
        && m.get("maxItems") == Some(&json!(4))
        && m.get("items").and_then(|i| i.get("format")) == Some(&json!("uint8"))
}
