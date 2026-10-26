use std::collections::BTreeMap;

use anyhow::{Result, bail};
use polly_core::glam::Vec3;
use polly_core::ids::{self, Selection};
use polly_core::select::{Query, convert, run};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::common::*;
use crate::registry::{Ctx, Out, ToolDef, def};

pub fn tools() -> Vec<ToolDef> {
    vec![
        def(
            "describe_mesh",
            "Summary of one object's mesh: counts, local bounds, face sizes (tris/quads/ngons), materials used, seams/sharp counts, current selection and topology warnings.",
            true,
            describe_mesh,
        ),
        def(
            "list_faces",
            "List faces with vertex ids, center, normal, area, material and smooth flag (local coordinates). Paginated.",
            true,
            list_faces,
        ),
        def("list_verts", "List vertices with positions (local coordinates). Paginated.", true, list_verts),
        def("list_edges", "List edges with length, adjacent faces, seam/sharp flags. Paginated.", true, list_edges),
        def(
            "select",
            "Select elements by a query (normal direction, box, plane side, edge loop/ring, face loop, linked, material, part, boundary, sharp, nearest, grow, shrink, ids, all) and store it as the object's selection. Most edit tools act on the selection when `elements` is omitted. Returns the selected ids.",
            false,
            select,
        ),
        def(
            "measure",
            "Measure elements: 2 verts -> distance & delta; 2 faces -> angle between normals; otherwise bounding box, total face area and centroid.",
            true,
            measure,
        ),
    ]
}

#[derive(Deserialize, JsonSchema)]
struct ObjectArg {
    /// Object name or id like "o2" (default: active object).
    object: Option<String>,
}

/// Named parts with their face ids, e.g. {"head": "f400..f507"}.
pub fn parts_json(m: &polly_core::mesh::Mesh) -> serde_json::Value {
    let mut parts: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for (id, f) in &m.faces {
        if let Some(p) = &f.part {
            parts.entry(p).or_default().push(*id);
        }
    }
    let mut j = serde_json::Map::new();
    for (name, ids) in parts {
        j.insert(name.to_string(), json!(ids::compress("f", ids)));
    }
    let unlabelled = m.faces.values().filter(|f| f.part.is_none()).count();
    if !j.is_empty() && unlabelled > 0 {
        j.insert("(no part)".into(), json!(format!("{unlabelled} faces")));
    }
    j.into()
}

fn describe_mesh(ctx: &mut Ctx, a: ObjectArg) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let o = ctx.doc.scene.object(id)?;
    let m = &o.mesh;
    let mut sizes = BTreeMap::new();
    for f in m.faces.values() {
        let k = match f.len() {
            3 => "tris",
            4 => "quads",
            _ => "ngons",
        };
        *sizes.entry(k).or_insert(0) += 1;
    }
    let mut mats: BTreeMap<String, usize> = BTreeMap::new();
    for f in m.faces.values() {
        let name = ctx.doc.scene.materials.get(&f.material).map_or(format!("#{}", f.material), |m| m.name.clone());
        *mats.entry(name).or_default() += 1;
    }
    let mut j = json!({
        "object": o.name,
        "parts": parts_json(m),
        "verts": m.verts.len(),
        "edges": m.edges().len(),
        "faces": m.faces.len(),
        "triangles": m.triangle_count(),
        "face_sizes": sizes,
        "materials": mats,
        "seams": m.seams.len(),
        "sharp_edges": m.sharp.len(),
        "boundary_edges": m.boundary_edges().len(),
        "smooth_faces": m.faces.values().filter(|f| f.smooth).count(),
    });
    if let Some((lo, hi)) = m.bounds() {
        j["local_bounds"] = json!({"min": round3(lo), "max": round3(hi), "size": round3(hi - lo)});
    }
    if !o.selection.is_empty() {
        j["selection"] = o.selection.to_json();
    }
    let w = m.validate();
    j["warnings"] = json!(w);
    Ok(Out::json(j))
}

#[derive(Deserialize, JsonSchema)]
struct ListArgs {
    object: Option<String>,
    /// Restrict to these ids / "selection" (default: all).
    elements: Option<Vec<String>>,
    #[serde(default)]
    offset: usize,
    /// Max rows (default 100).
    limit: Option<usize>,
}

fn scope(ctx: &Ctx, a: &ListArgs) -> Result<(u32, Option<Selection>)> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let sel = match &a.elements {
        None => None,
        Some(_) => Some(elements(&ctx.doc.scene, id, &a.elements)?),
    };
    Ok((id, sel))
}

fn page<T: serde::Serialize>(rows: Vec<T>, offset: usize, limit: Option<usize>) -> serde_json::Value {
    let total = rows.len();
    let limit = limit.unwrap_or(100);
    let rows: Vec<_> = rows.into_iter().skip(offset).take(limit).collect();
    let mut j = json!({"total": total, "rows": rows});
    if offset + limit < total {
        j["next_offset"] = json!(offset + limit);
    }
    j
}

fn list_faces(ctx: &mut Ctx, a: ListArgs) -> Result<Out> {
    let (id, sel) = scope(ctx, &a)?;
    let m = &ctx.doc.scene.object(id)?.mesh;
    let filter = sel.map(|s| s.implied_faces(m));
    let rows: Vec<_> = m
        .faces
        .iter()
        .filter(|(fid, _)| filter.as_ref().is_none_or(|s| s.contains(fid)))
        .map(|(fid, f)| {
            let mut r = json!({
                "id": ids::face(*fid),
                "verts": ids::verts(f.verts.iter().copied()),
                "center": round3(m.face_center(f)),
                "normal": round3(m.face_normal(f)),
                "area": (m.face_area(f) * 10000.0).round() / 10000.0,
            });
            if f.material != 0 {
                r["material"] = json!(f.material);
            }
            if f.smooth {
                r["smooth"] = json!(true);
            }
            if let Some(p) = &f.part {
                r["part"] = json!(p);
            }
            r
        })
        .collect();
    Ok(Out::json(page(rows, a.offset, a.limit)))
}

fn list_verts(ctx: &mut Ctx, a: ListArgs) -> Result<Out> {
    let (id, sel) = scope(ctx, &a)?;
    let m = &ctx.doc.scene.object(id)?.mesh;
    let filter = sel.map(|s| s.touched_verts(m));
    let rows: Vec<_> = m
        .verts
        .iter()
        .filter(|(v, _)| filter.as_ref().is_none_or(|s| s.contains(v)))
        .map(|(v, p)| json!([ids::vert(*v), round3(p.pos)]))
        .collect();
    let mut j = page(rows, a.offset, a.limit);
    j["format"] = json!("[id, [x, y, z]]");
    Ok(Out::json(j))
}

fn list_edges(ctx: &mut Ctx, a: ListArgs) -> Result<Out> {
    let (id, sel) = scope(ctx, &a)?;
    let m = &ctx.doc.scene.object(id)?.mesh;
    let ef = m.edge_faces();
    let filter = sel.map(|s| {
        let mut e = s.implied_edges(m);
        for f in &s.faces {
            e.extend(m.faces[f].edges());
        }
        e
    });
    let rows: Vec<_> = m
        .edges()
        .into_iter()
        .filter(|e| filter.as_ref().is_none_or(|s| s.contains(e)))
        .map(|e| {
            let mut r = json!({
                "id": ids::edge(e),
                "length": ((m.pos(e.0) - m.pos(e.1)).length() * 10000.0).round() / 10000.0,
                "faces": ids::faces(ef[&e].iter().copied()),
            });
            if m.seams.contains(&e) {
                r["seam"] = json!(true);
            }
            if m.sharp.contains(&e) {
                r["sharp"] = json!(true);
            }
            r
        })
        .collect();
    Ok(Out::json(page(rows, a.offset, a.limit)))
}

#[derive(Deserialize, JsonSchema)]
struct SelectArgs {
    object: Option<String>,
    query: Query,
    /// replace (default), add, subtract, intersect.
    #[serde(default)]
    mode: Option<String>,
    /// Convert the result to one element kind: verts, edges, faces (default: as produced by the query).
    #[serde(default)]
    kind: Option<String>,
}

fn select(ctx: &mut Ctx, a: SelectArgs) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let o = ctx.doc.scene.object(id)?;
    let mut s = run(&o.mesh, &a.query, &o.selection)?;
    if let Some(k) = &a.kind {
        s = convert(&o.mesh, &s, k)?;
    }
    let mut cur = o.selection.clone();
    let new = match a.mode.as_deref().unwrap_or("replace") {
        "replace" | "set" => s,
        "add" => {
            cur.union(&s);
            cur
        }
        "subtract" | "remove" => {
            cur.subtract(&s);
            cur
        }
        "intersect" => Selection {
            verts: cur.verts.intersection(&s.verts).copied().collect(),
            edges: cur.edges.intersection(&s.edges).copied().collect(),
            faces: cur.faces.intersection(&s.faces).copied().collect(),
        },
        m => bail!("unknown mode '{m}'"),
    };
    let name = o.name.clone();
    ctx.doc.scene.object_mut(id)?.selection = new.clone();
    ctx.doc.scene.active = Some(id);
    ctx.doc.touch();
    Ok(Out::json(json!({"object": name, "selection": new.to_json()})))
}

#[derive(Deserialize, JsonSchema)]
struct MeasureArgs {
    object: Option<String>,
    elements: Option<Vec<String>>,
}

fn measure(ctx: &mut Ctx, a: MeasureArgs) -> Result<Out> {
    let id = resolve(&ctx.doc.scene, &a.object)?;
    let s = elements(&ctx.doc.scene, id, &a.elements)?;
    let m = &ctx.doc.scene.object(id)?.mesh;
    if s.verts.len() == 2 && s.faces.is_empty() && s.edges.is_empty() {
        let mut it = s.verts.iter();
        let (p, q) = (m.pos(*it.next().unwrap()), m.pos(*it.next().unwrap()));
        return Ok(Out::json(json!({"distance": (q - p).length(), "delta": round3(q - p)})));
    }
    if s.faces.len() == 2 && s.verts.is_empty() && s.edges.is_empty() {
        let mut it = s.faces.iter();
        let (f, g) = (&m.faces[it.next().unwrap()], &m.faces[it.next().unwrap()]);
        let ang = m.face_normal(f).dot(m.face_normal(g)).clamp(-1.0, 1.0).acos().to_degrees();
        return Ok(Out::json(json!({
            "angle_between_normals_deg": (ang * 100.0).round() / 100.0,
            "center_distance": m.face_center(f).distance(m.face_center(g)),
        })));
    }
    let verts = s.touched_verts(m);
    let pts: Vec<Vec3> = verts.iter().map(|v| m.pos(*v)).collect();
    let lo = pts.iter().fold(Vec3::splat(f32::MAX), |a, b| a.min(*b));
    let hi = pts.iter().fold(Vec3::splat(f32::MIN), |a, b| a.max(*b));
    let area: f32 = s.implied_faces(m).iter().map(|f| m.face_area(&m.faces[f])).sum();
    let edge_len: f32 = s.edges.iter().map(|e| m.pos(e.0).distance(m.pos(e.1))).sum();
    Ok(Out::json(json!({
        "verts": verts.len(),
        "bounds": {"min": round3(lo), "max": round3(hi), "size": round3(hi - lo)},
        "centroid": round3(pts.iter().sum::<Vec3>() / pts.len().max(1) as f32),
        "face_area": area,
        "edge_length": edge_len,
    })))
}
