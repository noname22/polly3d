//! Agent-facing element ids: `v12` (vertex), `f3` (face), `e3-9` (edge), `o1` (object).

use std::collections::BTreeSet;

use crate::error::{Error, Result};
use crate::mesh::{Edge, FaceId, Mesh, VertId};

pub fn vert(v: VertId) -> String {
    format!("v{v}")
}
pub fn face(f: FaceId) -> String {
    format!("f{f}")
}
pub fn edge(e: Edge) -> String {
    format!("e{}-{}", e.0, e.1)
}

/// Compress ids into ranges, e.g. `f400..f507, f512`. The output is valid
/// input for element arguments. Ids are expected sorted (BTreeSet order).
pub fn compress(prefix: &str, ids: impl IntoIterator<Item = u32>) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut run: Option<(u32, u32)> = None;
    let flush = |r: (u32, u32), out: &mut Vec<String>| {
        out.push(if r.0 == r.1 { format!("{prefix}{}", r.0) } else { format!("{prefix}{}..{prefix}{}", r.0, r.1) })
    };
    for id in ids {
        run = match run {
            Some((a, b)) if id == b + 1 => Some((a, id)),
            Some(r) => {
                flush(r, &mut out);
                Some((id, id))
            }
            None => Some((id, id)),
        };
    }
    if let Some(r) = run {
        flush(r, &mut out);
    }
    out.join(", ")
}

/// Edges don't form id ranges; list them, capped.
pub fn compress_edges(es: impl IntoIterator<Item = Edge>) -> String {
    let all: Vec<String> = es.into_iter().map(edge).collect();
    crate::mesh::short_list(&all)
}

pub fn verts(vs: impl IntoIterator<Item = VertId>) -> Vec<String> {
    vs.into_iter().map(vert).collect()
}
pub fn faces(fs: impl IntoIterator<Item = FaceId>) -> Vec<String> {
    fs.into_iter().map(face).collect()
}
pub fn edges(es: impl IntoIterator<Item = Edge>) -> Vec<String> {
    es.into_iter().map(edge).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    Vert(VertId),
    Face(FaceId),
    Edge(Edge),
}

pub fn parse(s: &str) -> Result<Element> {
    let s = s.trim();
    let bad = || Error::BadId(s.to_string());
    let (kind, rest) = s.split_at(s.find(|c: char| c.is_ascii_digit()).ok_or_else(bad)?);
    match kind {
        "v" => Ok(Element::Vert(rest.parse().map_err(|_| bad())?)),
        "f" => Ok(Element::Face(rest.parse().map_err(|_| bad())?)),
        "e" => {
            let (a, b) = rest.split_once('-').ok_or_else(bad)?;
            Ok(Element::Edge(Edge::new(a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?)))
        }
        _ => Err(bad()),
    }
}

/// A set of mesh elements, as given by the agent.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Selection {
    pub verts: BTreeSet<VertId>,
    pub edges: BTreeSet<Edge>,
    pub faces: BTreeSet<crate::mesh::FaceId>,
}

impl Selection {
    pub fn is_empty(&self) -> bool {
        self.verts.is_empty() && self.edges.is_empty() && self.faces.is_empty()
    }

    pub fn all(mesh: &Mesh) -> Self {
        Selection { verts: mesh.verts.keys().copied().collect(), edges: mesh.edges(), faces: mesh.faces.keys().copied().collect() }
    }

    /// Parse ids. The special tokens `all` and `selection` expand to the whole
    /// mesh and to `current` respectively.
    /// Ids may also be ranges (`f400..f507`, covering the ids that exist) and
    /// comma-separated lists in one string (`"f1, f4..f9, v3"`).
    pub fn parse(ids: &[String], mesh: &Mesh, current: &Selection) -> Result<Self> {
        let mut s = Selection::default();
        for id in ids.iter().flat_map(|s| s.split(',')).map(str::trim).filter(|s| !s.is_empty()) {
            if let Some((a, b)) = id.split_once("..") {
                s.union(&parse_range(a, b, mesh)?);
                continue;
            }
            match id {
                "all" => s.union(&Selection::all(mesh)),
                "selection" | "selected" => s.union(current),
                other => match parse(other)? {
                    Element::Vert(v) => {
                        mesh.try_pos(v)?;
                        s.verts.insert(v);
                    }
                    Element::Face(f) => {
                        mesh.face(f)?;
                        s.faces.insert(f);
                    }
                    Element::Edge(e) => {
                        mesh.try_pos(e.0)?;
                        mesh.try_pos(e.1)?;
                        if !mesh.faces.values().any(|f| f.edge_index(e).is_some()) {
                            return Err(Error::NotFound { kind: "edge", id: edge(e) });
                        }
                        s.edges.insert(e);
                    }
                },
            }
        }
        Ok(s)
    }

    /// Compact JSON for tool results: id ranges as strings, plus counts.
    pub fn summary(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        if !self.faces.is_empty() {
            m.insert("faces".into(), compress("f", self.faces.iter().copied()).into());
        }
        if !self.verts.is_empty() {
            m.insert("verts".into(), compress("v", self.verts.iter().copied()).into());
        }
        if !self.edges.is_empty() {
            m.insert("edges".into(), compress_edges(self.edges.iter().copied()).into());
        }
        m.insert(
            "counts".into(),
            serde_json::json!({"faces": self.faces.len(), "verts": self.verts.len(), "edges": self.edges.len()}),
        );
        m.into()
    }

    pub fn union(&mut self, o: &Selection) {
        self.verts.extend(&o.verts);
        self.edges.extend(&o.edges);
        self.faces.extend(&o.faces);
    }

    pub fn subtract(&mut self, o: &Selection) {
        self.verts.retain(|v| !o.verts.contains(v));
        self.edges.retain(|v| !o.edges.contains(v));
        self.faces.retain(|v| !o.faces.contains(v));
    }

    /// All vertices touched by any selected element.
    pub fn touched_verts(&self, mesh: &Mesh) -> BTreeSet<VertId> {
        let mut out = self.verts.clone();
        for e in &self.edges {
            out.insert(e.0);
            out.insert(e.1);
        }
        for f in &self.faces {
            if let Some(face) = mesh.faces.get(f) {
                out.extend(face.verts.iter().copied());
            }
        }
        out.retain(|v| mesh.verts.contains_key(v));
        out
    }

    /// Faces explicitly selected, plus faces whose vertices are all selected.
    pub fn implied_faces(&self, mesh: &Mesh) -> BTreeSet<FaceId> {
        let mut out = self.faces.clone();
        if !self.verts.is_empty() {
            for (&id, f) in &mesh.faces {
                if f.verts.iter().all(|v| self.verts.contains(v)) {
                    out.insert(id);
                }
            }
        }
        out
    }

    /// Edges explicitly selected, plus edges whose both endpoints are selected vertices.
    pub fn implied_edges(&self, mesh: &Mesh) -> BTreeSet<Edge> {
        let mut out = self.edges.clone();
        if !self.verts.is_empty() {
            for e in mesh.edges() {
                if self.verts.contains(&e.0) && self.verts.contains(&e.1) {
                    out.insert(e);
                }
            }
        }
        out
    }

    /// Drop references to elements that no longer exist.
    pub fn prune(&mut self, mesh: &Mesh) {
        self.verts.retain(|v| mesh.verts.contains_key(v));
        self.faces.retain(|f| mesh.faces.contains_key(f));
        let edges = mesh.edges();
        self.edges.retain(|e| edges.contains(e));
    }

    /// Compact JSON: id ranges per kind (see [`Selection::summary`]).
    pub fn to_json(&self) -> serde_json::Value {
        self.summary()
    }
}

/// `f400..f507` / `v1..v9`: the existing elements with ids in the range.
fn parse_range(a: &str, b: &str, mesh: &Mesh) -> Result<Selection> {
    let bad = || Error::BadId(format!("{a}..{b}"));
    let (ka, kb) = (parse(a).map_err(|_| bad())?, parse(b.trim()).map_err(|_| bad())?);
    let mut s = Selection::default();
    match (ka, kb) {
        (Element::Face(x), Element::Face(y)) => s.faces = mesh.faces.range(x.min(y)..=x.max(y)).map(|(i, _)| *i).collect(),
        (Element::Vert(x), Element::Vert(y)) => s.verts = mesh.verts.range(x.min(y)..=x.max(y)).map(|(i, _)| *i).collect(),
        _ => return Err(Error::Invalid(format!("'{a}..{b}': ranges work for faces (f1..f9) or vertices (v1..v9)"))),
    }
    if s.is_empty() {
        return Err(Error::NotFound { kind: "element", id: format!("{a}..{b} (no ids in that range)") });
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranges_roundtrip() {
        let mut m = crate::primitives::plane(glam::Vec2::ONE, 4, 4); // f0..f15
        m.remove_face(5);
        let s = Selection::parse(&["f2..f8, f12".into(), "v0".into()], &m, &Selection::default()).unwrap();
        assert_eq!(s.faces, [2, 3, 4, 6, 7, 8, 12].into());
        assert_eq!(compress("f", s.faces.iter().copied()), "f2..f4, f6..f8, f12");
        // compressed output parses back to the same set
        let again = Selection::parse(&[compress("f", s.faces.iter().copied())], &m, &Selection::default()).unwrap();
        assert_eq!(again.faces, s.faces);
        assert!(Selection::parse(&["f100..f200".into()], &m, &Selection::default()).is_err());
        assert!(Selection::parse(&["f1..v3".into()], &m, &Selection::default()).is_err());
    }

    #[test]
    fn parse_ids() {
        assert_eq!(parse("v12").unwrap(), Element::Vert(12));
        assert_eq!(parse("f3").unwrap(), Element::Face(3));
        assert_eq!(parse("e9-3").unwrap(), Element::Edge(Edge(3, 9)));
        assert!(parse("x1").is_err());
        assert!(parse("v").is_err());
    }
}
