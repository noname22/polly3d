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
    pub fn parse(ids: &[String], mesh: &Mesh, current: &Selection) -> Result<Self> {
        let mut s = Selection::default();
        for id in ids {
            match id.trim() {
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

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "verts": verts(self.verts.iter().copied()),
            "edges": edges(self.edges.iter().copied()),
            "faces": faces(self.faces.iter().copied()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_ids() {
        assert_eq!(parse("v12").unwrap(), Element::Vert(12));
        assert_eq!(parse("f3").unwrap(), Element::Face(3));
        assert_eq!(parse("e9-3").unwrap(), Element::Edge(Edge(3, 9)));
        assert!(parse("x1").is_err());
        assert!(parse("v").is_err());
    }
}
