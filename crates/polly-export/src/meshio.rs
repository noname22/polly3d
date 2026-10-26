//! Plain geometry files for scripted workflows: Wavefront OBJ and a simple
//! JSON format, read into new objects and written from existing ones.
//!
//! JSON: `{"vertices": [[x,y,z], ...], "faces": [[i, j, k, ...], ...]}` with
//! optional per-face `"uvs"` (one `[u, v]` per corner, glTF orientation:
//! v grows downward), `"parts"` (name or null) and `"smooth"` (bool). Exports
//! add `"vertex_ids"` and `"face_ids"`: the Polly ids of each row, so a script
//! can refer back to elements ("v12", "f3").
//!
//! OBJ: `v`, `vt`, `f` (any `v`, `v/vt`, `v/vt/vn`, `v//vn` form, negative
//! indices allowed), `g` (faces after it get that part) and `s` (smoothing).
//! OBJ texture v grows upward, so it is flipped on the way in and out.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result, bail};
use polly_core::glam::{Mat4, Vec2, Vec3};
use polly_core::mesh::{Corner, Mesh};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MeshData {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uvs: Option<Vec<Vec<[f32; 2]>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<Option<String>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth: Option<Vec<bool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertex_ids: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub face_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Obj,
    Json,
}

impl Format {
    pub fn from_path(path: &Path) -> Result<Format> {
        match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
            Some("obj") => Ok(Format::Obj),
            Some("json") => Ok(Format::Json),
            _ => bail!("{}: use a .obj or .json file", path.display()),
        }
    }
}

pub fn read_mesh_file(path: &Path) -> Result<MeshData> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    match Format::from_path(path)? {
        Format::Obj => parse_obj(&text),
        Format::Json => serde_json::from_str(&text).with_context(|| format!("{}: not a mesh JSON file", path.display())),
    }
}

/// Build a mesh, checking indices and per-face array lengths.
pub fn mesh_from_data(d: &MeshData) -> Result<Mesh> {
    let mut m = Mesh::new();
    let ids: Vec<_> = d.vertices.iter().map(|p| m.add_vert(Vec3::from_array(*p))).collect();
    let n = d.faces.len();
    for (name, len) in [("uvs", d.uvs.as_ref().map(Vec::len)), ("parts", d.parts.as_ref().map(Vec::len)), ("smooth", d.smooth.as_ref().map(Vec::len))] {
        if let Some(len) = len
            && len != n
        {
            bail!("'{name}' has {len} entries but there are {n} faces");
        }
    }
    for (i, f) in d.faces.iter().enumerate() {
        let verts = f
            .iter()
            .map(|k| ids.get(*k).copied().ok_or_else(|| anyhow::anyhow!("face {i}: vertex index {k} out of range")))
            .collect::<Result<Vec<_>>>()?;
        let corners = match d.uvs.as_ref().map(|u| &u[i]) {
            Some(uv) if uv.len() != f.len() => bail!("face {i}: {} uvs for {} corners", uv.len(), f.len()),
            Some(uv) => uv.iter().map(|t| Corner { uv: Vec2::from_array(*t), ..Default::default() }).collect(),
            None => vec![Corner::default(); f.len()],
        };
        let smooth = d.smooth.as_ref().is_some_and(|s| s[i]);
        let fid = m.add_face_with(verts, corners, 0, smooth).map_err(|e| anyhow::anyhow!("face {i}: {e}"))?;
        if let Some(p) = d.parts.as_ref().and_then(|p| p[i].clone()) {
            m.faces.get_mut(&fid).unwrap().part = Some(p);
        }
    }
    Ok(m)
}

/// Rows of a mesh, positions transformed by `xform` (object or world space).
pub fn data_from_mesh(m: &Mesh, xform: Mat4) -> MeshData {
    let index: HashMap<u32, usize> = m.verts.keys().enumerate().map(|(i, v)| (*v, i)).collect();
    let has_uvs = m.faces.values().flat_map(|f| &f.corners).any(|c| c.uv != Vec2::ZERO);
    let has_parts = m.faces.values().any(|f| f.part.is_some());
    let has_smooth = m.faces.values().any(|f| f.smooth);
    MeshData {
        vertices: m.verts.values().map(|v| xform.transform_point3(v.pos).to_array()).collect(),
        faces: m.faces.values().map(|f| f.verts.iter().map(|v| index[v]).collect()).collect(),
        uvs: has_uvs.then(|| m.faces.values().map(|f| f.corners.iter().map(|c| c.uv.to_array()).collect()).collect()),
        parts: has_parts.then(|| m.faces.values().map(|f| f.part.clone()).collect()),
        smooth: has_smooth.then(|| m.faces.values().map(|f| f.smooth).collect()),
        vertex_ids: Some(m.verts.keys().copied().collect()),
        face_ids: Some(m.faces.keys().copied().collect()),
    }
}

pub fn write_mesh_file(path: &Path, d: &MeshData, name: &str) -> Result<()> {
    let bytes = match Format::from_path(path)? {
        Format::Json => serde_json::to_vec(d)?,
        Format::Obj => to_obj(d, name).into_bytes(),
    };
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn parse_obj(text: &str) -> Result<MeshData> {
    let mut d = MeshData::default();
    let mut tex: Vec<[f32; 2]> = Vec::new();
    let mut uvs: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut any_uv = false;
    let mut parts = Vec::new();
    let mut smooth = Vec::new();
    let (mut part, mut s) = (None::<String>, false);
    for (ln, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        let mut it = line.split_whitespace();
        let Some(tag) = it.next() else { continue };
        let ctx = || format!("line {}: '{line}'", ln + 1);
        let nums = |it: std::str::SplitWhitespace| -> Result<Vec<f32>> {
            it.map(|x| x.parse::<f32>().with_context(ctx)).collect()
        };
        match tag {
            "v" => {
                let p = nums(it)?;
                if p.len() < 3 {
                    bail!("{}: a vertex needs x y z", ctx());
                }
                d.vertices.push([p[0], p[1], p[2]]);
            }
            "vt" => {
                let t = nums(it)?;
                tex.push([t.first().copied().unwrap_or(0.0), 1.0 - t.get(1).copied().unwrap_or(0.0)]);
            }
            "f" => {
                let mut face = Vec::new();
                let mut fuv = Vec::new();
                for c in it {
                    let mut idx = c.split('/');
                    let resolve = |s: &str, len: usize| -> Result<usize> {
                        let i: i64 = s.parse().with_context(ctx)?;
                        let k = if i < 0 { len as i64 + i } else { i - 1 };
                        if k < 0 || k as usize >= len {
                            bail!("{}: index {i} out of range", ctx());
                        }
                        Ok(k as usize)
                    };
                    face.push(resolve(idx.next().unwrap_or(""), d.vertices.len())?);
                    match idx.next().filter(|t| !t.is_empty()) {
                        Some(t) => {
                            fuv.push(tex[resolve(t, tex.len())?]);
                            any_uv = true;
                        }
                        None => fuv.push([0.0, 0.0]),
                    }
                }
                d.faces.push(face);
                uvs.push(fuv);
                parts.push(part.clone());
                smooth.push(s);
            }
            "g" => part = it.next().filter(|g| *g != "default").map(str::to_string),
            "s" => s = it.next().is_some_and(|v| v != "off" && v != "0"),
            _ => {} // vn, usemtl, mtllib, o, l, ... are ignored
        }
    }
    if any_uv {
        d.uvs = Some(uvs);
    }
    if parts.iter().any(Option::is_some) {
        d.parts = Some(parts);
    }
    if smooth.iter().any(|s| *s) {
        d.smooth = Some(smooth);
    }
    Ok(d)
}

fn to_obj(d: &MeshData, name: &str) -> String {
    let mut o = String::new();
    writeln!(o, "# polly3d\no {name}").unwrap();
    for v in &d.vertices {
        writeln!(o, "v {} {} {}", v[0], v[1], v[2]).unwrap();
    }
    // Shared texture coordinates, deduplicated.
    let mut vt: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut vt_list = Vec::new();
    let corner_vt: Option<Vec<Vec<usize>>> = d.uvs.as_ref().map(|uvs| {
        uvs.iter()
            .map(|f| {
                f.iter()
                    .map(|t| {
                        *vt.entry((t[0].to_bits(), t[1].to_bits())).or_insert_with(|| {
                            vt_list.push(*t);
                            vt_list.len()
                        })
                    })
                    .collect()
            })
            .collect()
    });
    for t in &vt_list {
        writeln!(o, "vt {} {}", t[0], 1.0 - t[1]).unwrap();
    }
    let (mut part, mut smooth) = (None, false);
    writeln!(o, "s off").unwrap();
    for (i, f) in d.faces.iter().enumerate() {
        let p = d.parts.as_ref().and_then(|p| p[i].as_deref());
        if p != part {
            writeln!(o, "g {}", p.unwrap_or("default")).unwrap();
            part = p;
        }
        let s = d.smooth.as_ref().is_some_and(|s| s[i]);
        if s != smooth {
            writeln!(o, "s {}", if s { "1" } else { "off" }).unwrap();
            smooth = s;
        }
        o.push('f');
        for (k, v) in f.iter().enumerate() {
            match &corner_vt {
                Some(t) => write!(o, " {}/{}", v + 1, t[i][k]).unwrap(),
                None => write!(o, " {}", v + 1).unwrap(),
            }
        }
        o.push('\n');
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_round_trip_keeps_uvs_parts_and_smoothing() {
        let text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\ng sleeve\ns 1\nf 1/1 2/2 3/3 4/4\ng default\ns off\nf -4 -2 -1\n";
        let d = parse_obj(text).unwrap();
        assert_eq!(d.faces, vec![vec![0, 1, 2, 3], vec![0, 2, 3]]);
        // OBJ v grows upward, Polly's downward.
        assert_eq!(d.uvs.as_ref().unwrap()[0][0], [0.0, 1.0]);
        assert_eq!(d.parts.as_ref().unwrap(), &vec![Some("sleeve".to_string()), None]);
        assert_eq!(d.smooth.as_ref().unwrap(), &vec![true, false]);
        let m = mesh_from_data(&d).unwrap();
        let back = parse_obj(&to_obj(&data_from_mesh(&m, Mat4::IDENTITY), "x")).unwrap();
        assert_eq!(back.faces, d.faces);
        assert_eq!(back.uvs, d.uvs);
        assert_eq!(back.parts, d.parts);
        assert_eq!(back.smooth, d.smooth);
    }

    #[test]
    fn json_lengths_are_checked() {
        let d: MeshData = serde_json::from_str(r#"{"vertices": [[0,0,0],[1,0,0],[0,1,0]], "faces": [[0,1,2]], "parts": []}"#).unwrap();
        assert!(mesh_from_data(&d).unwrap_err().to_string().contains("parts"));
        let d: MeshData = serde_json::from_str(r#"{"vertices": [[0,0,0]], "faces": [[0,1,2]]}"#).unwrap();
        assert!(mesh_from_data(&d).is_err());
    }
}
