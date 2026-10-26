//! Swappable gear: slots (e.g. "hair", "head") hold variants, each a set of
//! objects. One variant per slot is active; a variant can hide parts of other
//! objects (a helmet hiding the top of the hair, a blazer the skin under it)
//! to avoid poke-through. An object can be in several variants of one slot
//! ("blouse" and "suit" = blouse + blazer), but not in two slots.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{Result, invalid};
use crate::mesh::{Face, FaceId};
use crate::scene::{ObjectId, Scene};

/// Hide faces of `object` while the variant is active: the faces labelled
/// `part`, or an explicit face set named `part` (e.g. found by auto_hide).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HideRule {
    pub object: ObjectId,
    pub part: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<BTreeSet<FaceId>>,
}

impl HideRule {
    pub fn hides(&self, id: FaceId, f: &Face) -> bool {
        match &self.faces {
            Some(set) => set.contains(&id),
            None => f.part.as_deref() == Some(self.part.as_str()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variant {
    pub name: String,
    /// May be empty (e.g. "none" for no helmet).
    pub objects: Vec<ObjectId>,
    #[serde(default)]
    pub hides: Vec<HideRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Slot {
    pub variants: Vec<Variant>,
    /// Shown when the model is loaded (exported as visible).
    pub default: String,
    /// Shown in Polly right now.
    pub active: String,
}

impl Slot {
    pub fn variant(&self, name: &str) -> Result<&Variant> {
        self.variants.iter().find(|v| v.name == name).ok_or_else(|| {
            crate::Error::Invalid(format!(
                "no variant '{name}' (variants: {})",
                self.variants.iter().map(|v| v.name.as_str()).collect::<Vec<_>>().join(", ")
            ))
        })
    }
}

impl Scene {
    /// The slot an object belongs to and its variants there, if any.
    pub fn variants_of(&self, id: ObjectId) -> Option<(&str, Vec<&str>)> {
        self.slots.iter().find_map(|(s, slot)| {
            let vs: Vec<&str> = slot.variants.iter().filter(|v| v.objects.contains(&id)).map(|v| v.name.as_str()).collect();
            (!vs.is_empty()).then_some((s.as_str(), vs))
        })
    }

    /// Add or replace a variant. The slot is created if needed; its first
    /// variant becomes the default and the active one.
    pub fn set_variant_def(&mut self, slot: &str, v: Variant) -> Result<()> {
        if slot.trim().is_empty() || v.name.trim().is_empty() {
            return invalid("slot and variant names must not be empty");
        }
        for o in &v.objects {
            let ob = self.object(*o)?;
            match self.variants_of(*o) {
                Some((s, vs)) if s != slot => {
                    return invalid(format!(
                        "'{}' already belongs to slot '{s}' (variant {}); an object can be in several variants of one slot, not in two slots",
                        ob.name,
                        vs.join(", ")
                    ));
                }
                _ => {}
            }
        }
        let mut names = BTreeSet::new();
        for h in &v.hides {
            let ob = self.object(h.object)?;
            if !names.insert((h.object, h.part.as_str())) {
                return invalid(format!("two hide rules for '{}' are both named '{}'", ob.name, h.part));
            }
            if let Some(faces) = &h.faces {
                if !faces.iter().any(|f| ob.mesh.faces.contains_key(f)) {
                    return invalid(format!("hide rule '{}': none of its faces exist on '{}'", h.part, ob.name));
                }
                continue;
            }
            if !ob.mesh.faces.values().any(|f| f.part.as_deref() == Some(h.part.as_str())) {
                let parts: BTreeSet<&str> = ob.mesh.faces.values().filter_map(|f| f.part.as_deref()).collect();
                return invalid(format!(
                    "'{}' has no part '{}' to hide (parts: {}); label faces with set_part first",
                    ob.name,
                    h.part,
                    if parts.is_empty() { "none".into() } else { parts.into_iter().collect::<Vec<_>>().join(", ") }
                ));
            }
        }
        let s = self.slots.entry(slot.to_string()).or_insert_with(|| Slot { variants: vec![], default: v.name.clone(), active: v.name.clone() });
        match s.variants.iter_mut().find(|x| x.name == v.name) {
            Some(x) => *x = v,
            None => s.variants.push(v),
        }
        Ok(())
    }

    /// Forget an object everywhere in slots (when it is deleted).
    pub(crate) fn drop_from_slots(&mut self, id: ObjectId) {
        for slot in self.slots.values_mut() {
            for v in &mut slot.variants {
                v.objects.retain(|o| *o != id);
                v.hides.retain(|h| h.object != id);
            }
        }
    }
}

/// Display copy with the chosen variants: objects of other variants hidden and
/// parts hidden by the chosen variants' hide rules removed. `choice`
/// overrides each slot's active variant.
pub fn variant_view(scene: &Scene, choice: &BTreeMap<String, String>) -> Scene {
    if scene.slots.is_empty() {
        return scene.clone();
    }
    let mut out = scene.clone();
    let mut hide: BTreeMap<ObjectId, Vec<&HideRule>> = BTreeMap::new();
    for (name, slot) in &scene.slots {
        let pick = choice.get(name).unwrap_or(&slot.active);
        let chosen = slot.variants.iter().find(|v| &v.name == pick);
        let shown: BTreeSet<ObjectId> = chosen.map(|v| v.objects.iter().copied().collect()).unwrap_or_default();
        for h in chosen.into_iter().flat_map(|v| &v.hides) {
            hide.entry(h.object).or_default().push(h);
        }
        for o in slot.variants.iter().flat_map(|v| &v.objects).filter(|o| !shown.contains(o)) {
            if let Some(ob) = out.objects.get_mut(o) {
                ob.visible = false;
            }
        }
    }
    for (id, rules) in hide {
        if let Some(ob) = out.objects.get_mut(&id) {
            let m = std::sync::Arc::make_mut(&mut ob.mesh);
            m.faces.retain(|fid, f| !rules.iter().any(|r| r.hides(*fid, f)));
        }
    }
    out
}

/// For each object with hide rules, its faces grouped by the set of rule
/// names that hide them (faces no rule hides are left out). Overlapping
/// rules (a blouse and a blazer both covering the chest) give separate groups.
pub fn hide_groups(scene: &Scene) -> BTreeMap<ObjectId, BTreeMap<BTreeSet<String>, BTreeSet<FaceId>>> {
    let mut rules: BTreeMap<ObjectId, BTreeMap<&str, &HideRule>> = BTreeMap::new();
    for h in scene.slots.values().flat_map(|s| &s.variants).flat_map(|v| &v.hides) {
        rules.entry(h.object).or_default().insert(&h.part, h);
    }
    let mut out = BTreeMap::new();
    for (id, rs) in rules {
        let Some(o) = scene.objects.get(&id) else { continue };
        let mut groups: BTreeMap<BTreeSet<String>, BTreeSet<FaceId>> = BTreeMap::new();
        for (fid, f) in &o.mesh.faces {
            let key: BTreeSet<String> = rs.iter().filter(|(_, r)| r.hides(*fid, f)).map(|(n, _)| n.to_string()).collect();
            if !key.is_empty() {
                groups.entry(key).or_default().insert(*fid);
            }
        }
        out.insert(id, groups);
    }
    out
}

/// What should be shown: variants chosen, then posed (see [`crate::anim::pose_scene`]).
pub fn display(scene: &Scene, plays: &[crate::anim::ClipPlay], choice: &BTreeMap<String, String>) -> Scene {
    crate::anim::pose_scene(&variant_view(scene, choice), plays)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::cube;
    use glam::Vec3;

    fn knight() -> (Scene, ObjectId, ObjectId, ObjectId, ObjectId) {
        let mut s = Scene::default();
        let mut hair = cube(Vec3::ONE);
        // label the top face as "hair_top"
        for f in hair.faces.values_mut() {
            f.part = Some("hair".into());
        }
        let top = *hair.faces.iter().find(|(_, f)| f.verts.len() == 4).map(|(i, _)| i).unwrap();
        hair.faces.get_mut(&top).unwrap().part = Some("hair_top".into());
        let hair_a = s.add_object("HairA", hair);
        let hair_b = s.add_object("HairB", cube(Vec3::ONE));
        let helmet = s.add_object("Helmet", cube(Vec3::ONE));
        let body = s.add_object("Body", cube(Vec3::ONE));
        (s, hair_a, hair_b, helmet, body)
    }

    #[test]
    fn variants_choose_objects_and_hide_parts() {
        let (mut s, a, b, helmet, _) = knight();
        s.set_variant_def("hair", Variant { name: "short".into(), objects: vec![a], hides: vec![] }).unwrap();
        s.set_variant_def("hair", Variant { name: "long".into(), objects: vec![b], hides: vec![] }).unwrap();
        s.set_variant_def("head", Variant { name: "none".into(), objects: vec![], hides: vec![] }).unwrap();
        s.set_variant_def("head", Variant { name: "helmet".into(), objects: vec![helmet], hides: vec![HideRule { object: a, part: "hair_top".into(), faces: None }] })
            .unwrap();
        // defaults: first variants
        assert_eq!(s.slots["hair"].default, "short");
        let v = variant_view(&s, &BTreeMap::new());
        assert!(v.objects[&a].visible && !v.objects[&b].visible && !v.objects[&helmet].visible);
        assert_eq!(v.objects[&a].mesh.faces.len(), 6);
        // choosing the helmet shows it and hides the top of the short hair
        let v = variant_view(&s, &BTreeMap::from([("head".to_string(), "helmet".to_string())]));
        assert!(v.objects[&helmet].visible);
        assert_eq!(v.objects[&a].mesh.faces.len(), 5);
        // an object can't be in two slots; hide rules need a real part
        assert!(s.set_variant_def("head", Variant { name: "cap".into(), objects: vec![a], hides: vec![] }).is_err());
        assert!(s.set_variant_def("head", Variant { name: "cap".into(), objects: vec![], hides: vec![HideRule { object: b, part: "x".into(), faces: None }] }).is_err());
        // deleting an object removes it from variants and hide rules
        s.remove_object(a).unwrap();
        assert!(s.slots["hair"].variants[0].objects.is_empty());
        assert!(s.slots["head"].variants[1].hides.is_empty());
    }
}
