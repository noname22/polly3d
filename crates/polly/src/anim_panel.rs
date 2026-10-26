//! The viewer's animation panel: transport, timeline with loop handles, clip
//! blend weights and morph sliders.
//!
//! Everything here is a preview that lives only in the viewer: playback time,
//! play/pause, clip weights, morph slider values and loop regions moved with
//! the handles (applied to the display copy by [`AnimState::apply_loops`]).
//! The scene's clips are never changed.

use std::collections::BTreeMap;

use egui::{Color32, Rect, Sense, Stroke, pos2, vec2};
use polly_core::anim::{ClipId, ClipPlay};
use polly_core::scene::{ObjectId, Scene};

#[derive(Clone, Copy)]
struct ClipMix {
    enabled: bool,
    weight: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum Grab {
    Playhead,
    LoopStart,
    LoopEnd,
}

/// Per-scene playback state.
#[derive(Default)]
pub struct AnimState {
    /// Playback time in seconds (each clip maps it through its own loop region).
    time: f32,
    playing: bool,
    mix: BTreeMap<ClipId, ClipMix>,
    /// Clip shown on the timeline.
    focus: Option<ClipId>,
    /// Morph slider values overriding the scene's static weights.
    morphs: BTreeMap<(ObjectId, String), f32>,
    grab: Option<Grab>,
    /// Loop handles while being dragged (seconds).
    drag_loop: Option<(f32, f32)>,
    /// Loop regions set with the handles (seconds), overriding the clips'.
    loops: BTreeMap<ClipId, (f32, f32)>,
}

impl AnimState {
    /// Bring the state in line with the scene's current clips.
    fn sync(&mut self, scene: &Scene) {
        self.mix.retain(|id, _| scene.clips.contains_key(id));
        let first = self.mix.is_empty();
        for id in scene.clips.keys() {
            self.mix.entry(*id).or_insert(ClipMix { enabled: first && Some(*id) == scene.clips.keys().next().copied(), weight: 1.0 });
        }
        if self.focus.is_none_or(|f| !scene.clips.contains_key(&f)) {
            self.focus = self.mix.iter().find(|(_, m)| m.enabled).map(|(id, _)| *id).or(scene.clips.keys().next().copied());
        }
        self.morphs.retain(|(o, m), _| scene.objects.get(o).is_some_and(|ob| ob.mesh.morphs.iter().any(|x| &x.name == m)));
    }

    /// Lay the user's loop regions over the clips of a display copy.
    pub fn apply_loops(&mut self, scene: &mut Scene) {
        self.loops.retain(|id, _| scene.clips.contains_key(id));
        for (id, (start, end)) in &self.loops {
            let c = scene.clips.get_mut(id).unwrap();
            c.loop_start = start.clamp(0.0, c.length);
            c.loop_end = Some(end.clamp(c.loop_start, c.length));
        }
    }

    pub fn has_content(scene: &Scene) -> bool {
        !scene.clips.is_empty() || scene.objects.values().any(|o| !o.mesh.morphs.is_empty())
    }

    pub fn playing(&self) -> bool {
        self.playing
    }

    /// Advance playback by `dt` seconds; stops at the end when nothing loops.
    pub fn advance(&mut self, scene: &Scene, dt: f32) {
        self.sync(scene);
        if !self.playing {
            return;
        }
        self.time += dt;
        let active: Vec<_> = self.mix.iter().filter(|(_, m)| m.enabled).filter_map(|(id, _)| scene.clips.get(id)).collect();
        if active.iter().all(|c| !c.looping) {
            let end = active.iter().map(|c| c.length).fold(0.0, f32::max);
            if self.time >= end {
                self.time = end;
                self.playing = false;
            }
        }
    }

    pub fn plays(&self, scene: &Scene) -> Vec<ClipPlay> {
        self.mix
            .iter()
            .filter(|(_, m)| m.enabled && m.weight > 0.0)
            .filter_map(|(id, m)| Some(ClipPlay { clip: *id, time: scene.clips.get(id)?.playback_time(self.time), weight: m.weight }))
            .collect()
    }

    /// The scene as it should be shown right now.
    pub fn display(&self, scene: &Scene) -> Scene {
        if self.morphs.is_empty() {
            return polly_core::anim::pose_scene(scene, &self.plays(scene));
        }
        let mut s = scene.clone();
        for ((o, m), w) in &self.morphs {
            if let Some(ob) = s.objects.get_mut(o) {
                ob.morph_weights.insert(m.clone(), *w);
            }
        }
        polly_core::anim::pose_scene(&s, &self.plays(&s))
    }

    pub fn toggle_play(&mut self, scene: &Scene) {
        self.sync(scene);
        if !self.playing && !self.mix.values().any(|m| m.enabled) {
            // Nothing selected: play the timeline's clip.
            if let Some(f) = self.focus.and_then(|f| self.mix.get_mut(&f)) {
                f.enabled = true;
            }
        }
        self.playing = !self.playing;
    }
}

/// Draw the panel.
pub fn panel(ui: &mut egui::Ui, st: &mut AnimState, scene: &Scene) {
    st.sync(scene);
    if !scene.clips.is_empty() {
        transport(ui, st, scene);
        if let Some(id) = st.focus {
            timeline(ui, st, scene, id);
            if st.loops.contains_key(&id) && ui.small_button("Reset loop").on_hover_text("back to the clip's own loop region").clicked() {
                st.loops.remove(&id);
            }
        }
        ui.add_space(4.0);
        clips(ui, st, scene);
    }
    morphs(ui, st, scene);
}

fn transport(ui: &mut egui::Ui, st: &mut AnimState, scene: &Scene) {
    ui.horizontal(|ui| {
        let play = if st.playing { "Pause" } else { "Play" };
        if ui.button(play).on_hover_text("Space").clicked() {
            st.toggle_play(scene);
        }
        if ui.button("Stop").on_hover_text("stop and rewind").clicked() {
            st.playing = false;
            st.time = 0.0;
        }
        if let Some(c) = st.focus.and_then(|f| scene.clips.get(&f)) {
            let t = c.playback_time(st.time);
            ui.monospace(format!("frame {:6.1} / {}", t * c.fps, (c.length * c.fps).round()));
            ui.weak(format!("{} @ {} fps{}", c.name, c.fps, if c.looping { ", looping" } else { "" }));
        }
    });
}

/// Timeline of the focused clip: ticks, loop region, playhead, loop handles.
fn timeline(ui: &mut egui::Ui, st: &mut AnimState, scene: &Scene, id: ClipId) {
    let clip = &scene.clips[&id];
    let frames = (clip.length * clip.fps).max(1.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 40.0), Sense::click_and_drag());
    let track = Rect::from_min_max(pos2(rect.left() + 8.0, rect.top() + 12.0), pos2(rect.right() - 8.0, rect.bottom() - 4.0));
    let x_of = |sec: f32| track.left() + (sec * clip.fps / frames).clamp(0.0, 1.0) * track.width();
    let sec_of = |x: f32| ((x - track.left()) / track.width()).clamp(0.0, 1.0) * frames / clip.fps;
    let snap = |sec: f32| (sec * clip.fps).round() / clip.fps;
    let (mut ls, mut le) = st.drag_loop.unwrap_or_else(|| clip.loop_range());

    // Interaction.
    if resp.drag_started() || resp.clicked() {
        // Where the button went down: by the time a drag is recognised the
        // pointer has already moved away from a small handle.
        let p = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos()).unwrap_or_default();
        let near = |x: f32| (p.x - x).abs() < 7.0 && p.y < track.top() + 6.0;
        st.grab = Some(if near(x_of(ls)) {
            Grab::LoopStart
        } else if near(x_of(le)) {
            Grab::LoopEnd
        } else {
            Grab::Playhead
        });
    }
    if let (Some(g), Some(p)) = (st.grab, resp.interact_pointer_pos()) {
        if resp.dragged() || resp.clicked() || resp.drag_started() {
            match g {
                Grab::Playhead => {
                    st.playing = false;
                    st.time = sec_of(p.x);
                }
                Grab::LoopStart => ls = snap(sec_of(p.x)).min(le - 1.0 / clip.fps).max(0.0),
                Grab::LoopEnd => le = snap(sec_of(p.x)).max(ls + 1.0 / clip.fps).min(clip.length),
            }
            if g != Grab::Playhead {
                st.drag_loop = Some((ls, le));
            }
        }
    }
    if resp.drag_stopped() || resp.clicked() {
        if matches!(st.grab, Some(Grab::LoopStart | Grab::LoopEnd)) {
            st.loops.insert(id, (ls, le));
        }
        st.grab = None;
        st.drag_loop = None;
    }

    // Drawing.
    let p = ui.painter_at(rect);
    let vis = ui.visuals();
    p.rect_filled(track, 3.0, vis.extreme_bg_color);
    let loop_rect = Rect::from_min_max(pos2(x_of(ls), track.top()), pos2(x_of(le), track.bottom()));
    p.rect_filled(loop_rect, 0.0, Color32::from_rgba_unmultiplied(80, 140, 230, if clip.looping { 70 } else { 25 }));
    let px_per_frame = track.width() / frames;
    let label_every = [1, 2, 5, 10, 20, 50, 100, 200].into_iter().find(|n| *n as f32 * px_per_frame >= 40.0).unwrap_or(500);
    for f in 0..=frames as i32 {
        let x = track.left() + f as f32 * px_per_frame;
        let major = f % label_every == 0;
        if major || px_per_frame >= 5.0 {
            let h = if major { 10.0 } else { 5.0 };
            p.line_segment([pos2(x, track.bottom() - h), pos2(x, track.bottom())], Stroke::new(1.0, vis.weak_text_color()));
        }
        if major {
            // Right-align labels near the end so they stay inside the track.
            let (at, align) = if x > track.right() - 20.0 {
                (pos2(x - 2.0, track.top() + 1.0), egui::Align2::RIGHT_TOP)
            } else {
                (pos2(x + 2.0, track.top() + 1.0), egui::Align2::LEFT_TOP)
            };
            p.text(at, align, f.to_string(), egui::FontId::monospace(10.0), vis.weak_text_color());
        }
    }
    // Loop handles: downward triangles above the track.
    for x in [x_of(ls), x_of(le)] {
        let tri = vec![pos2(x - 6.0, rect.top() + 1.0), pos2(x + 6.0, rect.top() + 1.0), pos2(x, track.top() + 1.0)];
        p.add(egui::Shape::convex_polygon(tri, Color32::from_rgb(90, 150, 240), Stroke::new(1.0, Color32::BLACK)));
        p.line_segment([pos2(x, track.top()), pos2(x, track.bottom())], Stroke::new(1.0, Color32::from_rgb(90, 150, 240)));
    }
    let head = x_of(clip.playback_time(st.time));
    p.line_segment([pos2(head, rect.top() + 4.0), pos2(head, track.bottom())], Stroke::new(2.0, Color32::from_rgb(250, 200, 60)));
    resp.on_hover_text("click/drag: scrub · drag the blue triangles: loop region (only in this window)");
}

fn clips(ui: &mut egui::Ui, st: &mut AnimState, scene: &Scene) {
    egui::Grid::new("clips").num_columns(3).spacing([12.0, 2.0]).show(ui, |ui| {
        for (id, c) in &scene.clips {
            let m = st.mix.get_mut(id).unwrap();
            ui.checkbox(&mut m.enabled, "").on_hover_text("play this clip");
            if ui.selectable_label(st.focus == Some(*id), &c.name).on_hover_text("show on the timeline").clicked() {
                st.focus = Some(*id);
            }
            ui.add_enabled(m.enabled, egui::Slider::new(&mut m.weight, 0.0..=1.0).text("weight"));
            ui.end_row();
        }
    });
}

fn morphs(ui: &mut egui::Ui, st: &mut AnimState, scene: &Scene) {
    let with: Vec<_> = scene.objects.iter().filter(|(_, o)| !o.mesh.morphs.is_empty()).collect();
    if with.is_empty() {
        return;
    }
    ui.separator();
    ui.horizontal(|ui| {
        ui.strong("Morphs");
        if !st.morphs.is_empty() && ui.small_button("reset").on_hover_text("back to the scene's weights").clicked() {
            st.morphs.clear();
        }
    });
    egui::Grid::new("morphs").num_columns(2).spacing([12.0, 2.0]).show(ui, |ui| {
        for (oid, o) in with {
            for m in &o.mesh.morphs {
                let key = (*oid, m.name.clone());
                let base = o.morph_weights.get(&m.name).copied().unwrap_or(0.0);
                let mut w = st.morphs.get(&key).copied().unwrap_or(base);
                ui.label(format!("{} / {}", o.name, m.name));
                if ui.add(egui::Slider::new(&mut w, 0.0..=1.0).clamping(egui::SliderClamping::Never)).changed() {
                    st.morphs.insert(key, w);
                }
                ui.end_row();
            }
        }
    });
}
