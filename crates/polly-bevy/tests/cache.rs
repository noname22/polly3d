//! Scene -> entity caching (own test binary: the service is process-wide).

use polly_bevy::BevyRenderer;
use polly_core::primitives::Primitive;
use polly_core::scene::Scene;
use polly_render::{Overlays, RenderOptions, View};

#[test]
fn unchanged_objects_reuse_their_meshes() {
    let Ok(r) = BevyRenderer::global() else { return };
    let mut s = Scene::default();
    let a = s.add_object("A", Primitive::Cube { size: [1.0, 1.0, 1.0] }.build().unwrap());
    s.add_object("B", Primitive::Cube { size: [0.5, 0.5, 0.5] }.build().unwrap());
    let o = RenderOptions { view: View::Persp, width: 128, height: 128, overlays: Overlays::none(), ..Default::default() };
    r.render(&s, &o).unwrap();
    let first = r.stats().unwrap();
    assert_eq!((first.objects, first.meshes), (2, 2));
    // Re-render and orbit: nothing new is built.
    r.render(&s, &o).unwrap();
    r.render(&s, &RenderOptions { view: View::Front, ..o.clone() }).unwrap();
    let again = r.stats().unwrap();
    assert_eq!((again.objects, again.meshes, again.materials), (first.objects, first.meshes, first.materials));
    // Moving one object rebuilds only that one.
    s.object_mut(a).unwrap().transform.translation.x = 2.0;
    r.render(&s, &o).unwrap();
    assert_eq!(r.stats().unwrap().objects, 3);
    // Old entries are evicted after a while.
    for _ in 0..70 {
        r.render(&s, &o).unwrap();
    }
    assert_eq!(r.stats().unwrap().objects, 2);
}
