//! polly-core uses a newer glam than Bevy; convert through plain arrays.

pub fn m4(m: polly_core::glam::Mat4) -> bevy::math::Mat4 {
    bevy::math::Mat4::from_cols_array(&m.to_cols_array())
}

pub fn v3(v: polly_core::glam::Vec3) -> bevy::math::Vec3 {
    bevy::math::Vec3::from_array(v.to_array())
}
