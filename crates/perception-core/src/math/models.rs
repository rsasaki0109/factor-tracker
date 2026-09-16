//! Constant-velocity motion and position measurement models.
//!
//! # State vectors
//!
//! 2D tracker state (8 elements), all in image pixels:
//!
//! ```text
//! [ cx, cy, w, h, vcx, vcy, vw, vh ]
//! ```
//!
//! where `cx, cy` is the box center, `w, h` are its width and height, and the
//! velocities are in pixels per second.
//!
//! 3D tracker state (6 elements), position in meters and velocity in
//! meters per second:
//!
//! ```text
//! [ x, y, z, vx, vy, vz ]
//! ```
//!
//! # Process noise
//!
//! Both models use the continuous-time white-acceleration model. For a state
//! ordered `[position, velocity]` along one axis with acceleration spectral
//! density `q` (units: position-unit squared per second cubed):
//!
//! ```text
//! Q_axis = q * [ dt^3/3  dt^2/2 ]
//!             [ dt^2/2  dt     ]
//! ```
//!
//! Axes are independent, so `Q` is block diagonal. For the 2D tracker the
//! center axes use `q_pos` (pixels^2 / s^3) and the size axes use `q_size`
//! (pixels^2 / s^3). For the 3D tracker a single `q` (m^2 / s^3) is used.

use nalgebra::SMatrix;

/// Constant-velocity transition for the 6-element 3D state.
pub fn cv_f6(dt: f64) -> SMatrix<f64, 6, 6> {
    let mut f = SMatrix::<f64, 6, 6>::identity();
    for i in 0..3 {
        f[(i, i + 3)] = dt;
    }
    f
}

/// Constant-velocity process noise for the 6-element 3D state.
pub fn cv_q6(dt: f64, q: f64) -> SMatrix<f64, 6, 6> {
    let mut out = SMatrix::<f64, 6, 6>::zeros();
    for i in 0..3 {
        add_acceleration_block(&mut out, i, i + 3, dt, q);
    }
    out
}

/// Constant-velocity transition for the 8-element 2D state.
pub fn cv_f8(dt: f64) -> SMatrix<f64, 8, 8> {
    let mut f = SMatrix::<f64, 8, 8>::identity();
    for i in 0..4 {
        f[(i, i + 4)] = dt;
    }
    f
}

/// Constant-velocity process noise for the 8-element 2D state.
///
/// The two center axes use `q_pos`; the two size axes use `q_size`.
pub fn cv_q8(dt: f64, q_pos: f64, q_size: f64) -> SMatrix<f64, 8, 8> {
    let mut out = SMatrix::<f64, 8, 8>::zeros();
    for i in 0..4 {
        let q = if i < 2 { q_pos } else { q_size };
        add_acceleration_block(&mut out, i, i + 4, dt, q);
    }
    out
}

fn add_acceleration_block<const N: usize>(
    out: &mut SMatrix<f64, N, N>,
    pos: usize,
    vel: usize,
    dt: f64,
    q: f64,
) {
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    out[(pos, pos)] += q * dt3 / 3.0;
    out[(pos, vel)] += q * dt2 / 2.0;
    out[(vel, pos)] += q * dt2 / 2.0;
    out[(vel, vel)] += q * dt;
}

/// Measurement matrix mapping the 8-element 2D state to `[cx, cy, w, h]`.
pub fn measurement_h4() -> SMatrix<f64, 4, 8> {
    let mut h = SMatrix::<f64, 4, 8>::zeros();
    for i in 0..4 {
        h[(i, i)] = 1.0;
    }
    h
}

/// Measurement matrix mapping the 6-element 3D state to `[x, y, z]`.
pub fn measurement_h3() -> SMatrix<f64, 3, 6> {
    let mut h = SMatrix::<f64, 3, 6>::zeros();
    for i in 0..3 {
        h[(i, i)] = 1.0;
    }
    h
}
