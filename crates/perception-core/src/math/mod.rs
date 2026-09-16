//! Numerical helpers and motion/measurement models.
//!
//! Implemented as a small module tree rather than a trait hierarchy: only the
//! operations the two trackers actually need are provided.

pub mod linalg;
pub mod models;

pub use linalg::{squared_mahalanobis, symmetrize, validate_covariance};
pub use models::{cv_f6, cv_f8, cv_q6, cv_q8, measurement_h3, measurement_h4};
