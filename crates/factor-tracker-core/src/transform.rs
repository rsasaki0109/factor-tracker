//! Rigid-transform helpers for 3D positions and covariances.
//!
//! Given a known rotation `R` and translation `t` mapping a *source* frame to
//! a *target* frame:
//!
//! ```text
//! p_target = R * p_source + t
//! C_target = R * C_source * R^T
//! ```
//!
//! The transform itself is treated as exact: **the uncertainty of `R` and `t`
//! is not modelled or propagated.** If the transform is itself uncertain the
//! caller must account for that separately.
//!
//! For a moving sensor, the caller is responsible for supplying the sensor
//! pose that corresponds to each observation's timestamp. The core does not
//! estimate ego-motion.

use nalgebra::{Matrix3, Vector3};

use crate::error::{InputError, TrackerError};

/// Numerical tolerance for checking that a rotation matrix is orthonormal.
pub const ROTATION_TOL: f64 = 1e-9;

/// A validated rigid body transform between two 3D frames.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidTransform3D {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

impl RigidTransform3D {
    /// Creates and validates a rigid transform.
    ///
    /// `rotation` must be a proper rotation: finite, orthonormal
    /// (`R R^T = I`) with determinant `+1` (reflections are rejected).
    pub fn new(rotation: Matrix3<f64>, translation: Vector3<f64>) -> Result<Self, InputError> {
        if !rotation.iter().all(|x| x.is_finite()) {
            return Err(InputError::NonFiniteValue {
                context: "rotation matrix",
            });
        }
        if !translation.iter().all(|x| x.is_finite()) {
            return Err(InputError::NonFiniteValue {
                context: "translation vector",
            });
        }
        let orthogonality_error = (rotation * rotation.transpose() - Matrix3::identity())
            .iter()
            .fold(0.0_f64, |acc, x| acc.max(x.abs()));
        if orthogonality_error > ROTATION_TOL {
            return Err(InputError::InvalidRotation {
                reason: "matrix is not orthonormal",
            });
        }
        let det = rotation.determinant();
        if (det - 1.0).abs() > ROTATION_TOL {
            return Err(InputError::InvalidRotation {
                reason: "determinant is not +1 (improper rotation or reflection)",
            });
        }
        Ok(RigidTransform3D {
            rotation,
            translation,
        })
    }

    /// The identity transform.
    pub fn identity() -> Self {
        RigidTransform3D {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Rotation component.
    pub fn rotation(&self) -> &Matrix3<f64> {
        &self.rotation
    }

    /// Translation component.
    pub fn translation(&self) -> &Vector3<f64> {
        &self.translation
    }

    /// Transforms a position.
    pub fn transform_position(&self, position: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * position + self.translation
    }

    /// Transforms a position covariance: `R C R^T`.
    ///
    /// The covariance must be finite and symmetric.
    pub fn transform_covariance(
        &self,
        covariance: &Matrix3<f64>,
    ) -> Result<Matrix3<f64>, TrackerError> {
        transform_covariance(&self.rotation, covariance)
    }
}

/// Transforms a position from a source frame into a target frame:
/// `p_target = R * p_source + t`.
///
/// No validation of `rotation` is performed here; construct a
/// [`RigidTransform3D`] to validate it once and reuse it.
pub fn transform_position(
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    position: &Vector3<f64>,
) -> Vector3<f64> {
    rotation * position + translation
}

/// Transforms a position covariance from a source frame into a target frame:
/// `C_target = R * C_source * R^T`.
///
/// The covariance must be finite and symmetric. Uncertainty of the rotation
/// itself is not propagated.
pub fn transform_covariance(
    rotation: &Matrix3<f64>,
    covariance: &Matrix3<f64>,
) -> Result<Matrix3<f64>, TrackerError> {
    if !covariance.iter().all(|x| x.is_finite()) {
        return Err(InputError::NonFiniteValue {
            context: "position covariance",
        }
        .into());
    }
    if !crate::math::linalg::is_symmetric(covariance) {
        return Err(InputError::InvalidCovariance {
            context: "position covariance",
            reason: "matrix is not symmetric",
        }
        .into());
    }
    let transformed = rotation * covariance * rotation.transpose();
    let transformed = crate::math::linalg::symmetrize(&transformed);
    if !transformed.iter().all(|x| x.is_finite()) {
        return Err(crate::error::NumericError::NonFiniteResult {
            context: "covariance transform",
        }
        .into());
    }
    Ok(transformed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn rotation_z(angle: f64) -> Matrix3<f64> {
        let (s, c) = angle.sin_cos();
        Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    }

    #[test]
    fn position_and_covariance_transform() {
        let transform = RigidTransform3D::new(
            rotation_z(core::f64::consts::FRAC_PI_2),
            Vector3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let rotated = transform.transform_position(&Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(rotated, Vector3::new(1.0, 1.0, 0.0), epsilon = 1e-12);

        let covariance = Matrix3::new(4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 9.0);
        let transformed = transform.transform_covariance(&covariance).unwrap();
        let expected = Matrix3::new(1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0);
        assert_relative_eq!(transformed, expected, epsilon = 1e-12);
    }

    #[test]
    fn identity_transform_is_noop() {
        let transform = RigidTransform3D::identity();
        let p = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(transform.transform_position(&p), p);
    }

    #[test]
    fn non_orthonormal_rotation_is_rejected() {
        let bad = Matrix3::new(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);
        assert!(RigidTransform3D::new(bad, Vector3::zeros()).is_err());
    }

    #[test]
    fn reflection_is_rejected() {
        let reflection = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let err = RigidTransform3D::new(reflection, Vector3::zeros()).unwrap_err();
        assert!(matches!(err, InputError::InvalidRotation { .. }));
    }

    #[test]
    fn non_finite_inputs_are_rejected() {
        assert!(RigidTransform3D::new(rotation_z(0.0), Vector3::new(f64::NAN, 0.0, 0.0)).is_err());
    }

    #[test]
    fn asymmetric_covariance_is_rejected() {
        let bad = Matrix3::new(1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(transform_covariance(&Matrix3::identity(), &bad).is_err());
    }
}
