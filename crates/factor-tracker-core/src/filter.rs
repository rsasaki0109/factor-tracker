//! A statically sized linear Kalman filter.
//!
//! The filter is deliberately minimal: it exposes `predict` and `update` on a
//! fixed-size state and never allocates. It is used by both trackers with
//! different `(N, M)` dimensions.
//!
//! Numerical notes:
//!
//! * The Kalman gain is obtained by solving `S K^T = (P H^T)^T` with a
//!   Cholesky decomposition. The inverse of `S` is never formed explicitly.
//! * The covariance update uses the Joseph form
//!   `P = (I - K H) P (I - K H)^T + K R K^T`, which is numerically stable for
//!   ill-conditioned gains.
//! * `P` is symmetrized after every step to suppress round-off asymmetry.

use nalgebra::{Const, DimName, SMatrix, SVector};

use crate::error::{InputError, NumericError, TrackerError};
use crate::math::linalg::{
    ensure_finite_matrix, ensure_finite_vector, solve_spd, symmetrize, validate_covariance,
};

/// A linear Kalman filter with `N` state dimensions and `M` measurement
/// dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanFilter<const N: usize, const M: usize> {
    state: SVector<f64, N>,
    covariance: SMatrix<f64, N, N>,
}

impl<const N: usize, const M: usize> KalmanFilter<N, M>
where
    Const<N>: DimName,
    Const<M>: DimName,
{
    /// Creates a filter from an initial state and covariance.
    ///
    /// The covariance must be finite, symmetric and positive definite.
    pub fn new(
        state: SVector<f64, N>,
        covariance: SMatrix<f64, N, N>,
    ) -> Result<Self, TrackerError> {
        ensure_finite_vector(&state, "initial state")?;
        validate_covariance(&covariance, "initial state covariance")?;
        Ok(KalmanFilter { state, covariance })
    }

    /// Current state estimate.
    pub fn state(&self) -> &SVector<f64, N> {
        &self.state
    }

    /// Current state covariance.
    pub fn covariance(&self) -> &SMatrix<f64, N, N> {
        &self.covariance
    }

    /// Mutable access to the state estimate, for domain-specific post-processing
    /// such as clamping physically invalid box sizes.
    ///
    /// Callers must keep every value finite; the covariance is left untouched.
    pub fn state_mut(&mut self) -> &mut SVector<f64, N> {
        &mut self.state
    }

    /// Replaces the state and covariance, validating both.
    pub fn set_estimate(
        &mut self,
        state: SVector<f64, N>,
        covariance: SMatrix<f64, N, N>,
    ) -> Result<(), TrackerError> {
        ensure_finite_vector(&state, "state")?;
        validate_covariance(&covariance, "state covariance")?;
        self.state = state;
        self.covariance = covariance;
        Ok(())
    }

    /// Predicts the state forward using transition `f` and process noise `q`.
    ///
    /// `q` must be finite and symmetric; it is allowed to be positive
    /// semi-definite (for example all-zero when `dt = 0`).
    pub fn predict(
        &mut self,
        f: &SMatrix<f64, N, N>,
        q: &SMatrix<f64, N, N>,
    ) -> Result<(), TrackerError> {
        ensure_finite_matrix(f, "transition matrix")?;
        ensure_finite_matrix(q, "process noise")?;
        if !crate::math::linalg::is_symmetric(q) {
            return Err(InputError::InvalidCovariance {
                context: "process noise",
                reason: "matrix is not symmetric",
            }
            .into());
        }

        let predicted_state = f * self.state;
        let predicted_cov = f * self.covariance * f.transpose() + q;
        let predicted_cov = symmetrize(&predicted_cov);

        if !predicted_state.iter().all(|x| x.is_finite())
            || !predicted_cov.iter().all(|x| x.is_finite())
        {
            return Err(NumericError::NonFiniteResult {
                context: "Kalman prediction",
            }
            .into());
        }

        self.state = predicted_state;
        self.covariance = predicted_cov;
        Ok(())
    }

    /// Updates the state with measurement `z`, measurement matrix `h` and
    /// measurement covariance `r`.
    ///
    /// `r` must be finite, symmetric and positive definite.
    pub fn update(
        &mut self,
        z: &SVector<f64, M>,
        h: &SMatrix<f64, M, N>,
        r: &SMatrix<f64, M, M>,
    ) -> Result<(), TrackerError> {
        ensure_finite_vector(z, "measurement")?;
        ensure_finite_matrix(h, "measurement matrix")?;
        validate_covariance(r, "measurement covariance")?;

        let innovation = z - h * self.state;
        let innovation_cov = h * self.covariance * h.transpose() + r;
        let innovation_cov = symmetrize(&innovation_cov);

        // K = P H^T S^-1, solved as S X = (P H^T)^T, K = X^T.
        let pht = self.covariance * h.transpose();
        let solved = solve_spd(&innovation_cov, &pht.transpose(), "innovation covariance")?;
        let gain = solved.transpose();

        let updated_state = self.state + gain * innovation;

        let identity = SMatrix::<f64, N, N>::identity();
        let i_kh = identity - gain * h;
        let updated_cov = i_kh * self.covariance * i_kh.transpose() + gain * r * gain.transpose();
        let updated_cov = symmetrize(&updated_cov);

        if !updated_state.iter().all(|x| x.is_finite())
            || !updated_cov.iter().all(|x| x.is_finite())
        {
            return Err(NumericError::NonFiniteResult {
                context: "Kalman update",
            }
            .into());
        }

        self.state = updated_state;
        self.covariance = updated_cov;
        Ok(())
    }

    /// Squared Mahalanobis distance of a measurement from the current estimate,
    /// using `H P H^T + R` as the innovation covariance.
    pub fn squared_mahalanobis(
        &self,
        z: &SVector<f64, M>,
        h: &SMatrix<f64, M, N>,
        r: &SMatrix<f64, M, M>,
    ) -> Result<f64, TrackerError> {
        ensure_finite_vector(z, "measurement")?;
        ensure_finite_matrix(h, "measurement matrix")?;
        validate_covariance(r, "measurement covariance")?;
        let innovation = z - h * self.state;
        let innovation_cov = symmetrize(&(h * self.covariance * h.transpose() + r));
        Ok(crate::math::linalg::squared_mahalanobis(
            &innovation,
            &innovation_cov,
            "innovation covariance",
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::models::cv_q6;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix1, Vector1};

    #[test]
    fn known_numeric_update() {
        // State [pos, vel], H = [1, 0], R = [1], z = 2.
        let mut filter = KalmanFilter::<2, 1>::new(
            SVector::<f64, 2>::new(0.0, 0.0),
            SMatrix::<f64, 2, 2>::identity(),
        )
        .unwrap();
        let h = SMatrix::<f64, 1, 2>::new(1.0, 0.0);
        let r = Matrix1::new(1.0);
        filter.update(&Vector1::new(2.0), &h, &r).unwrap();
        assert_relative_eq!(filter.state()[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(filter.state()[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(filter.covariance()[(0, 0)], 0.5, epsilon = 1e-12);
        assert_relative_eq!(filter.covariance()[(1, 1)], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn constant_velocity_prediction() {
        let mut filter = KalmanFilter::<2, 1>::new(
            SVector::<f64, 2>::new(0.0, 10.0),
            SMatrix::<f64, 2, 2>::identity(),
        )
        .unwrap();
        let f = SMatrix::<f64, 2, 2>::new(1.0, 2.0, 0.0, 1.0);
        let q = SMatrix::<f64, 2, 2>::zeros();
        filter.predict(&f, &q).unwrap();
        assert_relative_eq!(filter.state()[0], 20.0, epsilon = 1e-12);
        assert_relative_eq!(filter.state()[1], 10.0, epsilon = 1e-12);
    }

    #[test]
    fn process_noise_scales_with_dt() {
        let q = 3.0;
        let q1 = cv_q6(1.0, q);
        let q2 = cv_q6(2.0, q);
        // dt^3/3 term scales by 8, dt^2/2 term by 4, dt term by 2.
        assert_relative_eq!(q2[(0, 0)] / q1[(0, 0)], 8.0, epsilon = 1e-12);
        assert_relative_eq!(q2[(0, 3)] / q1[(0, 3)], 4.0, epsilon = 1e-12);
        assert_relative_eq!(q2[(3, 3)] / q1[(3, 3)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(q2[(0, 0)], q * 8.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn zero_dt_prediction_is_noop() {
        let mut filter = KalmanFilter::<2, 1>::new(
            SVector::<f64, 2>::new(1.0, 2.0),
            SMatrix::<f64, 2, 2>::identity(),
        )
        .unwrap();
        let state_before = *filter.state();
        let cov_before = *filter.covariance();
        filter
            .predict(
                &SMatrix::<f64, 2, 2>::identity(),
                &SMatrix::<f64, 2, 2>::zeros(),
            )
            .unwrap();
        assert_eq!(*filter.state(), state_before);
        assert_eq!(*filter.covariance(), cov_before);
    }

    #[test]
    fn sequential_updates_match_stacked_update() {
        // Two independent observations of the same position at the same time.
        let initial_state = SVector::<f64, 2>::new(0.0, 1.0);
        let initial_cov = SMatrix::<f64, 2, 2>::new(2.0, 0.3, 0.3, 1.5);

        let h1 = SMatrix::<f64, 1, 2>::new(1.0, 0.0);
        let h2 = SMatrix::<f64, 1, 2>::new(1.0, 0.0);
        let r1 = Matrix1::new(0.5);
        let r2 = Matrix1::new(0.8);
        let z1 = Vector1::new(1.0);
        let z2 = Vector1::new(1.4);

        let mut sequential = KalmanFilter::<2, 1>::new(initial_state, initial_cov).unwrap();
        sequential.update(&z1, &h1, &r1).unwrap();
        sequential.update(&z2, &h2, &r2).unwrap();

        let mut stacked = KalmanFilter::<2, 2>::new(initial_state, initial_cov).unwrap();
        let h = SMatrix::<f64, 2, 2>::new(1.0, 0.0, 1.0, 0.0);
        let r = SMatrix::<f64, 2, 2>::new(0.5, 0.0, 0.0, 0.8);
        let z = SVector::<f64, 2>::new(1.0, 1.4);
        stacked.update(&z, &h, &r).unwrap();

        for i in 0..2 {
            assert_relative_eq!(sequential.state()[i], stacked.state()[i], epsilon = 1e-9);
            for j in 0..2 {
                assert_relative_eq!(
                    sequential.covariance()[(i, j)],
                    stacked.covariance()[(i, j)],
                    epsilon = 1e-9
                );
            }
        }
    }

    #[test]
    fn covariance_stays_symmetric() {
        let mut filter = KalmanFilter::<2, 1>::new(
            SVector::<f64, 2>::new(0.0, 0.0),
            SMatrix::<f64, 2, 2>::new(1.0, 0.2, 0.2, 1.0),
        )
        .unwrap();
        let h = SMatrix::<f64, 1, 2>::new(1.0, 0.0);
        filter
            .update(&Vector1::new(0.5), &h, &Matrix1::new(0.3))
            .unwrap();
        let p = filter.covariance();
        assert_relative_eq!(p[(0, 1)], p[(1, 0)], epsilon = 1e-15);
    }

    #[test]
    fn non_positive_definite_measurement_covariance_is_rejected() {
        let mut filter =
            KalmanFilter::<2, 1>::new(SVector::<f64, 2>::zeros(), SMatrix::<f64, 2, 2>::identity())
                .unwrap();
        let h = SMatrix::<f64, 1, 2>::new(1.0, 0.0);
        let error = filter
            .update(&Vector1::new(0.0), &h, &Matrix1::new(0.0))
            .unwrap_err();
        assert!(error.is_numeric_error());
    }

    #[test]
    fn nan_measurement_is_rejected_as_input_error() {
        let mut filter =
            KalmanFilter::<2, 1>::new(SVector::<f64, 2>::zeros(), SMatrix::<f64, 2, 2>::identity())
                .unwrap();
        let h = SMatrix::<f64, 1, 2>::new(1.0, 0.0);
        let error = filter
            .update(&Vector1::new(f64::NAN), &h, &Matrix1::new(1.0))
            .unwrap_err();
        assert!(error.is_input_error());
    }

    #[test]
    fn non_symmetric_initial_covariance_is_rejected() {
        let bad = SMatrix::<f64, 2, 2>::new(1.0, 0.5, 0.0, 1.0);
        assert!(KalmanFilter::<2, 1>::new(SVector::<f64, 2>::zeros(), bad).is_err());
    }
}
