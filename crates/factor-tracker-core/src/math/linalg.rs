//! Linear-algebra helpers used by the filters and association code.
//!
//! All routines operate on statically sized `nalgebra` matrices so the two
//! trackers incur no heap allocation inside the filter loop.

use nalgebra::{Const, DimName, SMatrix, SVector};

use crate::error::{InputError, NumericError, TrackerError};

/// Tolerance used when checking that a covariance matrix is symmetric.
///
/// The check is relative to the largest magnitude entry so that very small or
/// very large covariances are handled consistently.
pub const SYMMETRY_REL_TOL: f64 = 1e-9;
/// Absolute floor used together with [`SYMMETRY_REL_TOL`].
pub const SYMMETRY_ABS_TOL: f64 = 1e-12;

/// Returns `true` if every element of a rectangular matrix is finite.
pub fn is_finite_matrix<const R: usize, const C: usize>(m: &SMatrix<f64, R, C>) -> bool {
    m.iter().all(|x| x.is_finite())
}

/// Rejects a rectangular matrix containing NaN or infinity.
pub fn ensure_finite_matrix<const R: usize, const C: usize>(
    m: &SMatrix<f64, R, C>,
    context: &'static str,
) -> Result<(), InputError> {
    if is_finite_matrix(m) {
        Ok(())
    } else {
        Err(InputError::NonFiniteValue { context })
    }
}

/// Rejects a vector containing NaN or infinity.
pub fn ensure_finite_vector<const N: usize>(
    v: &SVector<f64, N>,
    context: &'static str,
) -> Result<(), InputError> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(InputError::NonFiniteValue { context })
    }
}

/// Maximum absolute difference between `m` and its transpose.
pub fn asymmetry<const N: usize>(m: &SMatrix<f64, N, N>) -> f64 {
    let diff = m - m.transpose();
    diff.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()))
}

/// Returns `true` when `m` is symmetric within [`SYMMETRY_REL_TOL`].
pub fn is_symmetric<const N: usize>(m: &SMatrix<f64, N, N>) -> bool {
    let scale = m.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    asymmetry(m) <= SYMMETRY_ABS_TOL + SYMMETRY_REL_TOL * scale
}

/// Returns the symmetric part `(m + m^T) / 2` of a matrix.
pub fn symmetrize<const N: usize>(m: &SMatrix<f64, N, N>) -> SMatrix<f64, N, N> {
    0.5 * (m + m.transpose())
}

/// Validates that a matrix is finite, symmetric and positive definite.
///
/// Used for observation covariances and initial state covariances. A matrix
/// that passes this check can be handed to a Cholesky decomposition without a
/// fallback.
pub fn validate_covariance<const N: usize>(
    m: &SMatrix<f64, N, N>,
    context: &'static str,
) -> Result<(), TrackerError>
where
    Const<N>: DimName,
{
    ensure_finite_matrix(m, context)?;
    if !is_symmetric(m) {
        return Err(InputError::InvalidCovariance {
            context,
            reason: "matrix is not symmetric",
        }
        .into());
    }
    (*m).cholesky()
        .ok_or(TrackerError::Numeric(NumericError::NotPositiveDefinite {
            context,
        }))?;
    Ok(())
}

/// Solves `A X = B` for a symmetric positive-definite `A` using Cholesky.
///
/// Returns [`NumericError::NotPositiveDefinite`] when `A` cannot be
/// decomposed. No fallback matrix is substituted.
pub fn solve_spd<const N: usize, const M: usize>(
    a: &SMatrix<f64, N, N>,
    b: &SMatrix<f64, N, M>,
    context: &'static str,
) -> Result<SMatrix<f64, N, M>, NumericError>
where
    Const<N>: DimName,
    Const<M>: DimName,
{
    let chol = (*a)
        .cholesky()
        .ok_or(NumericError::NotPositiveDefinite { context })?;
    let solution = chol.solve(b);
    if !solution.iter().all(|x| x.is_finite()) {
        return Err(NumericError::NonFiniteResult { context });
    }
    Ok(solution)
}

/// Solves `A x = b` for a symmetric positive-definite `A` using Cholesky.
pub fn solve_spd_vector<const N: usize>(
    a: &SMatrix<f64, N, N>,
    b: &SVector<f64, N>,
    context: &'static str,
) -> Result<SVector<f64, N>, NumericError>
where
    Const<N>: DimName,
{
    let chol = (*a)
        .cholesky()
        .ok_or(NumericError::NotPositiveDefinite { context })?;
    let solution = chol.solve(b);
    if !solution.iter().all(|x| x.is_finite()) {
        return Err(NumericError::NonFiniteResult { context });
    }
    Ok(solution)
}

/// Squared Mahalanobis distance `d^2 = u^T S^{-1} u` for a symmetric
/// positive-definite `S`.
///
/// The inverse is never formed explicitly; a Cholesky solve is used instead.
/// Tiny negative values caused by floating-point round-off are clamped to
/// zero.
pub fn squared_mahalanobis<const N: usize>(
    innovation: &SVector<f64, N>,
    covariance: &SMatrix<f64, N, N>,
    context: &'static str,
) -> Result<f64, NumericError>
where
    Const<N>: DimName,
{
    let solved = solve_spd_vector(covariance, innovation, context)?;
    let d2 = innovation.dot(&solved);
    if !d2.is_finite() {
        return Err(NumericError::NonFiniteResult { context });
    }
    Ok(d2.max(0.0))
}
