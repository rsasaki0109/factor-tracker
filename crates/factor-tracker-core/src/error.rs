//! Error types produced by the tracking core.
//!
//! Errors are split into two disjoint categories so callers can react
//! appropriately:
//!
//! * [`InputError`] — the caller supplied data that violates the documented
//!   contract (non-finite values, malformed covariances, timestamps that go
//!   backwards, duplicated sensor batches, ...). These are *programming or
//!   integration* mistakes and should generally be surfaced to the caller
//!   rather than silently repaired.
//! * [`NumericError`] — the data was contractually valid but a numerical
//!   operation could not be completed (for example a non-positive-definite
//!   innovation covariance). These indicate ill-conditioned input or a
//!   modelling problem.
//!
//! The core never silently substitutes a zero or identity matrix for a failed
//! decomposition: a failure is always reported.

use crate::types::{SensorId, Timestamp};

/// Errors caused by invalid input or by violating a documented contract.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InputError {
    /// A timestamp was older than the tracker's current time.
    #[error("observation timestamp {received_ns} ns is earlier than current time {current_ns} ns")]
    TimestampWentBackwards {
        /// The timestamp that was received.
        received_ns: i64,
        /// The tracker's current time before applying the batch.
        current_ns: i64,
    },
    /// A timestamp subtraction overflowed `i64` nanoseconds.
    #[error("timestamp difference overflowed i64 nanoseconds")]
    TimestampOverflow,
    /// A batch with a `sequence` that was already seen (or older) was submitted.
    #[error(
        "duplicate or out-of-order batch for sensor {sensor}: received sequence {received}, last seen {last_seen}"
    )]
    DuplicateBatch {
        /// Sensor that produced the batch.
        sensor: SensorId,
        /// Sequence number of the rejected batch.
        received: u64,
        /// Highest sequence number previously accepted for this sensor.
        last_seen: u64,
    },
    /// The observation frame does not match the tracker frame.
    #[error("frame mismatch: tracker expects '{expected}', observation declares '{found}'")]
    FrameMismatch {
        /// Frame the tracker is configured for.
        expected: String,
        /// Frame declared by the observation batch.
        found: String,
    },
    /// A scalar, vector or matrix element was NaN or infinite.
    #[error("non-finite value in {context}")]
    NonFiniteValue {
        /// Human readable description of where the value came from.
        context: &'static str,
    },
    /// A covariance matrix was rejected.
    #[error("invalid covariance for {context}: {reason}")]
    InvalidCovariance {
        /// Human readable description of the covariance.
        context: &'static str,
        /// Why the covariance was rejected.
        reason: &'static str,
    },
    /// A bounding box had non-positive width or height.
    #[error("invalid bounding box: width={width}, height={height} (both must be > 0)")]
    InvalidBBox {
        /// Reported width.
        width: f64,
        /// Reported height.
        height: f64,
    },
    /// A configuration value was inconsistent or out of range.
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// Why the configuration was rejected.
        reason: &'static str,
    },
    /// A rotation matrix was not a proper orthonormal transform.
    #[error("invalid rotation matrix: {reason}")]
    InvalidRotation {
        /// Why the rotation was rejected.
        reason: &'static str,
    },
    /// A batch declared a different sensor than an observation it contains, or
    /// a batch mixed incompatible observation payloads.
    #[error("inconsistent observation: {reason}")]
    InconsistentObservation {
        /// Why the observation was rejected.
        reason: &'static str,
    },
    /// A detection score was outside `[0, 1]`.
    #[error("detection score {score} is outside [0, 1]")]
    InvalidScore {
        /// The offending score.
        score: f64,
    },
}

/// Errors produced by numerical algorithms operating on valid input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum NumericError {
    /// A Cholesky decomposition failed because a matrix was not positive
    /// definite (within the requested tolerance).
    #[error("matrix is not positive definite in {context}")]
    NotPositiveDefinite {
        /// Where the decomposition happened.
        context: &'static str,
    },
    /// A numerical operation produced a non-finite intermediate value.
    #[error("non-finite intermediate value in {context}")]
    NonFiniteResult {
        /// Where the value was produced.
        context: &'static str,
    },
    /// The linear assignment solver could not find a finite-cost complete
    /// assignment. In practice this should not happen because the solver pads
    /// the cost matrix with finite "unmatched" costs.
    #[error("linear assignment failed to produce a complete assignment")]
    AssignmentFailed,
    /// A square linear system could not be solved.
    #[error("failed to solve linear system in {context}")]
    SolveFailed {
        /// Where the solve happened.
        context: &'static str,
    },
}

/// Top level error returned by tracker operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TrackerError {
    /// Invalid input.
    #[error(transparent)]
    Input(#[from] InputError),
    /// Numerical failure on valid input.
    #[error(transparent)]
    Numeric(#[from] NumericError),
}

impl TrackerError {
    /// Convenience constructor for an [`InputError`].
    pub fn input(error: InputError) -> Self {
        TrackerError::Input(error)
    }

    /// Returns `true` when this is an [`InputError`].
    pub fn is_input_error(&self) -> bool {
        matches!(self, TrackerError::Input(_))
    }

    /// Returns `true` when this is a [`NumericError`].
    pub fn is_numeric_error(&self) -> bool {
        matches!(self, TrackerError::Numeric(_))
    }
}

/// Helper used by the trackers to reject observations whose timestamp is not
/// on or after the tracker's current time.
pub(crate) fn check_timestamp_order(
    current: Option<Timestamp>,
    received: Timestamp,
) -> Result<(), InputError> {
    if let Some(current) = current {
        if received < current {
            return Err(InputError::TimestampWentBackwards {
                received_ns: received.as_nanos(),
                current_ns: current.as_nanos(),
            });
        }
    }
    Ok(())
}
