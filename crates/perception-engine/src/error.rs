//! Errors returned by the engine layer.

use perception_core::TrackerError;

/// Errors produced by the engine layer.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The tracking core rejected a batch.
    #[error(transparent)]
    Core(#[from] TrackerError),
    /// Calibration or configuration was invalid.
    #[error("invalid calibration/configuration: {reason}")]
    Config {
        /// Description of the problem.
        reason: String,
    },
    /// A pose needed for a transform was not available.
    #[error("pose unavailable: {reason}")]
    PoseUnavailable {
        /// Description of the problem.
        reason: String,
    },
    /// Input data was malformed.
    #[error("invalid input: {reason}")]
    Data {
        /// Description of the problem.
        reason: String,
    },
    /// File I/O failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<perception_core::NumericError> for EngineError {
    fn from(error: perception_core::NumericError) -> Self {
        EngineError::Core(perception_core::TrackerError::Numeric(error))
    }
}

impl EngineError {
    /// Convenience constructor for a configuration error.
    pub fn config(reason: impl Into<String>) -> Self {
        EngineError::Config {
            reason: reason.into(),
        }
    }

    /// Convenience constructor for a data error.
    pub fn data(reason: impl Into<String>) -> Self {
        EngineError::Data {
            reason: reason.into(),
        }
    }
}

/// Convenience result type for the engine layer.
pub type EngineResult<T> = Result<T, EngineError>;
