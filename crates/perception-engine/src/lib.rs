//! `perception-engine` is the camera-assisted LiDAR tracking SDK.
//!
//! It sits on top of `perception-core` and turns sensor-native detections,
//! calibration and timestamps into 3D tracks plus association diagnostics,
//! without the user having to assemble the pipeline themselves.
//!
//! Pipeline:
//!
//! ```text
//! sensor detections
//!     -> bounded arrival-order reorder buffer (event time preserved)
//!     -> per-observation coordinate normalization (calibration + ego pose)
//!     -> LiDAR position tracking (perception-core Cartesian3DTracker)
//!     -> geometric camera association
//!     -> camera class evidence as a soft prior (optional)
//!     -> unified track output + diagnostics
//!     -> comparison / evaluation / HTML report
//! ```
//!
//! The camera model is a rectified pinhole. Distortion is not modelled.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod camera;
pub mod diagnostics;
pub mod engine;
pub mod error;
pub mod eval;
pub mod fgo;
pub mod fixture;
pub mod geometry;
pub mod inject;
pub mod input;
pub mod io;
pub mod kitti;
pub mod reorder;
pub mod replay;
pub mod setup;

pub use camera::{
    associate_camera, CameraAssociationConfig, CameraDetectionMatch, ClassEvidence,
    ClassEvidenceConfig,
};
pub use diagnostics::{
    AllCounters, DiagnosticEvent, DiagnosticLevel, DiagnosticLog, EngineCounters,
};
pub use engine::{
    CameraAssistedLidarEngine, CameraFrameOutput, EngineConfig, LidarFrameOutput, TrackOutput,
    TrackingMode,
};
pub use error::{EngineError, EngineResult};
pub use eval::{
    evaluate_mode, CameraAssociationTruth, EvalConfig, GroundTruthFrame, GroundTruthObject,
    ModeMetrics,
};
pub use fgo::{FgoConfig, FgoCounters, FixedLagRefiner};
pub use geometry::{
    compose, inverse, orthonormalize_rotation, Box3DShape, BoxDimensions, Calibration,
    CameraIntrinsics, PoseTrack, ProjectedBox, ProjectionRejection,
};
pub use input::{CameraBatch, CameraDetection, ClassMap, EngineBatch, LidarBatch, LidarDetection};
pub use reorder::{AsyncReorderBuffer, IngestOutcome, ReorderConfig, ReorderCounters};
pub use setup::EngineSetup;
