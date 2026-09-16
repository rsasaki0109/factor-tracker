//! `perception-core` is a detector-agnostic multi-object tracking and
//! multi-sensor fusion engine.
//!
//! It consumes discrete observations produced by external detectors or sensor
//! front-ends and performs prediction, data association, state estimation and
//! track lifecycle management. Sensor fusion is performed by updating a
//! *shared* track state with observations from several sensors — it is not an
//! average of independent tracker outputs.
//!
//! # What it provides
//!
//! * [`tracker::BBox2DTracker`] — axis-aligned 2D bounding boxes on one camera
//!   image plane (constant velocity, linear Kalman filter, IoU association,
//!   Hungarian assignment).
//! * [`tracker::Cartesian3DTracker`] — 3D positions in one common frame, with
//!   multi-sensor sequential fusion (constant velocity, linear Kalman filter,
//!   squared-Mahalanobis gating, Hungarian assignment).
//! * [`transform::RigidTransform3D`] — a validated helper to move positions and
//!   covariances between 3D frames before submitting them.
//!
//! # What it does not do
//!
//! There is no detection or inference, no raw image/point-cloud/radar
//! processing, no monocular depth, no appearance model, no 3D orientation or
//! extent estimation, no EKF/UKF, no JPDA/MHT, no track-to-track fusion, no
//! out-of-sequence measurement handling, no auto-calibration, no ego-motion
//! estimation and no ROS/Python/GPU integration. See `ROADMAP.md` in the
//! repository for the intended directions.
//!
//! # Contracts
//!
//! * **Time**: all timestamps are integer nanoseconds on one shared,
//!   pre-synchronized epoch. Observation times must be non-decreasing; a past
//!   batch is rejected and never treated as "now". Re-sent batches (same
//!   sensor and sequence) are rejected.
//! * **Frames**: every tracker is bound to one frame. A batch from another
//!   frame is rejected; renaming a frame is never treated as a transform.
//! * **Noise**: observations are assumed to have independent noise. Correlated
//!   observations are not modelled.
//! * **Existence**: detection `score` is a detector confidence, not a track
//!   existence probability. No existence probability is estimated or output.
//! * **Units**: 2D quantities are pixels (velocities pixels/second); 3D
//!   quantities are meters (velocities meters/second).
//!
//! # Reproducibility
//!
//! The library contains no randomness. Synthetic scenarios (in the CLI) use a
//! fixed, seeded PRNG so demos and tests are reproducible.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod assignment;
pub mod association;
pub mod error;
pub mod filter;
mod lifecycle;
pub mod math;
pub mod tracker;
pub mod transform;
pub mod types;

pub use assignment::{linear_assignment, Assignment};
pub use error::{InputError, NumericError, TrackerError};
pub use filter::KalmanFilter;
pub use tracker::{
    BBox2DTracker, BBox2DTrackerConfig, BatchReport3D, Cartesian3DTracker,
    Cartesian3DTrackerConfig, ObservationAssociation, ObservationOutcome, Track2DSnapshot,
    Track3DSnapshot,
};
pub use transform::{transform_covariance, transform_position, RigidTransform3D};
pub use types::{
    BBox2D, BBox2DObservation, ClassLabel, ClassPrior, FrameId, ObservationBatch2D,
    ObservationBatch3D, Position3DObservation, SensorId, Timestamp, TrackId, TrackLifecycleState,
    TrackerConfig,
};
