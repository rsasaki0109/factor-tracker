//! Trackers.
//!
//! Two concrete trackers are provided:
//!
//! * [`BBox2DTracker`] tracks axis-aligned bounding boxes on a single camera
//!   image plane.
//! * [`Cartesian3DTracker`] tracks 3D positions expressed in one common
//!   tracking frame and supports fusing several sensors into the same tracks.
//!
//! Both share the same lifecycle logic, linear Kalman filter and linear
//! assignment solver. They are kept as separate types because their state
//! spaces, units and association costs differ; they are not forced into one
//! ambiguous "generic object" type.

pub mod tracker2d;
pub mod tracker3d;

pub use tracker2d::{BBox2DTracker, BBox2DTrackerConfig, Track2DSnapshot};
pub use tracker3d::{
    BatchReport3D, Cartesian3DTracker, Cartesian3DTrackerConfig, ObservationAssociation,
    ObservationOutcome, Track3DSnapshot,
};

use crate::error::InputError;
use crate::types::SensorId;

/// Validates a detection score.
pub(crate) fn validate_score(score: f64) -> Result<(), InputError> {
    if !score.is_finite() {
        return Err(InputError::NonFiniteValue {
            context: "detection score",
        });
    }
    if !(0.0..=1.0).contains(&score) {
        return Err(InputError::InvalidScore { score });
    }
    Ok(())
}

/// Adds `sensor` to a bounded, de-duplicated list of contributing sensors.
///
/// When the list is full the oldest sensor is dropped.
pub(crate) fn record_sensor(sensors: &mut Vec<SensorId>, sensor: SensorId, max: usize) {
    if sensors.contains(&sensor) {
        return;
    }
    if max > 0 && sensors.len() >= max {
        sensors.remove(0);
    }
    sensors.push(sensor);
}
