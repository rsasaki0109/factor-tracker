//! Axis-aligned 2D bounding-box tracker for a single camera image plane.
//!
//! # State and units
//!
//! The Kalman state is
//!
//! ```text
//! [ cx, cy, w, h, vcx, vcy, vw, vh ]
//! ```
//!
//! where `cx, cy` are the box center and `w, h` its size, all in image
//! **pixels**; the velocities are in **pixels per second**. The measurement is
//! `[cx, cy, w, h]` and its `4x4` covariance is in `pixels^2`.
//!
//! # Scope
//!
//! One tracker instance handles exactly one camera frame. Pixel coordinates
//! from different cameras must not be fused into the same instance: give each
//! camera its own tracker (and, if needed, its own logical frame name).
//! Observations whose frame differs from the tracker frame are rejected.
//!
//! Association uses an IoU cost with a hard gate, class gating, and the shared
//! Hungarian solver. There is no appearance model and no existence-probability
//! model; `score` is only used to decide whether a new track may be spawned.

use std::collections::BTreeMap;

use nalgebra::{SMatrix, SVector, Vector2};

use crate::assignment::linear_assignment;
use crate::association::{classes_compatible, iou_association_cost};
use crate::error::{check_timestamp_order, InputError, TrackerError};
use crate::filter::KalmanFilter;
use crate::lifecycle::TrackLifecycle;
use crate::math::linalg::validate_covariance;
use crate::math::models::{cv_f8, cv_q8, measurement_h4};
use crate::tracker::{record_sensor, validate_score};
use crate::types::{
    BBox2D, BBox2DObservation, ClassLabel, FrameId, ObservationBatch2D, SensorId, Timestamp,
    TrackId, TrackLifecycleState, TrackerConfig,
};

const IDX_CX: usize = 0;
const IDX_CY: usize = 1;
const IDX_W: usize = 2;
const IDX_H: usize = 3;

/// Configuration for [`BBox2DTracker`].
#[derive(Debug, Clone, PartialEq)]
pub struct BBox2DTrackerConfig {
    /// Shared lifecycle parameters.
    pub common: TrackerConfig,
    /// The single image frame this tracker accepts.
    pub frame_id: FrameId,
    /// Acceleration spectral density for the two center axes, in
    /// `pixels^2 / s^3` (continuous-time white-acceleration model).
    pub accel_spectral_density_pos: f64,
    /// Acceleration spectral density for the width/height axes, in
    /// `pixels^2 / s^3`.
    pub accel_spectral_density_size: f64,
    /// Initial variance of the center velocities, in `pixels^2 / s^2`.
    pub initial_velocity_variance: f64,
    /// Initial variance of the size velocities, in `pixels^2 / s^2`.
    pub initial_size_velocity_variance: f64,
    /// Minimum IoU for an association to be allowed, in `[0, 1]`.
    pub gate_min_iou: f64,
    /// Positive size floor, in pixels. Predicted boxes smaller than this are
    /// clamped (see [`BBox2DTracker`] documentation).
    pub min_bbox_size: f64,
}

impl Default for BBox2DTrackerConfig {
    fn default() -> Self {
        BBox2DTrackerConfig {
            common: TrackerConfig::default(),
            frame_id: FrameId::new("camera"),
            accel_spectral_density_pos: 80.0,
            accel_spectral_density_size: 20.0,
            initial_velocity_variance: 400.0,
            initial_size_velocity_variance: 25.0,
            gate_min_iou: 0.3,
            min_bbox_size: 1.0,
        }
    }
}

impl BBox2DTrackerConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), InputError> {
        self.common.validate()?;
        if !self.gate_min_iou.is_finite() || !(0.0..=1.0).contains(&self.gate_min_iou) {
            return Err(InputError::InvalidConfig {
                reason: "gate_min_iou must be within [0, 1]",
            });
        }
        if !self.accel_spectral_density_pos.is_finite() || self.accel_spectral_density_pos < 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "accel_spectral_density_pos must be finite and >= 0",
            });
        }
        if !self.accel_spectral_density_size.is_finite() || self.accel_spectral_density_size < 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "accel_spectral_density_size must be finite and >= 0",
            });
        }
        if !self.initial_velocity_variance.is_finite() || self.initial_velocity_variance <= 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "initial_velocity_variance must be finite and > 0",
            });
        }
        if !self.initial_size_velocity_variance.is_finite()
            || self.initial_size_velocity_variance <= 0.0
        {
            return Err(InputError::InvalidConfig {
                reason: "initial_size_velocity_variance must be finite and > 0",
            });
        }
        if !self.min_bbox_size.is_finite() || self.min_bbox_size <= 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "min_bbox_size must be finite and > 0",
            });
        }
        Ok(())
    }
}

/// Snapshot of one 2D track at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct Track2DSnapshot {
    /// Track identifier.
    pub track_id: TrackId,
    /// Lifecycle state.
    pub lifecycle: TrackLifecycleState,
    /// Time the state estimate refers to.
    pub state_timestamp: Timestamp,
    /// Time of the most recent observation.
    pub last_observation_timestamp: Option<Timestamp>,
    /// Sensor that produced the most recent observation.
    pub last_update_sensor: Option<SensorId>,
    /// Distinct sensors that have contributed observations (bounded).
    pub contributing_sensors: Vec<SensorId>,
    /// Class label, if known.
    pub class_label: Option<ClassLabel>,
    /// Whether this track was updated by an observation in the last processed
    /// batch.
    pub updated_in_last_batch: bool,
    /// Number of distinct observation timestamps seen by this track. Used for
    /// confirmation and independent of how many sensors reported.
    pub observation_times_count: u32,
    /// Estimated bounding box in pixels.
    pub bbox: BBox2D,
    /// Estimated center velocity in pixels per second.
    pub velocity: Vector2<f64>,
    /// Full state estimate `[cx, cy, w, h, vcx, vcy, vw, vh]`.
    pub state: SVector<f64, 8>,
    /// Full state covariance.
    pub covariance: SMatrix<f64, 8, 8>,
}

impl Track2DSnapshot {
    /// Returns `true` when the estimate is a prediction not backed by an
    /// observation at [`Track2DSnapshot::state_timestamp`].
    pub fn is_prediction_only(&self) -> bool {
        self.last_observation_timestamp != Some(self.state_timestamp)
    }
}

#[derive(Debug, Clone)]
struct Track2D {
    id: TrackId,
    filter: KalmanFilter<8, 4>,
    lifecycle: TrackLifecycle,
    class_label: Option<ClassLabel>,
    state_timestamp: Timestamp,
    last_update_sensor: Option<SensorId>,
    contributing_sensors: Vec<SensorId>,
    updated_this_batch: bool,
}

/// Multi-object tracker for axis-aligned 2D boxes on one camera.
///
/// # Box sizes under prediction
///
/// Because the constant-velocity model is unbounded, a long prediction can
/// drive `w` or `h` negative. After every prediction the tracker clamps any
/// size at or below [`BBox2DTrackerConfig::min_bbox_size`] to that floor and
/// zeroes the corresponding size velocity. The covariance is left unchanged,
/// which keeps the state consistent with the observations but does not model
/// the clamp. This is documented behaviour, not a silent repair of invalid
/// input: observations with non-positive extents are rejected outright.
#[derive(Debug, Clone)]
pub struct BBox2DTracker {
    config: BBox2DTrackerConfig,
    tracks: Vec<Track2D>,
    next_track_id: u64,
    last_time: Option<Timestamp>,
    last_sequence: BTreeMap<SensorId, u64>,
}

impl BBox2DTracker {
    /// Creates a tracker, validating the configuration.
    pub fn new(config: BBox2DTrackerConfig) -> Result<Self, TrackerError> {
        config.validate()?;
        Ok(BBox2DTracker {
            config,
            tracks: Vec::new(),
            next_track_id: 1,
            last_time: None,
            last_sequence: BTreeMap::new(),
        })
    }

    /// The camera frame this tracker is bound to.
    pub fn frame_id(&self) -> &FrameId {
        &self.config.frame_id
    }

    /// Number of tracks currently alive (including tentative and lost ones).
    pub fn active_track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Current tracker time, if any batch has been processed.
    pub fn current_time(&self) -> Option<Timestamp> {
        self.last_time
    }

    /// Immutable access to the configuration.
    pub fn config(&self) -> &BBox2DTrackerConfig {
        &self.config
    }

    /// Current snapshots of every live track.
    pub fn snapshots(&self) -> Vec<Track2DSnapshot> {
        self.tracks
            .iter()
            .map(|track| self.snapshot_of(track))
            .collect()
    }

    /// Advances every live track to `timestamp` without any observation.
    ///
    /// Time-based lifecycle transitions run first, so tracks that have aged
    /// out are removed before prediction. This bounds the prediction interval
    /// of surviving tracks by the configured time-outs. Calling this with an
    /// earlier timestamp than the current time is an error.
    pub fn predict_to(&mut self, timestamp: Timestamp) -> Result<(), TrackerError> {
        let backup = self.clone();
        match self.predict_to_inner(timestamp) {
            Ok(()) => Ok(()),
            Err(error) => {
                *self = backup;
                Err(error)
            }
        }
    }

    /// Predicts, associates and updates using one sensor batch.
    ///
    /// Validation happens before any state change and the whole operation is
    /// rolled back on any error, so a failed batch never leaves the tracker in
    /// a partially updated state. Returns snapshots after the update.
    pub fn process_batch(
        &mut self,
        batch: &ObservationBatch2D,
    ) -> Result<Vec<Track2DSnapshot>, TrackerError> {
        self.validate_batch(batch)?;
        let backup = self.clone();
        match self.process_batch_inner(batch) {
            Ok(snapshots) => Ok(snapshots),
            Err(error) => {
                *self = backup;
                Err(error)
            }
        }
    }

    fn validate_batch(&self, batch: &ObservationBatch2D) -> Result<(), TrackerError> {
        if batch.frame_id != self.config.frame_id {
            return Err(InputError::FrameMismatch {
                expected: self.config.frame_id.to_string(),
                found: batch.frame_id.to_string(),
            }
            .into());
        }
        check_timestamp_order(self.last_time, batch.timestamp)?;
        if let Some(&last_seen) = self.last_sequence.get(&batch.sensor_id) {
            if batch.sequence <= last_seen {
                return Err(InputError::DuplicateBatch {
                    sensor: batch.sensor_id,
                    received: batch.sequence,
                    last_seen,
                }
                .into());
            }
        }
        for observation in &batch.observations {
            observation.bbox.validate()?;
            validate_score(observation.score)?;
            validate_covariance(&observation.covariance, "bbox2d measurement covariance")?;
        }
        Ok(())
    }

    fn process_batch_inner(
        &mut self,
        batch: &ObservationBatch2D,
    ) -> Result<Vec<Track2DSnapshot>, TrackerError> {
        self.predict_to_inner(batch.timestamp)?;
        for track in &mut self.tracks {
            track.updated_this_batch = false;
        }

        let h = measurement_h4();
        let n_tracks = self.tracks.len();
        let n_observations = batch.observations.len();
        let mut costs = vec![vec![f64::INFINITY; n_observations]; n_tracks];
        for (i, track) in self.tracks.iter().enumerate() {
            let predicted = predicted_bbox(track.filter.state());
            for (j, observation) in batch.observations.iter().enumerate() {
                if !classes_compatible(track.class_label, observation.class_label) {
                    continue;
                }
                if let Some(cost) =
                    iou_association_cost(&predicted, &observation.bbox, self.config.gate_min_iou)
                {
                    costs[i][j] = cost;
                }
            }
        }

        let unmatched_row_cost = 1.0 - self.config.gate_min_iou;
        let assignment = linear_assignment(&costs, n_observations, unmatched_row_cost, 0.0)?;

        for &(i, j) in &assignment.pairs {
            let observation = &batch.observations[j];
            let measurement = observation.measurement();
            self.tracks[i]
                .filter
                .update(&measurement, &h, &observation.covariance)?;
            self.apply_observation(i, observation.class_label, batch, observation.bbox);
            self.tracks[i].updated_this_batch = true;
        }

        for &j in &assignment.unmatched_columns {
            let observation = &batch.observations[j];
            let spawn_allowed = self
                .config
                .common
                .new_track_score_threshold
                .is_none_or(|threshold| observation.score >= threshold);
            if spawn_allowed {
                self.spawn_track(observation, batch.timestamp, batch.sensor_id)?;
            }
        }

        self.last_sequence.insert(batch.sensor_id, batch.sequence);
        self.last_time = Some(batch.timestamp);
        Ok(self.snapshots())
    }

    fn apply_observation(
        &mut self,
        index: usize,
        class_label: Option<ClassLabel>,
        batch: &ObservationBatch2D,
        _bbox: BBox2D,
    ) {
        let track = &mut self.tracks[index];
        track
            .lifecycle
            .register_observation(batch.timestamp, &self.config.common);
        track.state_timestamp = batch.timestamp;
        track.last_update_sensor = Some(batch.sensor_id);
        record_sensor(
            &mut track.contributing_sensors,
            batch.sensor_id,
            self.config.common.max_contributing_sensors,
        );
        if track.class_label.is_none() {
            track.class_label = class_label;
        }
    }

    fn spawn_track(
        &mut self,
        observation: &BBox2DObservation,
        timestamp: Timestamp,
        sensor: SensorId,
    ) -> Result<(), TrackerError> {
        let center = observation.bbox.center();
        let mut state = SVector::<f64, 8>::zeros();
        state[IDX_CX] = center.x;
        state[IDX_CY] = center.y;
        state[IDX_W] = observation.bbox.width();
        state[IDX_H] = observation.bbox.height();

        let mut covariance = SMatrix::<f64, 8, 8>::zeros();
        for r in 0..4 {
            for c in 0..4 {
                covariance[(r, c)] = observation.covariance[(r, c)];
            }
        }
        covariance[(4, 4)] = self.config.initial_velocity_variance;
        covariance[(5, 5)] = self.config.initial_velocity_variance;
        covariance[(6, 6)] = self.config.initial_size_velocity_variance;
        covariance[(7, 7)] = self.config.initial_size_velocity_variance;

        let filter = KalmanFilter::<8, 4>::new(state, covariance)?;
        let mut lifecycle = TrackLifecycle::new(timestamp);
        lifecycle.register_observation(timestamp, &self.config.common);
        let mut contributing_sensors = Vec::new();
        record_sensor(
            &mut contributing_sensors,
            sensor,
            self.config.common.max_contributing_sensors,
        );

        let id = TrackId::new(self.next_track_id);
        self.next_track_id = self.next_track_id.saturating_add(1);
        self.tracks.push(Track2D {
            id,
            filter,
            lifecycle,
            class_label: observation.class_label,
            state_timestamp: timestamp,
            last_update_sensor: Some(sensor),
            contributing_sensors,
            updated_this_batch: true,
        });
        Ok(())
    }

    fn predict_to_inner(&mut self, timestamp: Timestamp) -> Result<(), TrackerError> {
        if let Some(previous) = self.last_time {
            if timestamp < previous {
                return Err(InputError::TimestampWentBackwards {
                    received_ns: timestamp.as_nanos(),
                    current_ns: previous.as_nanos(),
                }
                .into());
            }
        }

        let common = self.config.common.clone();
        self.tracks
            .retain_mut(|track| !track.lifecycle.advance(timestamp, &common));

        let q_pos = self.config.accel_spectral_density_pos;
        let q_size = self.config.accel_spectral_density_size;
        let min_size = self.config.min_bbox_size;
        for track in &mut self.tracks {
            let dt_ns = timestamp.checked_duration_since(track.state_timestamp)?;
            if dt_ns == 0 {
                continue;
            }
            let dt = dt_ns as f64 / 1.0e9;
            let transition = cv_f8(dt);
            let process_noise = cv_q8(dt, q_pos, q_size);
            track.filter.predict(&transition, &process_noise)?;
            clamp_box_sizes(track.filter.state_mut(), min_size);
            track.state_timestamp = timestamp;
        }

        self.last_time = Some(timestamp);
        Ok(())
    }

    fn snapshot_of(&self, track: &Track2D) -> Track2DSnapshot {
        let state = track.filter.state();
        Track2DSnapshot {
            track_id: track.id,
            lifecycle: track.lifecycle.state(),
            state_timestamp: track.state_timestamp,
            last_observation_timestamp: track.lifecycle.last_observation_timestamp(),
            last_update_sensor: track.last_update_sensor,
            contributing_sensors: track.contributing_sensors.clone(),
            class_label: track.class_label,
            updated_in_last_batch: track.updated_this_batch,
            observation_times_count: track.lifecycle.observation_times_count(),
            bbox: predicted_bbox(state),
            velocity: Vector2::new(state[4], state[5]),
            state: *state,
            covariance: *track.filter.covariance(),
        }
    }
}

fn predicted_bbox(state: &SVector<f64, 8>) -> BBox2D {
    BBox2D::from_center_size(state[IDX_CX], state[IDX_CY], state[IDX_W], state[IDX_H])
}

/// Clamps non-positive predicted box extents to `min_size` and stops their
/// velocities from growing further.
fn clamp_box_sizes(state: &mut SVector<f64, 8>, min_size: f64) {
    if state[IDX_W] <= min_size {
        state[IDX_W] = min_size;
        state[6] = 0.0;
    }
    if state[IDX_H] <= min_size {
        state[IDX_H] = min_size;
        state[7] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Matrix4;

    fn config() -> BBox2DTrackerConfig {
        let mut config = BBox2DTrackerConfig::default();
        config.common.confirm_min_observation_times = 1;
        config.common.confirmed_to_lost_ns = 500_000_000;
        config.common.lost_to_delete_ns = 1_000_000_000;
        config.common.tentative_timeout_ns = 1_000_000_000;
        config.gate_min_iou = 0.2;
        config
    }

    fn observation(cx: f64, cy: f64, width: f64, height: f64, score: f64) -> BBox2DObservation {
        BBox2DObservation {
            bbox: BBox2D::from_center_size(cx, cy, width, height),
            score,
            class_label: None,
            covariance: Matrix4::<f64>::identity(),
        }
    }

    fn batch(
        sensor: u32,
        sequence: u64,
        timestamp_ns: i64,
        observations: Vec<BBox2DObservation>,
    ) -> ObservationBatch2D {
        ObservationBatch2D {
            sensor_id: SensorId::new(sensor),
            sequence,
            timestamp: Timestamp::from_nanos(timestamp_ns),
            frame_id: FrameId::new("camera"),
            observations,
        }
    }

    #[test]
    fn empty_batch_is_accepted() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        let snapshots = tracker.process_batch(&batch(1, 0, 0, vec![])).unwrap();
        assert!(snapshots.is_empty());
        assert_eq!(tracker.active_track_count(), 0);
    }

    #[test]
    fn spawns_and_confirms_track() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        let first = tracker
            .process_batch(&batch(
                1,
                0,
                0,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].lifecycle, TrackLifecycleState::Confirmed);
        assert!(first[0].updated_in_last_batch);
    }

    #[test]
    fn constant_velocity_prediction_moves_box() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                0,
                vec![observation(0.0, 0.0, 100.0, 100.0, 0.9)],
            ))
            .unwrap();
        let after_second = tracker
            .process_batch(&batch(
                1,
                1,
                1_000_000_000,
                vec![observation(10.0, 0.0, 100.0, 100.0, 0.9)],
            ))
            .unwrap();
        assert_eq!(after_second.len(), 1);
        let velocity_x = after_second[0].velocity.x;
        assert!(
            velocity_x > 0.0,
            "expected positive x velocity, got {velocity_x}"
        );

        tracker
            .predict_to(Timestamp::from_nanos(2_000_000_000))
            .unwrap();
        let snapshot = &tracker.snapshots()[0];
        assert_eq!(
            snapshot.state_timestamp,
            Timestamp::from_nanos(2_000_000_000)
        );
        assert!(snapshot.is_prediction_only());
        assert!(snapshot.bbox.center().x > 10.0);
    }

    #[test]
    fn timestamp_reversal_is_rejected_without_state_change() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                1_000_000_000,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        let before = tracker.snapshots();
        let error = tracker
            .process_batch(&batch(1, 1, 500_000_000, vec![]))
            .unwrap_err();
        assert!(error.is_input_error());
        assert_eq!(tracker.snapshots(), before);
    }

    #[test]
    fn duplicate_batch_is_rejected_without_state_change() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        let input = batch(1, 7, 0, vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)]);
        tracker.process_batch(&input).unwrap();
        let before = tracker.snapshots();
        let error = tracker.process_batch(&input).unwrap_err();
        assert!(matches!(
            error,
            TrackerError::Input(InputError::DuplicateBatch { .. })
        ));
        assert_eq!(tracker.snapshots(), before);
    }

    #[test]
    fn invalid_bbox_is_rejected_without_state_change() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                0,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        let before = tracker.snapshots();
        let bad = batch(1, 1, 1_000_000, vec![observation(0.0, 0.0, 0.0, 10.0, 0.9)]);
        let error = tracker.process_batch(&bad).unwrap_err();
        assert!(matches!(
            error,
            TrackerError::Input(InputError::InvalidBBox { .. })
        ));
        assert_eq!(tracker.snapshots(), before);
    }

    #[test]
    fn frame_mismatch_is_rejected() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        let mut bad = batch(1, 0, 0, vec![]);
        bad.frame_id = FrameId::new("other-camera");
        let error = tracker.process_batch(&bad).unwrap_err();
        assert!(matches!(
            error,
            TrackerError::Input(InputError::FrameMismatch { .. })
        ));
    }

    #[test]
    fn outlier_observation_spawns_new_track() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                0,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        let snapshots = tracker
            .process_batch(&batch(
                1,
                1,
                1_000_000,
                vec![observation(1000.0, 1000.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].track_id.value(), 1);
        assert_eq!(snapshots[1].track_id.value(), 2);
    }

    #[test]
    fn lost_then_deleted_by_time() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                0,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        tracker
            .predict_to(Timestamp::from_nanos(600_000_000))
            .unwrap();
        assert_eq!(tracker.snapshots()[0].lifecycle, TrackLifecycleState::Lost);
        tracker
            .predict_to(Timestamp::from_nanos(2_000_000_000))
            .unwrap();
        assert_eq!(tracker.active_track_count(), 0);
    }

    #[test]
    fn lost_track_is_regained_and_not_deleted() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                0,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        tracker
            .predict_to(Timestamp::from_nanos(600_000_000))
            .unwrap();
        assert_eq!(tracker.snapshots()[0].lifecycle, TrackLifecycleState::Lost);
        let snapshots = tracker
            .process_batch(&batch(
                1,
                1,
                700_000_000,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        assert_eq!(snapshots[0].lifecycle, TrackLifecycleState::Confirmed);
        assert_eq!(tracker.active_track_count(), 1);
    }

    #[test]
    fn predicted_box_size_is_clamped() {
        let mut config = config();
        config.common.confirmed_to_lost_ns = 2_000_000_000;
        config.common.lost_to_delete_ns = 4_000_000_000;
        config.accel_spectral_density_size = 0.0;
        config.min_bbox_size = 1.0;
        let mut tracker = BBox2DTracker::new(config).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                0,
                vec![observation(0.0, 0.0, 20.0, 20.0, 0.9)],
            ))
            .unwrap();
        tracker
            .process_batch(&batch(
                1,
                1,
                1_000_000_000,
                vec![observation(0.0, 0.0, 5.0, 5.0, 0.9)],
            ))
            .unwrap();
        tracker
            .predict_to(Timestamp::from_nanos(1_500_000_000))
            .unwrap();
        let snapshot = &tracker.snapshots()[0];
        assert!(snapshot.bbox.width() >= 1.0);
        assert!(snapshot.bbox.height() >= 1.0);
        assert_relative_eq!(snapshot.state[6], 0.0, epsilon = 1e-12);
        assert_relative_eq!(snapshot.state[7], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn class_mismatch_prevents_association() {
        let mut config = config();
        config.common.new_track_score_threshold = None;
        let mut tracker = BBox2DTracker::new(config).unwrap();
        let mut first = observation(0.0, 0.0, 10.0, 10.0, 0.9);
        first.class_label = Some(ClassLabel::new(1));
        tracker.process_batch(&batch(1, 0, 0, vec![first])).unwrap();

        let mut second = observation(0.0, 0.0, 10.0, 10.0, 0.9);
        second.class_label = Some(ClassLabel::new(2));
        let snapshots = tracker
            .process_batch(&batch(1, 1, 100_000_000, vec![second]))
            .unwrap();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].track_id.value(), 1);
        assert_eq!(snapshots[1].track_id.value(), 2);
    }

    #[test]
    fn track_ids_are_never_reused() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                0,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        tracker
            .predict_to(Timestamp::from_nanos(3_000_000_000))
            .unwrap();
        assert_eq!(tracker.active_track_count(), 0);
        let snapshots = tracker
            .process_batch(&batch(
                1,
                1,
                4_000_000_000,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        assert_eq!(snapshots[0].track_id.value(), 2);
    }

    #[test]
    fn timestamp_difference_overflow_is_reported() {
        let mut tracker = BBox2DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(
                1,
                0,
                i64::MIN,
                vec![observation(0.0, 0.0, 10.0, 10.0, 0.9)],
            ))
            .unwrap();
        let error = tracker
            .predict_to(Timestamp::from_nanos(i64::MAX))
            .unwrap_err();
        assert!(matches!(
            error,
            TrackerError::Input(InputError::TimestampOverflow)
        ));
    }

    #[test]
    fn non_positive_config_is_rejected() {
        let mut config = config();
        config.gate_min_iou = 1.5;
        assert!(BBox2DTracker::new(config).is_err());
    }
}
