//! 3D position tracker with multi-sensor fusion.
//!
//! # State and units
//!
//! The Kalman state is
//!
//! ```text
//! [ x, y, z, vx, vy, vz ]
//! ```
//!
//! with position in **meters** and velocity in **meters per second**.
//! Measurements are positions `[x, y, z]` in meters with a `3x3` covariance in
//! `meters^2`.
//!
//! # Scope
//!
//! This is **position tracking**, not full 3D box tracking: orientation and
//! extent are not estimated because there is no model for them here. All
//! positions are expected to already be expressed in one common tracking
//! frame; batches from other frames are rejected rather than transformed
//! implicitly. Use [`crate::transform::RigidTransform3D`] to bring sensor
//! measurements into the tracking frame before submitting them.
//!
//! Several sensors can update the same track. Each batch is fused
//! sequentially and independently; observations are assumed to have
//! independent noise and correlated observations / track-to-track fusion are
//! out of scope.

use std::collections::BTreeMap;

use nalgebra::{SMatrix, SVector, Vector3};

use crate::assignment::{linear_assignment, Assignment};
use crate::association::{classes_compatible, mahalanobis_association_cost};
use crate::error::{check_timestamp_order, InputError, TrackerError};
use crate::filter::KalmanFilter;
use crate::lifecycle::TrackLifecycle;
use crate::math::linalg::validate_covariance;
use crate::math::models::{cv_f6, cv_q6, measurement_h3};
use crate::tracker::record_sensor;
use crate::types::{
    ClassLabel, ClassPrior, FrameId, ObservationBatch3D, Position3DObservation, SensorId,
    Timestamp, TrackId, TrackLifecycleState, TrackerConfig,
};

/// Default squared-Mahalanobis gate for a 3D position.
///
/// This is the `0.999` quantile of a chi-square distribution with 3 degrees of
/// freedom (the measurement dimension).
pub const DEFAULT_GATE_SQUARED_MAHALANOBIS: f64 = 16.266;

/// Configuration for [`Cartesian3DTracker`].
#[derive(Debug, Clone, PartialEq)]
pub struct Cartesian3DTrackerConfig {
    /// Shared lifecycle parameters.
    pub common: TrackerConfig,
    /// The single tracking frame this tracker accepts.
    pub frame_id: FrameId,
    /// Acceleration spectral density for the continuous-time white-acceleration
    /// model, in `meters^2 / s^3`.
    pub accel_spectral_density: f64,
    /// Initial velocity variance in `meters^2 / s^2`.
    pub initial_velocity_variance: f64,
    /// Gating threshold on the **squared** Mahalanobis distance of a 3D
    /// position (measurement dimension 3). Values above this are never
    /// associated.
    pub gate_squared_mahalanobis: f64,
    /// Maximum soft cost added when an observation's known class contradicts a
    /// track's [`ClassPrior`]. The added cost is
    /// `class_prior_penalty * prior.confidence`, so a weak prior barely shifts
    /// the association and a strong prior can outweigh a small spatial
    /// difference. Set to `0.0` to disable class-prior influence entirely.
    ///
    /// The prior is never a hard gate; the spatial gate always applies.
    pub class_prior_penalty: f64,
    /// Two-stage association (a matching cascade). When set, tracks observed
    /// within this window before the batch are associated first and the
    /// remaining tracks only compete for the observations left over.
    ///
    /// Without it, a coasting track whose covariance has grown has a smaller
    /// squared Mahalanobis distance than a freshly updated track at the same
    /// place, so two tracks on one object can take the observation from each
    /// other in turn. `None` keeps a single assignment over all tracks.
    pub recent_track_window_ns: Option<i64>,
    /// Weight of the appearance term added to the association cost:
    /// `appearance_weight * (1 - cos(track, observation))`, applied when both
    /// the track and the observation carry an embedding. `0.0` disables it.
    /// The spatial gate always applies first.
    pub appearance_weight: f64,
    /// Exponential moving average factor for a track's embedding: the new
    /// embedding is `momentum * old + (1 - momentum) * observed`.
    pub appearance_momentum: f64,
    /// Optional hard gate on the cosine distance `1 - cos`; pairs above it are
    /// never associated (only when both embeddings exist).
    pub appearance_max_distance: Option<f64>,
    /// When set, an unassociated observation closer than this (metres, 3D)
    /// to a confirmed track does not start a new track. Detectors often
    /// report a second box next to a person; such boxes otherwise become
    /// tracks that later take the person's own detections. `None` disables it.
    pub min_birth_distance_m: Option<f64>,
    /// With `min_birth_distance_m`: a suppressed observation that keeps
    /// reappearing at the same place is a real neighbour rather than a second
    /// box on the person. After this many sightings (each within
    /// `birth_candidate_radius_m` of the previous one and at most
    /// `birth_candidate_timeout_ns` after it) a track is started. `None`
    /// suppresses such births for as long as the confirmed track is near.
    pub birth_persistence: Option<u32>,
    /// Radius (m) within which a suppressed observation continues a candidate.
    pub birth_candidate_radius_m: f64,
    /// A candidate not seen again within this time (ns) is forgotten.
    pub birth_candidate_timeout_ns: i64,
}

impl Default for Cartesian3DTrackerConfig {
    fn default() -> Self {
        Cartesian3DTrackerConfig {
            common: TrackerConfig::default(),
            frame_id: FrameId::new("world"),
            accel_spectral_density: 1.0,
            initial_velocity_variance: 4.0,
            gate_squared_mahalanobis: DEFAULT_GATE_SQUARED_MAHALANOBIS,
            class_prior_penalty: 8.0,
            recent_track_window_ns: None,
            appearance_weight: 0.0,
            appearance_momentum: 0.9,
            appearance_max_distance: None,
            min_birth_distance_m: None,
            birth_persistence: None,
            birth_candidate_radius_m: 0.5,
            birth_candidate_timeout_ns: 300_000_000,
        }
    }
}

impl Cartesian3DTrackerConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), InputError> {
        self.common.validate()?;
        if !self.accel_spectral_density.is_finite() || self.accel_spectral_density < 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "accel_spectral_density must be finite and >= 0",
            });
        }
        if !self.initial_velocity_variance.is_finite() || self.initial_velocity_variance <= 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "initial_velocity_variance must be finite and > 0",
            });
        }
        if !self.gate_squared_mahalanobis.is_finite() || self.gate_squared_mahalanobis <= 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "gate_squared_mahalanobis must be finite and > 0",
            });
        }
        if !self.class_prior_penalty.is_finite() || self.class_prior_penalty < 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "class_prior_penalty must be finite and >= 0",
            });
        }
        if !self.appearance_weight.is_finite() || self.appearance_weight < 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "appearance_weight must be finite and >= 0",
            });
        }
        if !(0.0..1.0).contains(&self.appearance_momentum) {
            return Err(InputError::InvalidConfig {
                reason: "appearance_momentum must be in [0, 1)",
            });
        }
        if matches!(self.appearance_max_distance, Some(d) if !d.is_finite() || d <= 0.0) {
            return Err(InputError::InvalidConfig {
                reason: "appearance_max_distance must be finite and > 0",
            });
        }
        if matches!(self.min_birth_distance_m, Some(d) if !d.is_finite() || d <= 0.0) {
            return Err(InputError::InvalidConfig {
                reason: "min_birth_distance_m must be finite and > 0",
            });
        }
        if matches!(self.birth_persistence, Some(n) if n < 2) {
            return Err(InputError::InvalidConfig {
                reason: "birth_persistence must be >= 2",
            });
        }
        if !self.birth_candidate_radius_m.is_finite() || self.birth_candidate_radius_m <= 0.0 {
            return Err(InputError::InvalidConfig {
                reason: "birth_candidate_radius_m must be finite and > 0",
            });
        }
        if self.birth_candidate_timeout_ns <= 0 {
            return Err(InputError::InvalidConfig {
                reason: "birth_candidate_timeout_ns must be > 0",
            });
        }
        if matches!(self.recent_track_window_ns, Some(window) if window <= 0) {
            return Err(InputError::InvalidConfig {
                reason: "recent_track_window_ns must be > 0",
            });
        }
        Ok(())
    }
}

/// Snapshot of one 3D track at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct Track3DSnapshot {
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
    /// Class label derived from positional observations, if known.
    pub class_label: Option<ClassLabel>,
    /// Soft class prior supplied by an auxiliary source (for example camera
    /// evidence). Never a hard gate.
    pub class_prior: Option<ClassPrior>,
    /// Whether this track was updated by an observation in the last processed
    /// batch.
    pub updated_in_last_batch: bool,
    /// Number of distinct observation timestamps seen by this track. Used for
    /// confirmation and independent of how many sensors reported.
    pub observation_times_count: u32,
    /// Estimated position in meters.
    pub position: Vector3<f64>,
    /// Estimated velocity in meters per second.
    pub velocity: Vector3<f64>,
    /// Full state estimate `[x, y, z, vx, vy, vz]`.
    pub state: SVector<f64, 6>,
    /// Full state covariance.
    pub covariance: SMatrix<f64, 6, 6>,
}

impl Track3DSnapshot {
    /// Returns `true` when the estimate is a prediction not backed by an
    /// observation at [`Track3DSnapshot::state_timestamp`].
    pub fn is_prediction_only(&self) -> bool {
        self.last_observation_timestamp != Some(self.state_timestamp)
    }
}

/// Outcome of associating one observation of a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationOutcome {
    /// Matched an existing track and updated it.
    Updated,
    /// Did not match and spawned a new track.
    Spawned,
    /// Did not match and did not spawn a track.
    Unmatched,
}

/// Association diagnostic for one observation of a batch.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationAssociation {
    /// Index of the observation within its batch.
    pub observation_index: usize,
    /// Track it updated or spawned, if any.
    pub track_id: Option<TrackId>,
    /// What happened to the observation.
    pub outcome: ObservationOutcome,
    /// Assignment cost used for this observation, when it was matched. For
    /// position tracking this is the squared Mahalanobis distance plus any soft
    /// class-prior penalty.
    pub cost: Option<f64>,
}

/// Diagnostic report for one processed 3D batch.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchReport3D {
    /// Batch timestamp.
    pub timestamp: Timestamp,
    /// Sensor that produced the batch.
    pub sensor_id: SensorId,
    /// One entry per observation, in batch order.
    pub associations: Vec<ObservationAssociation>,
    /// Tracks that were not matched by any observation in this batch.
    pub unmatched_track_ids: Vec<TrackId>,
}

#[derive(Debug, Clone)]
struct Track3D {
    id: TrackId,
    filter: KalmanFilter<6, 3>,
    lifecycle: TrackLifecycle,
    class_label: Option<ClassLabel>,
    class_prior: Option<ClassPrior>,
    state_timestamp: Timestamp,
    last_update_sensor: Option<SensorId>,
    contributing_sensors: Vec<SensorId>,
    updated_this_batch: bool,
    /// Unit-norm appearance embedding (EMA of associated observations).
    appearance: Option<Vec<f32>>,
}

/// A suppressed birth that may turn out to be a real neighbour.
#[derive(Debug, Clone)]
struct BirthCandidate {
    position: Vector3<f64>,
    last_seen: Timestamp,
    sightings: u32,
}

/// Unit-norm copy of an embedding; `None` for missing, empty, non-finite or
/// zero vectors.
fn normalized(vector: Option<&[f32]>) -> Option<Vec<f32>> {
    let vector = vector?;
    if vector.is_empty() || !vector.iter().all(|v| v.is_finite()) {
        return None;
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    (norm > 0.0).then(|| vector.iter().map(|v| v / norm).collect())
}

/// Cosine distance `1 - cos` between a unit-norm track embedding and an
/// observation embedding; `None` when either is missing or the sizes differ.
fn cosine_distance(track: Option<&[f32]>, observation: Option<&[f32]>) -> Option<f64> {
    let (track, observation) = (track?, normalized(observation)?);
    if track.len() != observation.len() {
        return None;
    }
    let cos: f32 = track.iter().zip(&observation).map(|(a, b)| a * b).sum();
    Some(f64::from(1.0 - cos.clamp(-1.0, 1.0)))
}

/// Assigns the rows flagged `first` before the others: stage one solves
/// `first` rows against all columns, stage two the remaining rows against the
/// columns stage one left unmatched.
fn cascade_assignment(
    costs: &[Vec<f64>],
    num_columns: usize,
    first: &[bool],
    unmatched_row_cost: f64,
) -> Result<Assignment, crate::error::NumericError> {
    let (stage_one_rows, stage_two_rows): (Vec<usize>, Vec<usize>) =
        (0..costs.len()).partition(|&i| first[i]);
    let stage_one_costs: Vec<Vec<f64>> = stage_one_rows.iter().map(|&i| costs[i].clone()).collect();
    let stage_one = linear_assignment(&stage_one_costs, num_columns, unmatched_row_cost, 0.0)?;

    let columns = stage_one.unmatched_columns.clone();
    let stage_two_costs: Vec<Vec<f64>> = stage_two_rows
        .iter()
        .map(|&i| columns.iter().map(|&j| costs[i][j]).collect())
        .collect();
    let stage_two = linear_assignment(&stage_two_costs, columns.len(), unmatched_row_cost, 0.0)?;

    let mut pairs: Vec<(usize, usize)> = stage_one
        .pairs
        .iter()
        .map(|&(r, c)| (stage_one_rows[r], c))
        .chain(
            stage_two
                .pairs
                .iter()
                .map(|&(r, c)| (stage_two_rows[r], columns[c])),
        )
        .collect();
    pairs.sort_unstable();
    let mut unmatched_rows: Vec<usize> = stage_one
        .unmatched_rows
        .iter()
        .map(|&r| stage_one_rows[r])
        .chain(stage_two.unmatched_rows.iter().map(|&r| stage_two_rows[r]))
        .collect();
    unmatched_rows.sort_unstable();
    let mut unmatched_columns: Vec<usize> = stage_two
        .unmatched_columns
        .iter()
        .map(|&c| columns[c])
        .collect();
    unmatched_columns.sort_unstable();
    Ok(Assignment {
        pairs,
        unmatched_rows,
        unmatched_columns,
    })
}

/// Constant-velocity tracker for 3D positions in one common frame.
///
/// Several sensors may update the same track; see the module documentation for
/// the fusion assumptions.
#[derive(Debug, Clone)]
pub struct Cartesian3DTracker {
    config: Cartesian3DTrackerConfig,
    tracks: Vec<Track3D>,
    next_track_id: u64,
    last_time: Option<Timestamp>,
    last_sequence: BTreeMap<SensorId, u64>,
    birth_candidates: Vec<BirthCandidate>,
}

impl Cartesian3DTracker {
    /// Creates a tracker, validating the configuration.
    pub fn new(config: Cartesian3DTrackerConfig) -> Result<Self, TrackerError> {
        config.validate()?;
        Ok(Cartesian3DTracker {
            config,
            tracks: Vec::new(),
            next_track_id: 1,
            last_time: None,
            last_sequence: BTreeMap::new(),
            birth_candidates: Vec::new(),
        })
    }

    /// The tracking frame this tracker is bound to.
    pub fn frame_id(&self) -> &FrameId {
        &self.config.frame_id
    }

    /// Number of tracks currently alive.
    pub fn active_track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Current tracker time, if any batch has been processed.
    pub fn current_time(&self) -> Option<Timestamp> {
        self.last_time
    }

    /// Immutable access to the configuration.
    pub fn config(&self) -> &Cartesian3DTrackerConfig {
        &self.config
    }

    /// Current snapshots of every live track.
    pub fn snapshots(&self) -> Vec<Track3DSnapshot> {
        self.tracks
            .iter()
            .map(|track| self.snapshot_of(track))
            .collect()
    }

    /// Advances every live track to `timestamp` without any observation.
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
    /// Each batch is fused independently, so sensors running at different
    /// rates (or at the same instant) update tracks sequentially. Invalid
    /// input is rejected before any state change and the operation rolls back
    /// on error.
    pub fn process_batch(
        &mut self,
        batch: &ObservationBatch3D,
    ) -> Result<Vec<Track3DSnapshot>, TrackerError> {
        self.validate_batch(batch)?;
        self.run_transactional(|tracker| {
            tracker
                .process_batch_inner(batch)
                .map(|(snapshots, _report)| snapshots)
        })
    }

    /// Like [`Cartesian3DTracker::process_batch`] but also returns a
    /// [`BatchReport3D`] describing which observation matched which track.
    ///
    /// The report is intended for the engine/SDK layer, which needs to attach
    /// per-detection metadata (for example a 3D box shape) to the tracks that
    /// were updated or created.
    pub fn process_batch_reported(
        &mut self,
        batch: &ObservationBatch3D,
    ) -> Result<(Vec<Track3DSnapshot>, BatchReport3D), TrackerError> {
        self.validate_batch(batch)?;
        self.run_transactional(|tracker| tracker.process_batch_inner(batch))
    }

    /// Runs `operation` against a clone of the tracker and restores the clone
    /// if it returns an error, so a failed batch never leaves partial state.
    fn run_transactional<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<T, TrackerError>,
    ) -> Result<T, TrackerError> {
        let backup = self.clone();
        match operation(self) {
            Ok(value) => Ok(value),
            Err(error) => {
                *self = backup;
                Err(error)
            }
        }
    }

    /// Sets or replaces a track's soft class prior.
    ///
    /// Returns `false` (and changes nothing) when no such track exists or when
    /// `confidence` is not finite. `confidence` is clamped to `[0, 1]`.
    pub fn set_class_prior(
        &mut self,
        track_id: TrackId,
        label: ClassLabel,
        confidence: f64,
    ) -> bool {
        if !confidence.is_finite() {
            return false;
        }
        let confidence = confidence.clamp(0.0, 1.0);
        if let Some(track) = self.tracks.iter_mut().find(|track| track.id == track_id) {
            track.class_prior = Some(ClassPrior { label, confidence });
            true
        } else {
            false
        }
    }

    /// Removes a track's soft class prior. Returns `false` if there is no such
    /// track.
    pub fn clear_class_prior(&mut self, track_id: TrackId) -> bool {
        if let Some(track) = self.tracks.iter_mut().find(|track| track.id == track_id) {
            track.class_prior = None;
            true
        } else {
            false
        }
    }

    /// Returns a track's current soft class prior.
    pub fn class_prior(&self, track_id: TrackId) -> Option<ClassPrior> {
        self.tracks
            .iter()
            .find(|track| track.id == track_id)
            .and_then(|track| track.class_prior)
    }

    fn validate_batch(&self, batch: &ObservationBatch3D) -> Result<(), TrackerError> {
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
            if !observation.position.iter().all(|x| x.is_finite()) {
                return Err(InputError::NonFiniteValue {
                    context: "position3d position",
                }
                .into());
            }
            validate_covariance(&observation.covariance, "position3d measurement covariance")?;
        }
        Ok(())
    }

    fn process_batch_inner(
        &mut self,
        batch: &ObservationBatch3D,
    ) -> Result<(Vec<Track3DSnapshot>, BatchReport3D), TrackerError> {
        self.predict_to_inner(batch.timestamp)?;
        for track in &mut self.tracks {
            track.updated_this_batch = false;
        }

        let h = measurement_h3();
        let n_tracks = self.tracks.len();
        let n_observations = batch.observations.len();
        let mut costs = vec![vec![f64::INFINITY; n_observations]; n_tracks];
        for (i, track) in self.tracks.iter().enumerate() {
            for (j, observation) in batch.observations.iter().enumerate() {
                if !classes_compatible(track.class_label, observation.class_label) {
                    continue;
                }
                let squared_distance = track.filter.squared_mahalanobis(
                    &observation.position,
                    &h,
                    &observation.covariance,
                )?;
                let appearance_distance = if self.config.appearance_weight > 0.0 {
                    cosine_distance(
                        track.appearance.as_deref(),
                        observation.appearance.as_deref(),
                    )
                } else {
                    None
                };
                if matches!(
                    (appearance_distance, self.config.appearance_max_distance),
                    (Some(d), Some(max)) if d > max
                ) {
                    continue;
                }
                if let Some(mut cost) = mahalanobis_association_cost(
                    squared_distance,
                    self.config.gate_squared_mahalanobis,
                ) {
                    if let Some(d) = appearance_distance {
                        cost += self.config.appearance_weight * d;
                    }
                    // Soft class prior (e.g. camera evidence): only penalize,
                    // never forbid. The spatial gate above has already applied.
                    if let (Some(prior), Some(observed)) =
                        (track.class_prior, observation.class_label)
                    {
                        if prior.label != observed {
                            cost += self.config.class_prior_penalty * prior.confidence;
                        }
                    }
                    costs[i][j] = cost;
                }
            }
        }

        let assignment = match self.config.recent_track_window_ns {
            None => linear_assignment(
                &costs,
                n_observations,
                self.config.gate_squared_mahalanobis,
                0.0,
            )?,
            Some(window) => {
                let recent: Vec<bool> = self
                    .tracks
                    .iter()
                    .map(|track| {
                        track
                            .lifecycle
                            .last_observation_timestamp()
                            .is_some_and(|last| {
                                batch.timestamp.as_nanos() - last.as_nanos() <= window
                            })
                    })
                    .collect();
                cascade_assignment(
                    &costs,
                    n_observations,
                    &recent,
                    self.config.gate_squared_mahalanobis,
                )?
            }
        };

        let mut associations: Vec<ObservationAssociation> = (0..n_observations)
            .map(|observation_index| ObservationAssociation {
                observation_index,
                track_id: None,
                outcome: ObservationOutcome::Unmatched,
                cost: None,
            })
            .collect();

        for &(i, j) in &assignment.pairs {
            let observation = &batch.observations[j];
            self.tracks[i]
                .filter
                .update(&observation.position, &h, &observation.covariance)?;
            self.apply_observation(i, observation.class_label, batch);
            if let Some(observed) = normalized(observation.appearance.as_deref()) {
                let momentum = self.config.appearance_momentum as f32;
                let track = &mut self.tracks[i];
                track.appearance = match track.appearance.take() {
                    Some(old) if old.len() == observed.len() => normalized(Some(
                        &old.iter()
                            .zip(&observed)
                            .map(|(a, b)| momentum * a + (1.0 - momentum) * b)
                            .collect::<Vec<f32>>(),
                    ))
                    .or(Some(observed)),
                    _ => Some(observed),
                };
            }
            self.tracks[i].updated_this_batch = true;
            associations[j] = ObservationAssociation {
                observation_index: j,
                track_id: Some(self.tracks[i].id),
                outcome: ObservationOutcome::Updated,
                cost: Some(costs[i][j]),
            };
        }

        let confirmed_positions: Vec<Vector3<f64>> = match self.config.min_birth_distance_m {
            Some(_) => self
                .tracks
                .iter()
                .filter(|track| track.lifecycle.state() == TrackLifecycleState::Confirmed)
                .map(|track| track.filter.state().fixed_rows::<3>(0).into_owned())
                .collect(),
            None => Vec::new(),
        };
        let now = batch.timestamp.as_nanos();
        let timeout = self.config.birth_candidate_timeout_ns;
        self.birth_candidates
            .retain(|c| now - c.last_seen.as_nanos() <= timeout);
        for &j in &assignment.unmatched_columns {
            // 3D position observations carry no detection score in this
            // version, so `new_track_score_threshold` does not apply here.
            let observation = &batch.observations[j];
            let mut sightings = 1;
            if let Some(min_distance) = self.config.min_birth_distance_m {
                if confirmed_positions
                    .iter()
                    .any(|p| (p - observation.position).norm() < min_distance)
                {
                    match self.persistent_birth(observation.position, batch.timestamp) {
                        Some(count) => sightings = count,
                        None => continue,
                    }
                }
            }
            let track_id = self.spawn_track(observation, batch.timestamp, batch.sensor_id)?;
            if sightings > 1 {
                // The candidate was seen at `sightings` distinct times already.
                let track = self.tracks.last_mut().expect("just spawned");
                track
                    .lifecycle
                    .credit_observation_times(sightings, &self.config.common);
            }
            associations[j] = ObservationAssociation {
                observation_index: j,
                track_id: Some(track_id),
                outcome: ObservationOutcome::Spawned,
                cost: None,
            };
        }

        let unmatched_track_ids = assignment
            .unmatched_rows
            .iter()
            .map(|&i| self.tracks[i].id)
            .collect();

        self.last_sequence.insert(batch.sensor_id, batch.sequence);
        self.last_time = Some(batch.timestamp);
        let report = BatchReport3D {
            timestamp: batch.timestamp,
            sensor_id: batch.sensor_id,
            associations,
            unmatched_track_ids,
        };
        Ok((self.snapshots(), report))
    }

    /// Records a suppressed birth; returns its number of sightings once it has
    /// been seen often enough to start a track. A candidate is continued at most once
    /// per batch, so two boxes in one batch never count as two sightings.
    fn persistent_birth(&mut self, position: Vector3<f64>, timestamp: Timestamp) -> Option<u32> {
        let required = self.config.birth_persistence?;
        let radius = self.config.birth_candidate_radius_m;
        let nearest = self
            .birth_candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.last_seen != timestamp)
            .map(|(k, c)| (k, (c.position - position).norm()))
            .filter(|&(_, d)| d < radius)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        match nearest {
            Some((k, _)) => {
                let candidate = &mut self.birth_candidates[k];
                candidate.position = position;
                candidate.last_seen = timestamp;
                candidate.sightings += 1;
                if candidate.sightings >= required {
                    return Some(self.birth_candidates.swap_remove(k).sightings);
                }
                None
            }
            None => {
                self.birth_candidates.push(BirthCandidate {
                    position,
                    last_seen: timestamp,
                    sightings: 1,
                });
                None
            }
        }
    }

    fn apply_observation(
        &mut self,
        index: usize,
        class_label: Option<ClassLabel>,
        batch: &ObservationBatch3D,
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
        observation: &Position3DObservation,
        timestamp: Timestamp,
        sensor: SensorId,
    ) -> Result<TrackId, TrackerError> {
        let mut state = SVector::<f64, 6>::zeros();
        state[0] = observation.position.x;
        state[1] = observation.position.y;
        state[2] = observation.position.z;

        let mut covariance = SMatrix::<f64, 6, 6>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                covariance[(r, c)] = observation.covariance[(r, c)];
            }
        }
        covariance[(3, 3)] = self.config.initial_velocity_variance;
        covariance[(4, 4)] = self.config.initial_velocity_variance;
        covariance[(5, 5)] = self.config.initial_velocity_variance;

        let filter = KalmanFilter::<6, 3>::new(state, covariance)?;
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
        self.tracks.push(Track3D {
            id,
            filter,
            lifecycle,
            class_label: observation.class_label,
            class_prior: None,
            state_timestamp: timestamp,
            last_update_sensor: Some(sensor),
            contributing_sensors,
            updated_this_batch: true,
            appearance: normalized(observation.appearance.as_deref()),
        });
        Ok(id)
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

        let q = self.config.accel_spectral_density;
        for track in &mut self.tracks {
            let dt_ns = timestamp.checked_duration_since(track.state_timestamp)?;
            if dt_ns == 0 {
                continue;
            }
            let dt = dt_ns as f64 / 1.0e9;
            track.filter.predict(&cv_f6(dt), &cv_q6(dt, q))?;
            track.state_timestamp = timestamp;
        }

        self.last_time = Some(timestamp);
        Ok(())
    }

    fn snapshot_of(&self, track: &Track3D) -> Track3DSnapshot {
        let state = track.filter.state();
        Track3DSnapshot {
            track_id: track.id,
            lifecycle: track.lifecycle.state(),
            state_timestamp: track.state_timestamp,
            last_observation_timestamp: track.lifecycle.last_observation_timestamp(),
            last_update_sensor: track.last_update_sensor,
            contributing_sensors: track.contributing_sensors.clone(),
            class_label: track.class_label,
            class_prior: track.class_prior,
            updated_in_last_batch: track.updated_this_batch,
            observation_times_count: track.lifecycle.observation_times_count(),
            position: Vector3::new(state[0], state[1], state[2]),
            velocity: Vector3::new(state[3], state[4], state[5]),
            state: *state,
            covariance: *track.filter.covariance(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_gives_recent_tracks_priority() {
        // One observation, two tracks on it: row 1 (a coasting track whose
        // covariance has grown) is cheaper, but row 0 was updated recently.
        let costs = vec![vec![4.0], vec![1.0]];
        let plain = linear_assignment(&costs, 1, 16.0, 0.0).unwrap();
        assert_eq!(plain.pairs, vec![(1, 0)]);
        let cascade = cascade_assignment(&costs, 1, &[true, false], 16.0).unwrap();
        assert_eq!(cascade.pairs, vec![(0, 0)]);
        assert_eq!(cascade.unmatched_rows, vec![1]);
        assert!(cascade.unmatched_columns.is_empty());
        // No recent tracks: identical to a single assignment.
        let all_old = cascade_assignment(&costs, 1, &[false, false], 16.0).unwrap();
        assert_eq!(all_old.pairs, plain.pairs);
        assert_eq!(all_old.unmatched_rows, plain.unmatched_rows);
        assert_eq!(all_old.unmatched_columns, plain.unmatched_columns);
    }
    use approx::assert_relative_eq;
    use nalgebra::Matrix3;

    fn config() -> Cartesian3DTrackerConfig {
        let mut config = Cartesian3DTrackerConfig::default();
        config.common.confirm_min_observation_times = 1;
        config.common.confirmed_to_lost_ns = 500_000_000;
        config.common.lost_to_delete_ns = 1_000_000_000;
        config.common.tentative_timeout_ns = 1_000_000_000;
        config.gate_squared_mahalanobis = 100.0;
        config
    }

    fn observation(x: f64, y: f64, z: f64) -> Position3DObservation {
        Position3DObservation {
            position: Vector3::new(x, y, z),
            covariance: Matrix3::identity(),
            class_label: None,
            appearance: None,
        }
    }

    fn with_appearance(
        mut observation: Position3DObservation,
        appearance: &[f32],
    ) -> Position3DObservation {
        observation.appearance = Some(appearance.to_vec());
        observation
    }

    #[test]
    fn appearance_breaks_a_spatial_tie_between_neighbours() {
        let run = |weight: f64| {
            let mut config = config();
            config.appearance_weight = weight;
            let mut tracker = Cartesian3DTracker::new(config).unwrap();
            // Two neighbours with different appearance.
            tracker
                .process_batch(&batch(
                    1,
                    0,
                    0,
                    vec![
                        with_appearance(observation(0.0, 0.0, 0.0), &[1.0, 0.0]),
                        with_appearance(observation(0.6, 0.0, 0.0), &[0.0, 2.0]),
                    ],
                ))
                .unwrap();
            // One observation, spatially closer to the first track but looking
            // like the second.
            let (_, report) = tracker
                .process_batch_reported(&batch(
                    1,
                    1,
                    100_000_000,
                    vec![with_appearance(observation(0.2, 0.0, 0.0), &[0.0, 1.0])],
                ))
                .unwrap();
            report.associations[0].track_id
        };
        assert_eq!(run(0.0), Some(TrackId::new(1)));
        assert_eq!(run(10.0), Some(TrackId::new(2)));
    }

    #[test]
    fn no_new_track_next_to_a_confirmed_one() {
        let mut config = config();
        config.min_birth_distance_m = Some(1.0);
        let mut tracker = Cartesian3DTracker::new(config).unwrap();
        // One confirmed track (confirm_min_observation_times = 1).
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        let (_, report) = tracker
            .process_batch_reported(&batch(
                1,
                1,
                100_000_000,
                vec![
                    observation(0.05, 0.0, 0.0), // updates the track
                    observation(0.6, 0.0, 0.0),  // a second box next to it: no new track
                    observation(5.0, 0.0, 0.0),  // far away: new track
                ],
            ))
            .unwrap();
        let outcomes: Vec<ObservationOutcome> =
            report.associations.iter().map(|a| a.outcome).collect();
        assert_eq!(
            outcomes,
            vec![
                ObservationOutcome::Updated,
                ObservationOutcome::Unmatched,
                ObservationOutcome::Spawned
            ]
        );
    }

    #[test]
    fn persistent_neighbour_gets_a_track() {
        let mut config = config();
        config.min_birth_distance_m = Some(1.0);
        config.birth_persistence = Some(3);
        let mut tracker = Cartesian3DTracker::new(config).unwrap();
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        let mut outcomes = Vec::new();
        for k in 1..=4u64 {
            let (_, report) = tracker
                .process_batch_reported(&batch(
                    1,
                    k,
                    k as i64 * 100_000_000,
                    vec![observation(0.0, 0.0, 0.0), observation(0.6, 0.0, 0.0)],
                ))
                .unwrap();
            outcomes.push(report.associations[1].outcome);
        }
        // Suppressed twice, started on the third sighting, then updated.
        assert_eq!(
            outcomes,
            vec![
                ObservationOutcome::Unmatched,
                ObservationOutcome::Unmatched,
                ObservationOutcome::Spawned,
                ObservationOutcome::Updated
            ]
        );
    }

    #[test]
    fn persistent_birth_counts_its_earlier_sightings() {
        let mut config = config();
        config.common.confirm_min_observation_times = 3;
        config.min_birth_distance_m = Some(1.0);
        config.birth_persistence = Some(3);
        let mut tracker = Cartesian3DTracker::new(config).unwrap();
        for k in 0..3u64 {
            tracker
                .process_batch(&batch(
                    1,
                    k,
                    k as i64 * 100_000_000,
                    vec![observation(0.0, 0.0, 0.0)],
                ))
                .unwrap();
        }
        let mut snapshots = Vec::new();
        for k in 3..6u64 {
            snapshots = tracker
                .process_batch(&batch(
                    1,
                    k,
                    k as i64 * 100_000_000,
                    vec![observation(0.0, 0.0, 0.0), observation(0.6, 0.0, 0.0)],
                ))
                .unwrap();
        }
        // The neighbour's track starts on its third sighting already confirmed.
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[1].lifecycle, TrackLifecycleState::Confirmed);
        assert_eq!(snapshots[1].observation_times_count, 3);
    }

    #[test]
    fn intermittent_second_box_stays_suppressed() {
        let mut config = config();
        config.min_birth_distance_m = Some(1.0);
        config.birth_persistence = Some(2);
        config.birth_candidate_timeout_ns = 150_000_000;
        let mut tracker = Cartesian3DTracker::new(config).unwrap();
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        for k in 1..=6u64 {
            let mut observations = vec![observation(0.0, 0.0, 0.0)];
            // The second box appears every other batch only (200 ms apart).
            if k % 2 == 1 {
                observations.push(observation(0.6, 0.0, 0.0));
            }
            let (_, report) = tracker
                .process_batch_reported(&batch(1, k, k as i64 * 100_000_000, observations))
                .unwrap();
            assert!(report
                .associations
                .iter()
                .all(|a| a.outcome != ObservationOutcome::Spawned));
        }
    }

    #[test]
    fn appearance_is_ignored_when_missing_or_mismatched() {
        assert_eq!(cosine_distance(None, Some(&[1.0])), None);
        assert_eq!(cosine_distance(Some(&[1.0, 0.0]), Some(&[1.0])), None);
        assert_eq!(cosine_distance(Some(&[1.0, 0.0]), Some(&[0.0, 0.0])), None);
        assert!((cosine_distance(Some(&[1.0, 0.0]), Some(&[3.0, 0.0])).unwrap()).abs() < 1e-6);
        assert!(
            (cosine_distance(Some(&[1.0, 0.0]), Some(&[0.0, -1.0])).unwrap() - 1.0).abs() < 1e-6
        );
    }

    fn batch(
        sensor: u32,
        sequence: u64,
        timestamp_ns: i64,
        observations: Vec<Position3DObservation>,
    ) -> ObservationBatch3D {
        ObservationBatch3D {
            sensor_id: SensorId::new(sensor),
            sequence,
            timestamp: Timestamp::from_nanos(timestamp_ns),
            frame_id: FrameId::new("world"),
            observations,
        }
    }

    #[test]
    fn two_sensors_fuse_into_one_track() {
        let mut tracker = Cartesian3DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        let snapshots = tracker
            .process_batch(&batch(2, 0, 100_000_000, vec![observation(0.1, 0.0, 0.0)]))
            .unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].contributing_sensors.len(), 2);
        assert_eq!(snapshots[0].contributing_sensors[0], SensorId::new(1));
        assert_eq!(snapshots[0].contributing_sensors[1], SensorId::new(2));
        assert_eq!(snapshots[0].last_update_sensor, Some(SensorId::new(2)));
    }

    #[test]
    fn same_timestamp_multi_sensor_updates_same_track() {
        let mut tracker = Cartesian3DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        tracker
            .process_batch(&batch(2, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        assert_eq!(tracker.active_track_count(), 1);
        let snapshots = tracker.snapshots();
        assert_eq!(snapshots[0].contributing_sensors.len(), 2);
    }

    #[test]
    fn different_observation_periods_are_supported() {
        let mut tracker = Cartesian3DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        tracker
            .process_batch(&batch(1, 1, 100_000_000, vec![observation(1.0, 0.0, 0.0)]))
            .unwrap();
        // Sensor 2 runs at a slower rate but still associates.
        let snapshots = tracker
            .process_batch(&batch(2, 0, 200_000_000, vec![observation(2.0, 0.0, 0.0)]))
            .unwrap();
        assert_eq!(snapshots.len(), 1);
        assert!(snapshots[0].velocity.x > 0.0);
    }

    #[test]
    fn frame_mismatch_is_rejected() {
        let mut tracker = Cartesian3DTracker::new(config()).unwrap();
        let mut bad = batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]);
        bad.frame_id = FrameId::new("sensor-local");
        let error = tracker.process_batch(&bad).unwrap_err();
        assert!(matches!(
            error,
            TrackerError::Input(InputError::FrameMismatch { .. })
        ));
    }

    #[test]
    fn outlier_is_gated_and_spawns_track() {
        let mut tracker = Cartesian3DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        let snapshots = tracker
            .process_batch(&batch(
                1,
                1,
                100_000_000,
                vec![observation(1000.0, 0.0, 0.0)],
            ))
            .unwrap();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[1].track_id.value(), 2);
    }

    #[test]
    fn invalid_position_is_rejected_without_state_change() {
        let mut tracker = Cartesian3DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
            .unwrap();
        let before = tracker.snapshots();
        let bad = batch(1, 1, 100_000_000, vec![observation(f64::NAN, 0.0, 0.0)]);
        assert!(tracker.process_batch(&bad).is_err());
        assert_eq!(tracker.snapshots(), before);
    }

    #[test]
    fn invalid_covariance_is_rejected() {
        let mut tracker = Cartesian3DTracker::new(config()).unwrap();
        let mut bad_observation = observation(0.0, 0.0, 0.0);
        bad_observation.covariance = Matrix3::zeros();
        let bad = batch(1, 0, 0, vec![bad_observation]);
        let error = tracker.process_batch(&bad).unwrap_err();
        assert!(error.is_numeric_error() || error.is_input_error());
    }

    #[test]
    fn lost_then_deleted_by_time() {
        let mut tracker = Cartesian3DTracker::new(config()).unwrap();
        tracker
            .process_batch(&batch(1, 0, 0, vec![observation(0.0, 0.0, 0.0)]))
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
    fn velocity_converges_for_constant_motion() {
        let mut config = config();
        config.accel_spectral_density = 0.01;
        config.initial_velocity_variance = 25.0;
        let mut tracker = Cartesian3DTracker::new(config).unwrap();
        let mut snapshots = Vec::new();
        for step in 0..6 {
            let t = step as i64 * 1_000_000_000;
            snapshots = tracker
                .process_batch(&batch(
                    1,
                    step,
                    t,
                    vec![observation(10.0 * step as f64, 0.0, 0.0)],
                ))
                .unwrap();
        }
        assert_eq!(snapshots.len(), 1);
        assert_relative_eq!(snapshots[0].velocity.x, 10.0, epsilon = 1.0);
    }
}
