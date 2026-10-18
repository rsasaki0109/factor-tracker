//! The camera-assisted LiDAR tracking engine.
//!
//! The engine is the SDK entry point. It ties together:
//!
//! * calibration and (optional) ego-pose based normalization,
//! * the bounded asynchronous reorder buffer,
//! * the core [`Cartesian3DTracker`] for LiDAR position tracking,
//! * camera-to-track association and class evidence,
//! * diagnostics and per-frame outputs.
//!
//! The tracking mode selects how much of the camera information is used:
//!
//! * [`TrackingMode::LidarOnly`] — camera batches are ignored entirely.
//! * [`TrackingMode::CameraAssociated`] — camera associations and class
//!   evidence are recorded, but not fed back into the LiDAR association.
//! * [`TrackingMode::CameraAssisted`] — the accumulated camera class evidence
//!   is applied as a *soft* class prior on the core tracker.
//!
//! The camera never updates the 3D state, so position accuracy is not assumed
//! to improve.

use std::collections::{BTreeMap, HashMap};

use nalgebra::{Matrix3, Vector3};
use perception_core::{
    BBox2D, BatchReport3D, Cartesian3DTracker, Cartesian3DTrackerConfig, ClassLabel, ClassPrior,
    FrameId, ObservationBatch3D, ObservationOutcome, Position3DObservation, Timestamp, TrackId,
    TrackLifecycleState,
};
use serde::{Deserialize, Serialize};

use crate::camera::{
    associate_camera, CameraAssociationConfig, CameraDetectionMatch, CameraMatchReason,
    ClassEvidence, ClassEvidenceConfig,
};
use crate::diagnostics::{
    AllCounters, DiagnosticEvent, DiagnosticLevel, DiagnosticLog, EngineCounters,
};
use crate::error::{EngineError, EngineResult};
use crate::geometry::{inverse, Box3DShape, Calibration, ProjectedBox};
use crate::input::{CameraBatch, EngineBatch, LidarBatch};
use crate::reorder::{AsyncReorderBuffer, IngestOutcome, ReorderConfig};

/// How much camera information is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackingMode {
    /// Ignore the camera entirely.
    LidarOnly,
    /// Associate the camera but do not influence LiDAR association.
    CameraAssociated,
    /// Use camera class evidence as a soft prior on LiDAR association.
    CameraAssisted,
}

impl TrackingMode {
    /// Stable lowercase name.
    pub const fn as_str(self) -> &'static str {
        match self {
            TrackingMode::LidarOnly => "lidar_only",
            TrackingMode::CameraAssociated => "camera_associated",
            TrackingMode::CameraAssisted => "camera_assisted",
        }
    }
}

/// Engine configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    /// Tracking mode.
    pub mode: TrackingMode,
    /// Core tracker configuration.
    pub tracker: Cartesian3DTrackerConfig,
    /// Camera association configuration.
    pub camera_association: CameraAssociationConfig,
    /// Class evidence configuration.
    pub class_evidence: ClassEvidenceConfig,
    /// Reorder buffer configuration.
    pub reorder: ReorderConfig,
    /// Camera class id to tracking class id map.
    pub class_map: crate::input::ClassMap,
    /// Shape metadata older than this (relative to the camera time) is not
    /// used for projection.
    pub max_shape_age_ns: i64,
    /// Tracks whose last spatial update is older than this are not projected.
    pub max_track_age_for_projection_ns: i64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            mode: TrackingMode::CameraAssociated,
            tracker: Cartesian3DTrackerConfig {
                // LiDAR position updates at ~10 Hz: tracks observed in the last
                // three scans (up to two misses) are associated before tracks
                // that have coasted longer. Tuned on CODa and KITTI; see
                // docs/evaluation.md ("Matching cascade").
                recent_track_window_ns: Some(350_000_000),
                ..Cartesian3DTrackerConfig::default()
            },
            camera_association: CameraAssociationConfig::default(),
            class_evidence: ClassEvidenceConfig::default(),
            reorder: ReorderConfig::default(),
            class_map: crate::input::ClassMap::identity(),
            max_shape_age_ns: 500_000_000,
            max_track_age_for_projection_ns: 1_000_000_000,
        }
    }
}

impl EngineConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> EngineResult<()> {
        self.tracker
            .validate()
            .map_err(|error| EngineError::config(format!("tracker: {error}")))?;
        self.camera_association.validate()?;
        self.class_evidence.validate()?;
        self.reorder.validate()?;
        if self.max_shape_age_ns <= 0 {
            return Err(EngineError::config("max_shape_age_ns must be > 0"));
        }
        if self.max_track_age_for_projection_ns <= 0 {
            return Err(EngineError::config(
                "max_track_age_for_projection_ns must be > 0",
            ));
        }
        Ok(())
    }
}

/// Per-track output for one processed timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackOutput {
    /// Track id.
    pub track_id: u64,
    /// Lifecycle name.
    pub lifecycle: String,
    /// State time.
    pub state_timestamp_ns: i64,
    /// Last spatial (LiDAR) update time.
    pub last_spatial_update_ns: Option<i64>,
    /// Last time the camera supported (associated with) this track.
    pub last_camera_support_ns: Option<i64>,
    /// Age since the last spatial update, at `state_timestamp_ns`.
    pub age_since_spatial_update_ns: i64,
    /// Whether the estimate is prediction-only.
    pub prediction_only: bool,
    /// Position in metres.
    pub position: Vec<f64>,
    /// Velocity in metres per second.
    pub velocity: Vec<f64>,
    /// Row-major `6x6` covariance.
    pub covariance: Vec<f64>,
    /// Class from positional observations.
    pub class_label: Option<u32>,
    /// Soft class prior `[label, confidence]`.
    pub class_prior: Option<Vec<f64>>,
    /// Contributing sensors.
    pub contributing_sensors: Vec<u32>,
    /// Distinct observation timestamps.
    pub observation_times_count: u32,
}

/// Association summary for one LiDAR observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarAssociationOutput {
    /// Index in the batch.
    pub observation_index: usize,
    /// Track it updated or spawned.
    pub track_id: Option<u64>,
    /// `updated`, `spawned` or `unmatched`.
    pub outcome: String,
    /// Assignment cost, if matched.
    pub cost: Option<f64>,
}

/// Output produced for each processed LiDAR batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarFrameOutput {
    /// Mode.
    pub mode: TrackingMode,
    /// Event time.
    pub timestamp_ns: i64,
    /// Live tracks after the batch.
    pub tracks: Vec<TrackOutput>,
    /// LiDAR observation positions in the tracking frame (normalized).
    pub observations: Vec<Vec<f64>>,
    /// LiDAR observation associations.
    pub associations: Vec<LidarAssociationOutput>,
    /// Tracks not matched by any observation.
    pub unmatched_track_ids: Vec<u64>,
}

/// One camera association record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraMatchOutput {
    /// Index in the camera batch.
    pub detection_index: usize,
    /// Detection id.
    pub detection_id: u64,
    /// Detection box `[x_min, y_min, x_max, y_max]` in pixels.
    pub detection_bbox: Vec<f64>,
    /// Associated track.
    pub track_id: Option<u64>,
    /// Projected track box `[x_min, y_min, x_max, y_max]`.
    pub projected_bbox: Option<Vec<f64>>,
    /// Assignment cost (`1 - IoU`).
    pub cost: Option<f64>,
    /// Best IoU over all candidate tracks.
    pub best_iou: f64,
    /// Whether accepted.
    pub accepted: bool,
    /// `accepted`, `gated_iou` or `unmatched`.
    pub reason: String,
}

/// A track that could not be projected, with the reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionRejectionOutput {
    /// Track id.
    pub track_id: u64,
    /// Machine-readable reason.
    pub reason: String,
}

/// Output produced for each processed camera batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraFrameOutput {
    /// Mode.
    pub mode: TrackingMode,
    /// Camera event time.
    pub timestamp_ns: i64,
    /// Camera association records.
    pub matches: Vec<CameraMatchOutput>,
    /// Tracks excluded from projection and why.
    pub projection_rejections: Vec<ProjectionRejectionOutput>,
}

#[derive(Debug, Clone)]
struct ShapeRecord {
    shape: Box3DShape,
    captured_at: Timestamp,
}

/// Camera-assisted LiDAR tracking engine.
pub struct CameraAssistedLidarEngine {
    config: EngineConfig,
    calibration: Calibration,
    poses: Option<crate::geometry::PoseTrack>,
    tracker: Cartesian3DTracker,
    shapes: HashMap<TrackId, ShapeRecord>,
    class_evidence: HashMap<TrackId, ClassEvidence>,
    last_camera_support: HashMap<TrackId, Timestamp>,
    buffer: AsyncReorderBuffer,
    counters: EngineCounters,
    log: DiagnosticLog,
    lidar_frames: Vec<LidarFrameOutput>,
    camera_frames: Vec<CameraFrameOutput>,
}

impl CameraAssistedLidarEngine {
    /// Creates an engine from configuration, calibration and optional poses.
    pub fn new(
        config: EngineConfig,
        calibration: Calibration,
        poses: Option<crate::geometry::PoseTrack>,
    ) -> EngineResult<Self> {
        config.validate()?;
        calibration.validate()?;
        let tracker = Cartesian3DTracker::new(config.tracker.clone())?;
        let initial_frame = tracker.frame_id().clone();
        if initial_frame != calibration.tracking_frame {
            return Err(EngineError::config(format!(
                "tracker frame '{}' does not match calibration tracking frame '{}'",
                initial_frame, calibration.tracking_frame
            )));
        }
        let buffer = AsyncReorderBuffer::new(config.reorder.clone())?;
        Ok(CameraAssistedLidarEngine {
            config,
            calibration,
            poses,
            tracker,
            shapes: HashMap::new(),
            class_evidence: HashMap::new(),
            last_camera_support: HashMap::new(),
            buffer,
            counters: EngineCounters::default(),
            log: DiagnosticLog::default(),
            lidar_frames: Vec::new(),
            camera_frames: Vec::new(),
        })
    }

    /// Configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Calibration.
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// Tracking mode.
    pub fn mode(&self) -> TrackingMode {
        self.config.mode
    }

    /// Combined counters.
    pub fn counters(&self) -> AllCounters {
        AllCounters {
            reorder: self.buffer.counters().clone(),
            engine: self.counters.clone(),
        }
    }

    /// Diagnostic log.
    pub fn log(&self) -> &DiagnosticLog {
        &self.log
    }

    /// LiDAR frame outputs, in processing order.
    pub fn lidar_frames(&self) -> &[LidarFrameOutput] {
        &self.lidar_frames
    }

    /// Camera frame outputs, in processing order.
    pub fn camera_frames(&self) -> &[CameraFrameOutput] {
        &self.camera_frames
    }

    /// Number of buffered batches.
    pub fn pending_len(&self) -> usize {
        self.buffer.pending_len()
    }

    /// Ingests a batch at `arrival_time`. Does not process it.
    pub fn ingest(&mut self, batch: EngineBatch, arrival_time: Timestamp) -> IngestOutcome {
        let event_time = batch.event_time();
        let sensor = batch.sensor_id();
        let outcome = self.buffer.ingest(batch, arrival_time);
        if outcome != IngestOutcome::Queued {
            let (code, message) = match outcome {
                IngestOutcome::Late => ("late_batch", "batch older than last released event time"),
                IngestOutcome::Duplicate => ("duplicate_batch", "duplicate sensor/sequence batch"),
                IngestOutcome::Overflow => ("buffer_overflow", "reorder buffer is full"),
                IngestOutcome::Queued => ("queued", ""),
            };
            self.log.push(DiagnosticEvent {
                arrival_time,
                event_time: Some(event_time),
                sensor_id: Some(sensor),
                code,
                level: DiagnosticLevel::Warn,
                message: message.to_string(),
            });
        }
        outcome
    }

    /// Releases and processes all batches ready at `arrival_now`.
    ///
    /// Returns the number of batches processed.
    pub fn poll(&mut self, arrival_now: Timestamp) -> EngineResult<usize> {
        let released = self.buffer.poll(arrival_now);
        let count = released.len();
        for batch in released {
            self.process_batch(batch, arrival_now)?;
        }
        Ok(count)
    }

    /// Processes all remaining buffered batches in event-time order.
    pub fn flush(&mut self) -> EngineResult<usize> {
        let released = self.buffer.flush();
        let count = released.len();
        for batch in released {
            self.process_batch(batch, Timestamp::from_nanos(i64::MAX))?;
        }
        Ok(count)
    }

    fn process_batch(&mut self, batch: EngineBatch, arrival_time: Timestamp) -> EngineResult<()> {
        match batch {
            EngineBatch::Lidar(batch) => self.process_lidar(&batch, arrival_time),
            EngineBatch::Camera(batch) => self.process_camera(&batch, arrival_time),
        }
    }

    /// Transforms a sensor frame into the tracking frame at `timestamp`.
    fn sensor_to_tracking(
        &self,
        extrinsics: &perception_core::RigidTransform3D,
        timestamp: Timestamp,
    ) -> EngineResult<perception_core::RigidTransform3D> {
        match &self.poses {
            Some(poses) => {
                let pose = poses.pose_at(timestamp)?;
                Ok(crate::geometry::compose(&pose, extrinsics))
            }
            None => Ok(extrinsics.clone()),
        }
    }

    fn process_lidar(&mut self, batch: &LidarBatch, arrival_time: Timestamp) -> EngineResult<()> {
        if batch.frame_id != self.calibration.lidar_frame {
            self.counters.rejected_invalid += 1;
            self.log.push(DiagnosticEvent {
                arrival_time,
                event_time: Some(batch.event_time),
                sensor_id: Some(batch.sensor_id),
                code: "frame_mismatch",
                level: DiagnosticLevel::Warn,
                message: format!(
                    "lidar batch frame '{}' does not match calibration '{}'",
                    batch.frame_id, self.calibration.lidar_frame
                ),
            });
            return Ok(());
        }

        let sensor_to_tracking =
            match self.sensor_to_tracking(&self.calibration.lidar_to_reference, batch.event_time) {
                Ok(transform) => transform,
                Err(error) => {
                    self.counters.transform_unavailable += 1;
                    self.log.push(DiagnosticEvent {
                        arrival_time,
                        event_time: Some(batch.event_time),
                        sensor_id: Some(batch.sensor_id),
                        code: "transform_unavailable",
                        level: DiagnosticLevel::Warn,
                        message: format!("lidar transform skipped: {error}"),
                    });
                    return Ok(());
                }
            };

        let rotation = *sensor_to_tracking.rotation();
        let translation = *sensor_to_tracking.translation();
        let mut observations = Vec::with_capacity(batch.detections.len());
        for detection in &batch.detections {
            let position = rotation * detection.position + translation;
            let covariance = sensor_to_tracking
                .transform_covariance(&detection.covariance)
                .map_err(EngineError::Core)?;
            observations.push(Position3DObservation {
                position,
                covariance,
                class_label: detection.class_label,
                appearance: detection.appearance.clone(),
            });
        }

        let core_batch = ObservationBatch3D {
            sensor_id: batch.sensor_id,
            sequence: batch.sequence,
            timestamp: batch.event_time,
            frame_id: self.calibration.tracking_frame.clone(),
            observations,
        };

        let (snapshots, report) = match self.tracker.process_batch_reported(&core_batch) {
            Ok(value) => value,
            Err(error) => {
                self.log.push(DiagnosticEvent {
                    arrival_time,
                    event_time: Some(batch.event_time),
                    sensor_id: Some(batch.sensor_id),
                    code: "core_rejected",
                    level: DiagnosticLevel::Warn,
                    message: format!("tracker rejected lidar batch: {error}"),
                });
                return Err(EngineError::Core(error));
            }
        };

        // Attach shape metadata to updated/spawned tracks.
        for association in &report.associations {
            let Some(track_id) = association.track_id else {
                continue;
            };
            match association.outcome {
                ObservationOutcome::Updated | ObservationOutcome::Spawned => {
                    if let Some(shape) = &batch.detections[association.observation_index].shape {
                        self.shapes.insert(
                            track_id,
                            ShapeRecord {
                                shape: shape.transformed(&rotation),
                                captured_at: batch.event_time,
                            },
                        );
                    }
                }
                ObservationOutcome::Unmatched => {}
            }
        }

        self.prune_track_state(&snapshots);
        self.counters.lidar_batches += 1;

        let associations = report
            .associations
            .iter()
            .map(|association| LidarAssociationOutput {
                observation_index: association.observation_index,
                track_id: association.track_id.map(TrackId::value),
                outcome: match association.outcome {
                    ObservationOutcome::Updated => "updated",
                    ObservationOutcome::Spawned => "spawned",
                    ObservationOutcome::Unmatched => "unmatched",
                }
                .to_string(),
                cost: association.cost,
            })
            .collect();

        let tracks = self.track_outputs(&snapshots);
        self.lidar_frames.push(LidarFrameOutput {
            mode: self.config.mode,
            timestamp_ns: batch.event_time.as_nanos(),
            tracks,
            observations: core_batch
                .observations
                .iter()
                .map(|observation| observation.position.iter().copied().collect())
                .collect(),
            associations,
            unmatched_track_ids: report
                .unmatched_track_ids
                .iter()
                .map(|id| id.value())
                .collect(),
        });
        let _ = report;
        Ok(())
    }

    fn process_camera(&mut self, batch: &CameraBatch, arrival_time: Timestamp) -> EngineResult<()> {
        self.counters.camera_batches += 1;
        if self.config.mode == TrackingMode::LidarOnly {
            self.log.push(DiagnosticEvent {
                arrival_time,
                event_time: Some(batch.event_time),
                sensor_id: Some(batch.sensor_id),
                code: "camera_skipped",
                level: DiagnosticLevel::Info,
                message: "camera batch ignored in lidar_only mode".to_string(),
            });
            return Ok(());
        }
        if batch.frame_id != self.calibration.camera_frame {
            self.counters.rejected_invalid += 1;
            self.log.push(DiagnosticEvent {
                arrival_time,
                event_time: Some(batch.event_time),
                sensor_id: Some(batch.sensor_id),
                code: "frame_mismatch",
                level: DiagnosticLevel::Warn,
                message: format!(
                    "camera batch frame '{}' does not match calibration '{}'",
                    batch.frame_id, self.calibration.camera_frame
                ),
            });
            return Ok(());
        }

        let camera_to_tracking = match self
            .sensor_to_tracking(&self.calibration.camera_to_reference, batch.event_time)
        {
            Ok(transform) => transform,
            Err(error) => {
                self.counters.transform_unavailable += 1;
                self.log.push(DiagnosticEvent {
                    arrival_time,
                    event_time: Some(batch.event_time),
                    sensor_id: Some(batch.sensor_id),
                    code: "transform_unavailable",
                    level: DiagnosticLevel::Warn,
                    message: format!("camera transform skipped: {error}"),
                });
                return Ok(());
            }
        };
        let tracking_to_camera = inverse(&camera_to_tracking);

        // Predict tracks to the camera time for projection.
        self.tracker.predict_to(batch.event_time)?;
        let snapshots = self.tracker.snapshots();

        let mut candidates: Vec<(TrackId, ProjectedBox)> = Vec::new();
        let mut rejections: Vec<ProjectionRejectionOutput> = Vec::new();
        for snapshot in &snapshots {
            let track_id = snapshot.track_id;
            let Some(record) = self.shapes.get(&track_id) else {
                rejections.push(ProjectionRejectionOutput {
                    track_id: track_id.value(),
                    reason: "no_shape".to_string(),
                });
                continue;
            };
            let shape_age = batch
                .event_time
                .checked_duration_since(record.captured_at)
                .unwrap_or(i64::MAX);
            if shape_age > self.config.max_shape_age_ns {
                rejections.push(ProjectionRejectionOutput {
                    track_id: track_id.value(),
                    reason: "stale_shape".to_string(),
                });
                continue;
            }
            let last_spatial = snapshot.last_observation_timestamp.map(|t| t.as_nanos());
            let track_age = last_spatial
                .map(|t| batch.event_time.as_nanos().saturating_sub(t))
                .unwrap_or(i64::MAX);
            if track_age > self.config.max_track_age_for_projection_ns {
                rejections.push(ProjectionRejectionOutput {
                    track_id: track_id.value(),
                    reason: "stale_track".to_string(),
                });
                continue;
            }

            let corners_tracking = record.shape.corners_at(&snapshot.position);
            let mut corners_camera = [Vector3::zeros(); 8];
            for (i, corner) in corners_tracking.iter().enumerate() {
                corners_camera[i] = tracking_to_camera.transform_position(corner);
            }
            match crate::geometry::project_box_corners(
                &self.calibration.intrinsics,
                self.calibration.near_plane,
                &corners_camera,
            ) {
                Ok(projected) => candidates.push((track_id, projected)),
                Err(rejection) => rejections.push(ProjectionRejectionOutput {
                    track_id: track_id.value(),
                    reason: rejection.as_str().to_string(),
                }),
            }
        }

        let (matches, _unmatched_tracks) = associate_camera(
            &batch.detections,
            &candidates,
            &self.config.camera_association,
        )?;

        let mut match_outputs = Vec::with_capacity(matches.len());
        for m in &matches {
            if m.accepted {
                self.counters.camera_matches += 1;
                if let Some(track_id) = m.track_id {
                    self.last_camera_support.insert(track_id, batch.event_time);
                    let mapped = self
                        .config
                        .class_map
                        .camera_class(batch.detections[m.detection_index].class_label);
                    if let (Some(label), Some(score)) =
                        (mapped, Some(batch.detections[m.detection_index].score))
                    {
                        let evidence = self
                            .class_evidence
                            .entry(track_id)
                            .or_insert_with(|| ClassEvidence::new(batch.event_time));
                        evidence.add(
                            label,
                            score,
                            batch.event_time,
                            self.config.class_evidence.half_life_ns,
                        );
                    }
                }
            } else {
                self.counters.camera_unmatched += 1;
                if m.reason == CameraMatchReason::GatedIoU {
                    self.counters.association_rejected += 1;
                }
            }
            match_outputs.push(convert_match(m, batch.detections[m.detection_index].bbox));
        }

        if self.config.mode == TrackingMode::CameraAssisted {
            self.apply_class_priors(batch.event_time);
        }

        self.prune_track_state(&snapshots);
        self.camera_frames.push(CameraFrameOutput {
            mode: self.config.mode,
            timestamp_ns: batch.event_time.as_nanos(),
            matches: match_outputs,
            projection_rejections: rejections,
        });
        Ok(())
    }

    fn apply_class_priors(&mut self, now: Timestamp) {
        let track_ids: Vec<TrackId> = self
            .tracker
            .snapshots()
            .iter()
            .map(|s| s.track_id)
            .collect();
        for track_id in track_ids {
            let prior: Option<ClassPrior> = self
                .class_evidence
                .get(&track_id)
                .and_then(|evidence| evidence.best(now, &self.config.class_evidence));
            match prior {
                Some(prior) => {
                    self.tracker
                        .set_class_prior(track_id, prior.label, prior.confidence);
                }
                None => {
                    self.tracker.clear_class_prior(track_id);
                }
            }
        }
    }

    fn prune_track_state(&mut self, snapshots: &[perception_core::Track3DSnapshot]) {
        let live: std::collections::HashSet<TrackId> =
            snapshots.iter().map(|snapshot| snapshot.track_id).collect();
        self.shapes.retain(|id, _| live.contains(id));
        self.class_evidence.retain(|id, _| live.contains(id));
        self.last_camera_support.retain(|id, _| live.contains(id));
    }

    fn track_outputs(&self, snapshots: &[perception_core::Track3DSnapshot]) -> Vec<TrackOutput> {
        snapshots
            .iter()
            .map(|snapshot| {
                let last_spatial = snapshot.last_observation_timestamp.map(|t| t.as_nanos());
                let state_ns = snapshot.state_timestamp.as_nanos();
                let age = last_spatial
                    .map(|t| state_ns.saturating_sub(t))
                    .unwrap_or(0);
                TrackOutput {
                    track_id: snapshot.track_id.value(),
                    lifecycle: snapshot.lifecycle.as_str().to_string(),
                    state_timestamp_ns: state_ns,
                    last_spatial_update_ns: last_spatial,
                    last_camera_support_ns: self
                        .last_camera_support
                        .get(&snapshot.track_id)
                        .map(|timestamp| timestamp.as_nanos()),
                    age_since_spatial_update_ns: age,
                    prediction_only: snapshot.is_prediction_only(),
                    position: snapshot.position.iter().copied().collect(),
                    velocity: snapshot.velocity.iter().copied().collect(),
                    covariance: row_major_6(&snapshot.covariance),
                    class_label: snapshot.class_label.map(ClassLabel::value),
                    class_prior: snapshot
                        .class_prior
                        .map(|prior| vec![prior.label.value() as f64, prior.confidence]),
                    contributing_sensors: snapshot
                        .contributing_sensors
                        .iter()
                        .map(|sensor| sensor.value())
                        .collect(),
                    observation_times_count: snapshot.observation_times_count,
                }
            })
            .collect()
    }
}

fn convert_match(m: &CameraDetectionMatch, detection_bbox: BBox2D) -> CameraMatchOutput {
    CameraMatchOutput {
        detection_index: m.detection_index,
        detection_id: m.detection_id,
        detection_bbox: vec![
            detection_bbox.x_min,
            detection_bbox.y_min,
            detection_bbox.x_max,
            detection_bbox.y_max,
        ],
        track_id: m.track_id.map(TrackId::value),
        projected_bbox: m
            .projected_bbox
            .map(|bbox| vec![bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]),
        cost: m.cost,
        best_iou: m.best_iou,
        accepted: m.accepted,
        reason: m.reason.as_str().to_string(),
    }
}

fn row_major_6(m: &nalgebra::SMatrix<f64, 6, 6>) -> Vec<f64> {
    (0..6)
        .flat_map(|r| (0..6).map(move |c| m[(r, c)]))
        .collect()
}

/// Helper for adapters: transform a covariance with a rotation.
pub fn rotate_covariance(rotation: &Matrix3<f64>, covariance: &Matrix3<f64>) -> Matrix3<f64> {
    let result = rotation * covariance * rotation.transpose();
    0.5 * (result + result.transpose())
}

/// Convenience: build a tracking frame id from a string.
pub fn frame(name: &str) -> FrameId {
    FrameId::new(name)
}

/// Convenience: lifecycle states considered alive for reporting.
pub fn is_alive(state: TrackLifecycleState) -> bool {
    !matches!(state, TrackLifecycleState::Deleted)
}

/// Re-export for adapters that need the core report type.
pub type CoreBatchReport = BatchReport3D;

/// A `BTreeMap` of class ids used by adapters.
pub type ClassIdMap = BTreeMap<u32, u32>;
