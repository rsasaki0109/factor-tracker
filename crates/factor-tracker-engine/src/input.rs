//! Engine-level input types.
//!
//! These carry sensor-native detections and their *event time* (when the
//! sensor observed the scene). Arrival time is supplied separately by the
//! caller and handled by the reorder buffer.

use std::collections::BTreeMap;

use factor_tracker_core::{BBox2D, ClassLabel, FrameId, SensorId, Timestamp};
use nalgebra::{Matrix3, Vector2, Vector3};

use crate::geometry::Box3DShape;

/// A LiDAR detection: a 3D position in the LiDAR frame plus optional shape
/// metadata used only for camera projection.
#[derive(Debug, Clone, PartialEq)]
pub struct LidarDetection {
    /// Position `[x, y, z]` in the LiDAR sensor frame, metres.
    pub position: Vector3<f64>,
    /// Position covariance in the LiDAR sensor frame, metres².
    pub covariance: Matrix3<f64>,
    /// Class in the tracking class space, if known.
    pub class_label: Option<ClassLabel>,
    /// Detector confidence in `[0, 1]`.
    pub score: f64,
    /// Optional oriented box metadata for projection. Not part of the tracked
    /// state.
    pub shape: Option<Box3DShape>,
    /// Optional appearance embedding (e.g. a ReID feature of the detection's
    /// image crop), used as an association cue by the tracker.
    pub appearance: Option<Vec<f32>>,
    /// Optional measured velocity over ground `[vx, vy]` in m/s, along the
    /// LiDAR frame's x and y axes (a detector that regresses velocity, such
    /// as CenterPoint). Used with `--velocity-std`.
    pub velocity: Option<Vector2<f64>>,
}

/// One LiDAR sensor's detections at one event time.
#[derive(Debug, Clone, PartialEq)]
pub struct LidarBatch {
    /// Sensor id.
    pub sensor_id: SensorId,
    /// Per-sensor sequence number.
    pub sequence: u64,
    /// Time the LiDAR observed the scene.
    pub event_time: Timestamp,
    /// Sensor frame of the positions.
    pub frame_id: FrameId,
    /// Detections.
    pub detections: Vec<LidarDetection>,
    /// Static objects (poles, trees, bins, ...) in the sensor frame. Used only
    /// for ego-pose correction (`EngineConfig::pose_correction`), never
    /// tracked. Send the reliable ones: near the sensor, high score.
    pub landmarks: Vec<Vector3<f64>>,
}

/// A single 2D camera detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraDetection {
    /// Detection id, unique within its batch.
    pub detection_id: u64,
    /// Box in image pixels.
    pub bbox: BBox2D,
    /// Class in the camera's class space, if known.
    pub class_label: Option<ClassLabel>,
    /// Detector confidence in `[0, 1]`.
    pub score: f64,
}

/// One camera's detections at one event time.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraBatch {
    /// Sensor id.
    pub sensor_id: SensorId,
    /// Per-sensor sequence number.
    pub sequence: u64,
    /// Time the camera observed the scene.
    pub event_time: Timestamp,
    /// Camera frame of the detections (must match the calibration).
    pub frame_id: FrameId,
    /// Detections.
    pub detections: Vec<CameraDetection>,
}

/// A batch from either sensor.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineBatch {
    /// LiDAR batch.
    Lidar(LidarBatch),
    /// Camera batch.
    Camera(CameraBatch),
}

impl EngineBatch {
    /// Sensor that produced the batch.
    pub fn sensor_id(&self) -> SensorId {
        match self {
            EngineBatch::Lidar(batch) => batch.sensor_id,
            EngineBatch::Camera(batch) => batch.sensor_id,
        }
    }

    /// Per-sensor sequence number.
    pub fn sequence(&self) -> u64 {
        match self {
            EngineBatch::Lidar(batch) => batch.sequence,
            EngineBatch::Camera(batch) => batch.sequence,
        }
    }

    /// Event time (when the sensor observed the scene).
    pub fn event_time(&self) -> Timestamp {
        match self {
            EngineBatch::Lidar(batch) => batch.event_time,
            EngineBatch::Camera(batch) => batch.event_time,
        }
    }

    /// Frame declared by the batch.
    pub fn frame_id(&self) -> &FrameId {
        match self {
            EngineBatch::Lidar(batch) => &batch.frame_id,
            EngineBatch::Camera(batch) => &batch.frame_id,
        }
    }

    /// Returns `true` for a LiDAR batch.
    pub fn is_lidar(&self) -> bool {
        matches!(self, EngineBatch::Lidar(_))
    }
}

/// Maps class ids from one sensor's class space into the tracking class space.
///
/// An empty map is the identity: every camera class passes through unchanged
/// (and an absent class stays absent). A non-empty map is authoritative:
/// camera classes with no entry become [`None`] (Unknown) and are never used as
/// evidence.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClassMap {
    /// Camera class id to tracking class id.
    pub camera_to_tracking: BTreeMap<u32, u32>,
}

impl ClassMap {
    /// The identity mapping.
    pub fn identity() -> Self {
        ClassMap::default()
    }

    /// Builds a mapping from pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, u32)>) -> Self {
        ClassMap {
            camera_to_tracking: pairs.into_iter().collect(),
        }
    }

    /// Returns `true` when the map is empty (identity behaviour).
    pub fn is_identity(&self) -> bool {
        self.camera_to_tracking.is_empty()
    }

    /// Maps a camera class into the tracking class space.
    pub fn camera_class(&self, label: Option<ClassLabel>) -> Option<ClassLabel> {
        let label = label?;
        if self.is_identity() {
            return Some(label);
        }
        self.camera_to_tracking
            .get(&label.value())
            .copied()
            .map(ClassLabel::new)
    }
}
