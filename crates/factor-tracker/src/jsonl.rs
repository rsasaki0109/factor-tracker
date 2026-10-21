//! JSONL serialization schema for observations, tracks and ground truth.
//!
//! See `docs/data-format.md` for the full schema documentation.
//!
//! Conventions
//! -----------
//! * Every record has an integer `schema_version` (currently `1`) and a string
//!   `type` discriminator.
//! * Timestamps are integer nanoseconds on one shared timeline.
//! * 2D bounding boxes are `[x_min, y_min, x_max, y_max]` in pixels.
//! * 2D covariance is row-major for the measurement vector `[cx, cy, w, h]`
//!   and has length 16.
//! * 3D positions are `[x, y, z]` in meters; 3D covariance is row-major for
//!   `[x, y, z]` and has length 9.
//! * Track covariance arrays are row-major for the full state vector.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use factor_tracker_core::{
    BBox2D, BBox2DObservation, FrameId, ObservationBatch2D, ObservationBatch3D,
    Position3DObservation, SensorId, Timestamp,
};
use nalgebra::{Matrix3, Matrix4, SMatrix, SVector};
use serde::{Deserialize, Serialize};

/// Current schema version written by this CLI.
pub const SCHEMA_VERSION: u32 = 1;

/// Convenience result type used across the CLI.
pub type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

/// A single observation record line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ObservationRecord {
    /// A batch of 2D bounding-box observations from one sensor.
    #[serde(rename = "bbox2d_batch")]
    Bbox2dBatch {
        /// Schema version.
        schema_version: u32,
        /// Sensor identifier.
        sensor_id: u32,
        /// Per-sensor sequence number.
        sequence: u64,
        /// Observation time in nanoseconds.
        timestamp_ns: i64,
        /// Image frame.
        frame_id: String,
        /// Detections.
        observations: Vec<Bbox2dObservationRecord>,
    },
    /// A batch of 3D position observations from one sensor.
    #[serde(rename = "position3d_batch")]
    Position3dBatch {
        /// Schema version.
        schema_version: u32,
        /// Sensor identifier.
        sensor_id: u32,
        /// Per-sensor sequence number.
        sequence: u64,
        /// Observation time in nanoseconds.
        timestamp_ns: i64,
        /// Coordinate frame.
        frame_id: String,
        /// Detections.
        observations: Vec<Position3dObservationRecord>,
    },
}

/// A single 2D detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bbox2dObservationRecord {
    /// `[x_min, y_min, x_max, y_max]` in pixels.
    pub bbox: Vec<f64>,
    /// Detector confidence in `[0, 1]`.
    pub score: f64,
    /// Optional class id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<u32>,
    /// Row-major `4x4` covariance for `[cx, cy, w, h]`, 16 entries.
    pub covariance: Vec<f64>,
}

/// A single 3D position detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position3dObservationRecord {
    /// `[x, y, z]` in meters.
    pub position: Vec<f64>,
    /// Optional class id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<u32>,
    /// Row-major `3x3` covariance, 9 entries.
    pub covariance: Vec<f64>,
}

/// A line of tracked 2D objects at one timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TrackRecord {
    /// 2D track states.
    #[serde(rename = "bbox2d_tracks")]
    Bbox2dTracks {
        /// Schema version.
        schema_version: u32,
        /// Timestamp in nanoseconds.
        timestamp_ns: i64,
        /// Tracks.
        tracks: Vec<Bbox2dTrackRecord>,
    },
    /// 3D track states.
    #[serde(rename = "position3d_tracks")]
    Position3dTracks {
        /// Schema version.
        schema_version: u32,
        /// Timestamp in nanoseconds.
        timestamp_ns: i64,
        /// Tracks.
        tracks: Vec<Position3dTrackRecord>,
    },
}

/// One tracked 2D object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bbox2dTrackRecord {
    /// Track id.
    pub track_id: u64,
    /// Lifecycle state name.
    pub lifecycle: String,
    /// State timestamp in nanoseconds.
    pub state_timestamp_ns: i64,
    /// Last observation timestamp, if any.
    pub last_observation_timestamp_ns: Option<i64>,
    /// Sensor that last updated the track.
    pub last_update_sensor: Option<u32>,
    /// Distinct contributing sensors.
    pub contributing_sensors: Vec<u32>,
    /// Class id, if known.
    pub class: Option<u32>,
    /// Whether an observation updated the track in this batch.
    pub updated: bool,
    /// Whether the estimate is prediction-only.
    pub prediction_only: bool,
    /// Number of distinct observation timestamps.
    pub observation_times_count: u32,
    /// `[x_min, y_min, x_max, y_max]` in pixels.
    pub bbox: Vec<f64>,
    /// `[vx, vy]` in pixels per second.
    pub velocity: Vec<f64>,
    /// Full state `[cx, cy, w, h, vcx, vcy, vw, vh]`.
    pub state: Vec<f64>,
    /// Row-major `8x8` covariance, 64 entries.
    pub covariance: Vec<f64>,
}

/// One tracked 3D object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position3dTrackRecord {
    /// Track id.
    pub track_id: u64,
    /// Lifecycle state name.
    pub lifecycle: String,
    /// State timestamp in nanoseconds.
    pub state_timestamp_ns: i64,
    /// Last observation timestamp, if any.
    pub last_observation_timestamp_ns: Option<i64>,
    /// Sensor that last updated the track.
    pub last_update_sensor: Option<u32>,
    /// Distinct contributing sensors.
    pub contributing_sensors: Vec<u32>,
    /// Class id, if known.
    pub class: Option<u32>,
    /// Whether an observation updated the track in this batch.
    pub updated: bool,
    /// Whether the estimate is prediction-only.
    pub prediction_only: bool,
    /// Number of distinct observation timestamps.
    pub observation_times_count: u32,
    /// Position `[x, y, z]` in meters.
    pub position: Vec<f64>,
    /// Velocity `[vx, vy, vz]` in meters per second.
    pub velocity: Vec<f64>,
    /// Full state `[x, y, z, vx, vy, vz]`.
    pub state: Vec<f64>,
    /// Row-major `6x6` covariance, 36 entries.
    pub covariance: Vec<f64>,
}

/// A ground-truth line. Ground truth is never given to a tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GroundTruthRecord {
    /// 2D ground truth.
    #[serde(rename = "truth2d")]
    Truth2d {
        /// Schema version.
        schema_version: u32,
        /// Timestamp in nanoseconds.
        timestamp_ns: i64,
        /// Objects.
        objects: Vec<Truth2dObject>,
    },
    /// 3D ground truth.
    #[serde(rename = "truth3d")]
    Truth3d {
        /// Schema version.
        schema_version: u32,
        /// Timestamp in nanoseconds.
        timestamp_ns: i64,
        /// Objects.
        objects: Vec<Truth3dObject>,
    },
}

/// A ground-truth 2D object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Truth2dObject {
    /// Ground-truth object id (distinct from any track id).
    pub truth_id: u32,
    /// Class id.
    pub class: u32,
    /// `[x_min, y_min, x_max, y_max]` in pixels.
    pub bbox: Vec<f64>,
    /// Whether the object existed at this time.
    pub present: bool,
    /// Whether the synthetic sensor detected it in this frame.
    pub detected: bool,
}

/// A ground-truth 3D object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Truth3dObject {
    /// Ground-truth object id.
    pub truth_id: u32,
    /// Class id.
    pub class: u32,
    /// `[x, y, z]` in meters.
    pub position: Vec<f64>,
    /// Whether the object existed at this time.
    pub present: bool,
    /// Whether the synthetic sensor detected it in this frame.
    pub detected: bool,
}

/// Reads JSONL records of type `T`, skipping blank lines.
pub fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> CliResult<Vec<T>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<T>(&line)
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_number + 1))?;
        records.push(record);
    }
    Ok(records)
}

/// Writes JSONL records, one per line.
pub fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> CliResult<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn check_schema_version(version: u32) -> CliResult<()> {
    if version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported schema_version {version} (this CLI supports {SCHEMA_VERSION})"
        )
        .into());
    }
    Ok(())
}

fn vector<const N: usize>(context: &str, values: &[f64]) -> CliResult<SVector<f64, N>> {
    if values.len() != N {
        return Err(format!("{context}: expected {N} values, found {}", values.len()).into());
    }
    let mut out = SVector::<f64, N>::zeros();
    for (i, value) in values.iter().enumerate() {
        out[i] = *value;
    }
    Ok(out)
}

fn matrix<const N: usize>(context: &str, values: &[f64]) -> CliResult<SMatrix<f64, N, N>> {
    if values.len() != N * N {
        return Err(format!(
            "{context}: expected {} values, found {}",
            N * N,
            values.len()
        )
        .into());
    }
    let mut out = SMatrix::<f64, N, N>::zeros();
    for (index, value) in values.iter().enumerate() {
        out[(index / N, index % N)] = *value;
    }
    Ok(out)
}

impl ObservationRecord {
    /// Converts a JSONL record into a 2D observation batch.
    pub fn into_batch_2d(self) -> CliResult<ObservationBatch2D> {
        match self {
            ObservationRecord::Bbox2dBatch {
                schema_version,
                sensor_id,
                sequence,
                timestamp_ns,
                frame_id,
                observations,
            } => {
                check_schema_version(schema_version)?;
                let observations = observations
                    .into_iter()
                    .enumerate()
                    .map(|(index, record)| {
                        if record.bbox.len() != 4 {
                            return Err(format!(
                                "observation {index}: bbox must have 4 values, found {}",
                                record.bbox.len()
                            )
                            .into());
                        }
                        if !(record.bbox[0] < record.bbox[2] && record.bbox[1] < record.bbox[3]) {
                            return Err(format!(
                                "observation {index}: bbox has non-positive extent {:?}",
                                record.bbox
                            )
                            .into());
                        }
                        Ok(BBox2DObservation {
                            bbox: BBox2D::from_corners(
                                record.bbox[0],
                                record.bbox[1],
                                record.bbox[2],
                                record.bbox[3],
                            ),
                            score: record.score,
                            class_label: record.class.map(factor_tracker_core::ClassLabel::new),
                            covariance: matrix::<4>(
                                &format!("observation {index} covariance"),
                                &record.covariance,
                            )?,
                        })
                    })
                    .collect::<CliResult<Vec<_>>>()?;
                Ok(ObservationBatch2D {
                    sensor_id: SensorId::new(sensor_id),
                    sequence,
                    timestamp: Timestamp::from_nanos(timestamp_ns),
                    frame_id: FrameId::new(frame_id),
                    observations,
                })
            }
            ObservationRecord::Position3dBatch { .. } => {
                Err("expected a bbox2d_batch record but found position3d_batch".into())
            }
        }
    }

    /// Converts a JSONL record into a 3D observation batch.
    pub fn into_batch_3d(self) -> CliResult<ObservationBatch3D> {
        match self {
            ObservationRecord::Position3dBatch {
                schema_version,
                sensor_id,
                sequence,
                timestamp_ns,
                frame_id,
                observations,
            } => {
                check_schema_version(schema_version)?;
                let observations = observations
                    .into_iter()
                    .enumerate()
                    .map(|(index, record)| {
                        Ok(Position3DObservation {
                            position: vector::<3>(
                                &format!("observation {index} position"),
                                &record.position,
                            )?,
                            covariance: matrix::<3>(
                                &format!("observation {index} covariance"),
                                &record.covariance,
                            )?,
                            class_label: record.class.map(factor_tracker_core::ClassLabel::new),
                            appearance: None,
                            score: None,
                        })
                    })
                    .collect::<CliResult<Vec<_>>>()?;
                Ok(ObservationBatch3D {
                    sensor_id: SensorId::new(sensor_id),
                    sequence,
                    timestamp: Timestamp::from_nanos(timestamp_ns),
                    frame_id: FrameId::new(frame_id),
                    observations,
                })
            }
            ObservationRecord::Bbox2dBatch { .. } => {
                Err("expected a position3d_batch record but found bbox2d_batch".into())
            }
        }
    }
}

/// Builds a 2D observation record from a core batch.
pub fn record_from_batch_2d(batch: &ObservationBatch2D) -> ObservationRecord {
    ObservationRecord::Bbox2dBatch {
        schema_version: SCHEMA_VERSION,
        sensor_id: batch.sensor_id.value(),
        sequence: batch.sequence,
        timestamp_ns: batch.timestamp.as_nanos(),
        frame_id: batch.frame_id.to_string(),
        observations: batch
            .observations
            .iter()
            .map(|observation| Bbox2dObservationRecord {
                bbox: vec![
                    observation.bbox.x_min,
                    observation.bbox.y_min,
                    observation.bbox.x_max,
                    observation.bbox.y_max,
                ],
                score: observation.score,
                class: observation.class_label.map(|label| label.value()),
                covariance: row_major_4(&observation.covariance),
            })
            .collect(),
    }
}

/// Builds a 3D observation record from a core batch.
pub fn record_from_batch_3d(batch: &ObservationBatch3D) -> ObservationRecord {
    ObservationRecord::Position3dBatch {
        schema_version: SCHEMA_VERSION,
        sensor_id: batch.sensor_id.value(),
        sequence: batch.sequence,
        timestamp_ns: batch.timestamp.as_nanos(),
        frame_id: batch.frame_id.to_string(),
        observations: batch
            .observations
            .iter()
            .map(|observation| Position3dObservationRecord {
                position: observation.position.iter().copied().collect(),
                class: observation.class_label.map(|label| label.value()),
                covariance: row_major_3(&observation.covariance),
            })
            .collect(),
    }
}

/// Snapshot of a 2D track converted for serialization.
pub fn track_record_2d(snapshot: &factor_tracker_core::Track2DSnapshot) -> Bbox2dTrackRecord {
    Bbox2dTrackRecord {
        track_id: snapshot.track_id.value(),
        lifecycle: snapshot.lifecycle.as_str().to_string(),
        state_timestamp_ns: snapshot.state_timestamp.as_nanos(),
        last_observation_timestamp_ns: snapshot.last_observation_timestamp.map(Timestamp::as_nanos),
        last_update_sensor: snapshot.last_update_sensor.map(SensorId::value),
        contributing_sensors: snapshot
            .contributing_sensors
            .iter()
            .map(|sensor| sensor.value())
            .collect(),
        class: snapshot.class_label.map(|label| label.value()),
        updated: snapshot.updated_in_last_batch,
        prediction_only: snapshot.is_prediction_only(),
        observation_times_count: snapshot.observation_times_count,
        bbox: vec![
            snapshot.bbox.x_min,
            snapshot.bbox.y_min,
            snapshot.bbox.x_max,
            snapshot.bbox.y_max,
        ],
        velocity: snapshot.velocity.iter().copied().collect(),
        state: snapshot.state.iter().copied().collect(),
        covariance: row_major_8(&snapshot.covariance),
    }
}

/// Snapshot of a 3D track converted for serialization.
pub fn track_record_3d(snapshot: &factor_tracker_core::Track3DSnapshot) -> Position3dTrackRecord {
    Position3dTrackRecord {
        track_id: snapshot.track_id.value(),
        lifecycle: snapshot.lifecycle.as_str().to_string(),
        state_timestamp_ns: snapshot.state_timestamp.as_nanos(),
        last_observation_timestamp_ns: snapshot.last_observation_timestamp.map(Timestamp::as_nanos),
        last_update_sensor: snapshot.last_update_sensor.map(SensorId::value),
        contributing_sensors: snapshot
            .contributing_sensors
            .iter()
            .map(|sensor| sensor.value())
            .collect(),
        class: snapshot.class_label.map(|label| label.value()),
        updated: snapshot.updated_in_last_batch,
        prediction_only: snapshot.is_prediction_only(),
        observation_times_count: snapshot.observation_times_count,
        position: snapshot.position.iter().copied().collect(),
        velocity: snapshot.velocity.iter().copied().collect(),
        state: snapshot.state.iter().copied().collect(),
        covariance: row_major_6(&snapshot.covariance),
    }
}

fn row_major_3(m: &Matrix3<f64>) -> Vec<f64> {
    (0..3)
        .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
        .collect()
}

fn row_major_4(m: &Matrix4<f64>) -> Vec<f64> {
    (0..4)
        .flat_map(|r| (0..4).map(move |c| m[(r, c)]))
        .collect()
}

fn row_major_6(m: &SMatrix<f64, 6, 6>) -> Vec<f64> {
    (0..6)
        .flat_map(|r| (0..6).map(move |c| m[(r, c)]))
        .collect()
}

fn row_major_8(m: &SMatrix<f64, 8, 8>) -> Vec<f64> {
    (0..8)
        .flat_map(|r| (0..8).map(move |c| m[(r, c)]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bbox2d_record_round_trip() {
        let batch = ObservationBatch2D {
            sensor_id: SensorId::new(3),
            sequence: 7,
            timestamp: Timestamp::from_nanos(123),
            frame_id: FrameId::new("camera"),
            observations: vec![BBox2DObservation {
                bbox: BBox2D::from_corners(1.0, 2.0, 11.0, 22.0),
                score: 0.8,
                class_label: Some(factor_tracker_core::ClassLabel::new(2)),
                covariance: Matrix4::<f64>::identity(),
            }],
        };
        let record = record_from_batch_2d(&batch);
        let json = serde_json::to_string(&record).unwrap();
        let parsed: ObservationRecord = serde_json::from_str(&json).unwrap();
        let rebuilt = parsed.into_batch_2d().unwrap();
        assert_eq!(rebuilt.sensor_id, batch.sensor_id);
        assert_eq!(rebuilt.sequence, batch.sequence);
        assert_eq!(rebuilt.timestamp, batch.timestamp);
        assert_eq!(rebuilt.frame_id, batch.frame_id);
        assert_eq!(rebuilt.observations.len(), 1);
        assert_relative_eq!(rebuilt.observations[0].bbox.x_max, 11.0, epsilon = 1e-12);
    }

    #[test]
    fn position3d_record_round_trip() {
        let batch = ObservationBatch3D {
            sensor_id: SensorId::new(1),
            sequence: 1,
            timestamp: Timestamp::from_nanos(0),
            frame_id: FrameId::new("world"),
            observations: vec![Position3DObservation {
                position: nalgebra::Vector3::new(1.0, 2.0, 3.0),
                covariance: Matrix3::<f64>::identity(),
                class_label: None,
                appearance: None,
                score: None,
            }],
        };
        let record = record_from_batch_3d(&batch);
        let json = serde_json::to_string(&record).unwrap();
        let parsed: ObservationRecord = serde_json::from_str(&json).unwrap();
        let rebuilt = parsed.into_batch_3d().unwrap();
        assert_relative_eq!(rebuilt.observations[0].position.z, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn schema_version_mismatch_is_rejected() {
        let json = r#"{"type":"bbox2d_batch","schema_version":99,"sensor_id":1,"sequence":0,"timestamp_ns":0,"frame_id":"camera","observations":[]}"#;
        let record: ObservationRecord = serde_json::from_str(json).unwrap();
        assert!(record.into_batch_2d().is_err());
    }

    #[test]
    fn wrong_record_kind_for_tracker_is_rejected() {
        let json = r#"{"type":"position3d_batch","schema_version":1,"sensor_id":1,"sequence":0,"timestamp_ns":0,"frame_id":"world","observations":[]}"#;
        let record: ObservationRecord = serde_json::from_str(json).unwrap();
        assert!(record.into_batch_2d().is_err());
    }
}
