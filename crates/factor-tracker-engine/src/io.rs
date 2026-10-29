//! JSONL I/O for engine inputs, outputs and evaluation ground truth.
//!
//! See `docs/camera-lidar.md` for the schema. Every record carries a
//! `schema_version` (currently `1`).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use factor_tracker_core::{BBox2D, ClassLabel, FrameId, SensorId, Timestamp};
use nalgebra::{Matrix3, Vector2, Vector3};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::engine::{CameraFrameOutput, LidarFrameOutput};
use crate::error::{EngineError, EngineResult};
use crate::eval::{
    CameraAssociationTruth, CameraDetectionTruth, GroundTruthFrame, GroundTruthObject,
};
use crate::geometry::{Box3DShape, BoxDimensions};
use crate::input::{CameraBatch, CameraDetection, EngineBatch, LidarBatch, LidarDetection};

/// Current engine schema version.
pub const ENGINE_SCHEMA_VERSION: u32 = 1;

/// A shape metadata record (projection-only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeMetadataRecord {
    /// Height in metres.
    pub height: f64,
    /// Width in metres.
    pub width: f64,
    /// Length in metres.
    pub length: f64,
    /// Yaw about the sensor `y` axis, radians.
    pub rotation_y: f64,
    /// Optional explicit corner offsets (8 corners x 3 components) relative to
    /// the object center. When present these take precedence over
    /// `dimensions` + `rotation_y`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corners: Option<Vec<f64>>,
}

/// A LiDAR detection record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LidarDetectionRecord {
    /// Position in the LiDAR frame, metres.
    pub position: Vec<f64>,
    /// Row-major `3x3` covariance, metres².
    pub covariance: Vec<f64>,
    /// Class in the tracking class space.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<u32>,
    /// Detector confidence.
    pub score: f64,
    /// Optional projection shape metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<ShapeMetadataRecord>,
    /// Optional appearance embedding (any length; compared by cosine).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub appearance: Option<Vec<f32>>,
    /// Optional measured velocity over ground `[vx, vy]` (m/s, LiDAR frame
    /// axes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub velocity: Option<Vec<f64>>,
}

/// A static landmark (pole, tree, bin, ...) seen by the LiDAR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandmarkRecord {
    /// Position in the LiDAR frame, metres.
    pub position: Vec<f64>,
}

/// A camera detection record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraDetectionRecord {
    /// Detection id.
    pub detection_id: u64,
    /// `[x_min, y_min, x_max, y_max]` in pixels.
    pub bbox: Vec<f64>,
    /// Class in the camera class space.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<u32>,
    /// Detector confidence.
    pub score: f64,
}

/// An engine input record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EngineInputRecord {
    /// A LiDAR batch.
    #[serde(rename = "lidar_batch")]
    Lidar {
        /// Schema version.
        schema_version: u32,
        /// Sensor id.
        sensor_id: u32,
        /// Sequence number.
        sequence: u64,
        /// Event time in nanoseconds.
        event_time_ns: i64,
        /// Arrival time in nanoseconds.
        arrival_time_ns: i64,
        /// Sensor frame.
        frame_id: String,
        /// Detections.
        detections: Vec<LidarDetectionRecord>,
        /// Static landmarks in the LiDAR frame (ego-pose correction only).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        landmarks: Vec<LandmarkRecord>,
    },
    /// A camera batch.
    #[serde(rename = "camera_batch")]
    Camera {
        /// Schema version.
        schema_version: u32,
        /// Sensor id.
        sensor_id: u32,
        /// Sequence number.
        sequence: u64,
        /// Event time in nanoseconds.
        event_time_ns: i64,
        /// Arrival time in nanoseconds.
        arrival_time_ns: i64,
        /// Camera frame.
        frame_id: String,
        /// Detections.
        detections: Vec<CameraDetectionRecord>,
    },
}

/// An evaluation ground-truth record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TruthRecord {
    /// 3D ground truth.
    #[serde(rename = "truth3d")]
    Truth3d {
        /// Schema version.
        schema_version: u32,
        /// Timestamp.
        timestamp_ns: i64,
        /// Objects.
        objects: Vec<GroundTruthObject>,
    },
    /// Camera association ground truth (evaluation only).
    #[serde(rename = "camera_association_truth")]
    CameraAssociation {
        /// Schema version.
        schema_version: u32,
        /// Camera timestamp.
        timestamp_ns: i64,
        /// Detection to truth mappings.
        entries: Vec<CameraDetectionTruth>,
    },
}

/// An engine output record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EngineOutputRecord {
    /// A processed LiDAR frame.
    #[serde(rename = "lidar_frame")]
    Lidar(LidarFrameOutput),
    /// A processed camera frame.
    #[serde(rename = "camera_frame")]
    Camera(CameraFrameOutput),
}

fn check_version(version: u32) -> EngineResult<()> {
    if version != ENGINE_SCHEMA_VERSION {
        return Err(EngineError::data(format!(
            "unsupported schema_version {version} (expected {ENGINE_SCHEMA_VERSION})"
        )));
    }
    Ok(())
}

fn vector<const N: usize>(context: &str, values: &[f64]) -> EngineResult<[f64; N]> {
    if values.len() != N {
        return Err(EngineError::data(format!(
            "{context}: expected {N} values, found {}",
            values.len()
        )));
    }
    let mut out = [0.0; N];
    out.copy_from_slice(values);
    Ok(out)
}

fn corner_offsets_from_slice(values: &[f64]) -> EngineResult<[Vector3<f64>; 8]> {
    if values.len() != 24 {
        return Err(EngineError::data(
            "shape corners must have 24 values (8 corners x 3)",
        ));
    }
    let mut offsets = [Vector3::zeros(); 8];
    for (i, offset) in offsets.iter_mut().enumerate() {
        *offset = Vector3::new(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
    }
    Ok(offsets)
}

fn matrix3(values: &[f64]) -> EngineResult<Matrix3<f64>> {
    if values.len() != 9 {
        return Err(EngineError::data("covariance must have 9 values"));
    }
    Ok(Matrix3::new(
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        values[8],
    ))
}

impl EngineInputRecord {
    /// Converts the record into an engine batch plus arrival time.
    pub fn into_batch(self) -> EngineResult<(EngineBatch, Timestamp)> {
        match self {
            EngineInputRecord::Lidar {
                schema_version,
                sensor_id,
                sequence,
                event_time_ns,
                arrival_time_ns,
                frame_id,
                detections,
                landmarks,
            } => {
                check_version(schema_version)?;
                let landmarks = landmarks
                    .iter()
                    .enumerate()
                    .map(|(index, record)| {
                        vector::<3>(&format!("landmark {index} position"), &record.position)
                            .map(Vector3::from)
                    })
                    .collect::<EngineResult<Vec<_>>>()?;
                let detections = detections
                    .into_iter()
                    .enumerate()
                    .map(|(index, record)| {
                        let position = Vector3::from(vector::<3>(
                            &format!("detection {index} position"),
                            &record.position,
                        )?);
                        let covariance = matrix3(&record.covariance)?;
                        let shape = match record.shape {
                            Some(shape) => {
                                let dimensions = BoxDimensions {
                                    height: shape.height,
                                    width: shape.width,
                                    length: shape.length,
                                };
                                match shape.corners {
                                    Some(corners) => Some(Box3DShape::from_corner_offsets(
                                        dimensions,
                                        corner_offsets_from_slice(&corners)?,
                                    )?),
                                    None => Some(Box3DShape::from_kitti_camera(
                                        dimensions,
                                        shape.rotation_y,
                                    )?),
                                }
                            }
                            None => None,
                        };
                        let velocity = match &record.velocity {
                            Some(v) => Some(Vector2::from(vector::<2>(
                                &format!("detection {index} velocity"),
                                v,
                            )?)),
                            None => None,
                        };
                        Ok(LidarDetection {
                            position,
                            covariance,
                            class_label: record.class.map(ClassLabel::new),
                            score: record.score,
                            shape,
                            appearance: record.appearance.clone(),
                            velocity,
                        })
                    })
                    .collect::<EngineResult<Vec<_>>>()?;
                Ok((
                    EngineBatch::Lidar(LidarBatch {
                        sensor_id: SensorId::new(sensor_id),
                        sequence,
                        event_time: Timestamp::from_nanos(event_time_ns),
                        frame_id: FrameId::new(frame_id),
                        detections,
                        landmarks,
                    }),
                    Timestamp::from_nanos(arrival_time_ns),
                ))
            }
            EngineInputRecord::Camera {
                schema_version,
                sensor_id,
                sequence,
                event_time_ns,
                arrival_time_ns,
                frame_id,
                detections,
            } => {
                check_version(schema_version)?;
                let detections = detections
                    .into_iter()
                    .enumerate()
                    .map(|(index, record)| {
                        let bbox = vector::<4>(&format!("detection {index} bbox"), &record.bbox)?;
                        if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
                            return Err(EngineError::data(format!(
                                "detection {index} bbox has non-positive extent"
                            )));
                        }
                        Ok(CameraDetection {
                            detection_id: record.detection_id,
                            bbox: BBox2D::from_corners(bbox[0], bbox[1], bbox[2], bbox[3]),
                            class_label: record.class.map(ClassLabel::new),
                            score: record.score,
                        })
                    })
                    .collect::<EngineResult<Vec<_>>>()?;
                Ok((
                    EngineBatch::Camera(CameraBatch {
                        sensor_id: SensorId::new(sensor_id),
                        sequence,
                        event_time: Timestamp::from_nanos(event_time_ns),
                        frame_id: FrameId::new(frame_id),
                        detections,
                    }),
                    Timestamp::from_nanos(arrival_time_ns),
                ))
            }
        }
    }
}

/// Reads engine input records and converts them to batches.
pub fn read_inputs(path: &Path) -> EngineResult<Vec<(EngineBatch, Timestamp)>> {
    let records: Vec<EngineInputRecord> = read_jsonl(path)?;
    records
        .into_iter()
        .map(EngineInputRecord::into_batch)
        .collect()
}

/// Reads truth records, splitting them into 3D truth and camera association
/// truth.
pub fn read_truth(
    path: &Path,
) -> EngineResult<(Vec<GroundTruthFrame>, Vec<CameraAssociationTruth>)> {
    let records: Vec<TruthRecord> = read_jsonl(path)?;
    let mut truth = Vec::new();
    let mut camera_truth = Vec::new();
    for record in records {
        match record {
            TruthRecord::Truth3d {
                schema_version,
                timestamp_ns,
                objects,
            } => {
                check_version(schema_version)?;
                truth.push(GroundTruthFrame {
                    timestamp_ns,
                    objects,
                });
            }
            TruthRecord::CameraAssociation {
                schema_version,
                timestamp_ns,
                entries,
            } => {
                check_version(schema_version)?;
                camera_truth.push(CameraAssociationTruth {
                    timestamp_ns,
                    entries,
                });
            }
        }
    }
    Ok((truth, camera_truth))
}

/// Reads JSONL records, skipping blank lines.
pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> EngineResult<Vec<T>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<T>(&line).map_err(|error| {
            EngineError::data(format!("{}:{}: {error}", path.display(), line_number + 1))
        })?;
        records.push(record);
    }
    Ok(records)
}

/// Writes JSONL records.
pub fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> EngineResult<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

/// Writes engine output records.
pub fn write_outputs(
    path: &Path,
    lidar_frames: &[LidarFrameOutput],
    camera_frames: &[CameraFrameOutput],
) -> EngineResult<()> {
    let mut records: Vec<EngineOutputRecord> = lidar_frames
        .iter()
        .cloned()
        .map(EngineOutputRecord::Lidar)
        .chain(
            camera_frames
                .iter()
                .cloned()
                .map(EngineOutputRecord::Camera),
        )
        .collect();
    // Stable sort keeps LiDAR before camera at equal timestamps.
    records.sort_by_key(EngineOutputRecord::timestamp_ns);
    write_jsonl(path, &records)
}

impl EngineOutputRecord {
    /// Timestamp of the record.
    pub fn timestamp_ns(&self) -> i64 {
        match self {
            EngineOutputRecord::Lidar(frame) => frame.timestamp_ns,
            EngineOutputRecord::Camera(frame) => frame.timestamp_ns,
        }
    }
}
