//! A serializable description of the calibration and optional ego poses.
//!
//! This lets a replay be configured from a JSON file without hard-coding a
//! dataset layout. Matrices are row-major.

use factor_tracker_core::{FrameId, RigidTransform3D, Timestamp};
use nalgebra::Matrix4;
use serde::{Deserialize, Serialize};

use crate::error::{EngineError, EngineResult};
use crate::geometry::{rigid_from_matrix4, Calibration, CameraIntrinsics, PoseTrack};

/// Camera intrinsics in a setup file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntrinsicsSetup {
    /// Focal length x.
    pub fx: f64,
    /// Focal length y.
    pub fy: f64,
    /// Principal point x.
    pub cx: f64,
    /// Principal point y.
    pub cy: f64,
    /// Skew.
    pub skew: f64,
    /// Image width.
    pub image_width: u32,
    /// Image height.
    pub image_height: u32,
}

/// A timestamped transform in a setup file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseSetup {
    /// Timestamp.
    pub timestamp_ns: i64,
    /// Row-major `4x4` transform.
    pub transform: Vec<f64>,
}

/// A complete engine setup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineSetup {
    /// Tracking frame name.
    pub tracking_frame: String,
    /// LiDAR frame name.
    pub lidar_frame: String,
    /// Camera frame name.
    pub camera_frame: String,
    /// LiDAR-to-reference row-major `4x4`.
    pub lidar_to_reference: Vec<f64>,
    /// Camera-to-reference row-major `4x4`.
    pub camera_to_reference: Vec<f64>,
    /// Camera intrinsics.
    pub intrinsics: IntrinsicsSetup,
    /// Near plane in metres.
    pub near_plane: f64,
    /// Optional pose samples (reference to world).
    #[serde(default)]
    pub poses: Option<Vec<PoseSetup>>,
    /// Maximum pose interpolation gap in nanoseconds.
    #[serde(default = "default_pose_gap")]
    pub pose_max_gap_ns: i64,
}

fn default_pose_gap() -> i64 {
    200_000_000
}

impl EngineSetup {
    /// Builds the engine [`Calibration`].
    pub fn to_calibration(&self) -> EngineResult<Calibration> {
        let lidar_to_reference = rigid_from_vec16(&self.lidar_to_reference, "lidar_to_reference")?;
        let camera_to_reference =
            rigid_from_vec16(&self.camera_to_reference, "camera_to_reference")?;
        Ok(Calibration {
            tracking_frame: FrameId::new(self.tracking_frame.clone()),
            lidar_frame: FrameId::new(self.lidar_frame.clone()),
            camera_frame: FrameId::new(self.camera_frame.clone()),
            lidar_to_reference,
            camera_to_reference,
            intrinsics: CameraIntrinsics {
                fx: self.intrinsics.fx,
                fy: self.intrinsics.fy,
                cx: self.intrinsics.cx,
                cy: self.intrinsics.cy,
                skew: self.intrinsics.skew,
                image_width: self.intrinsics.image_width,
                image_height: self.intrinsics.image_height,
            },
            near_plane: self.near_plane,
        })
    }

    /// Builds a setup from a calibration and optional poses.
    pub fn from_calibration(
        calibration: &Calibration,
        poses: Option<&PoseTrack>,
        pose_max_gap_ns: i64,
    ) -> EngineResult<Self> {
        let pose_setup = match poses {
            Some(poses) => {
                // Recover the samples through the public accessor.
                let samples = poses.samples();
                Some(
                    samples
                        .iter()
                        .map(|(timestamp, transform)| PoseSetup {
                            timestamp_ns: timestamp.as_nanos(),
                            transform: transform_to_vec16(transform),
                        })
                        .collect(),
                )
            }
            None => None,
        };
        Ok(EngineSetup {
            tracking_frame: calibration.tracking_frame.to_string(),
            lidar_frame: calibration.lidar_frame.to_string(),
            camera_frame: calibration.camera_frame.to_string(),
            lidar_to_reference: transform_to_vec16(&calibration.lidar_to_reference),
            camera_to_reference: transform_to_vec16(&calibration.camera_to_reference),
            intrinsics: IntrinsicsSetup {
                fx: calibration.intrinsics.fx,
                fy: calibration.intrinsics.fy,
                cx: calibration.intrinsics.cx,
                cy: calibration.intrinsics.cy,
                skew: calibration.intrinsics.skew,
                image_width: calibration.intrinsics.image_width,
                image_height: calibration.intrinsics.image_height,
            },
            near_plane: calibration.near_plane,
            poses: pose_setup,
            pose_max_gap_ns,
        })
    }

    /// Builds the optional pose track.
    pub fn to_poses(&self) -> EngineResult<Option<PoseTrack>> {
        let Some(poses) = &self.poses else {
            return Ok(None);
        };
        let samples = poses
            .iter()
            .map(|pose| {
                Ok((
                    Timestamp::from_nanos(pose.timestamp_ns),
                    rigid_from_vec16(&pose.transform, "pose")?,
                ))
            })
            .collect::<EngineResult<Vec<(Timestamp, RigidTransform3D)>>>()?;
        Ok(Some(PoseTrack::new(samples, self.pose_max_gap_ns)?))
    }
}

fn rigid_from_vec16(values: &[f64], context: &str) -> EngineResult<RigidTransform3D> {
    if values.len() != 16 {
        return Err(EngineError::data(format!(
            "{context} must have 16 values, found {}",
            values.len()
        )));
    }
    let matrix = Matrix4::from_row_slice(values);
    rigid_from_matrix4(&matrix).map_err(|error| EngineError::data(format!("{context}: {error}")))
}

/// Serializes a rigid transform to a row-major `4x4` vector.
pub fn transform_to_vec16(transform: &RigidTransform3D) -> Vec<f64> {
    let r = transform.rotation();
    let t = transform.translation();
    vec![
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        t[0],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        t[1],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        t[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}
