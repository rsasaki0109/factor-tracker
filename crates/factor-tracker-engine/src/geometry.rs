//! Calibration, ego-pose interpolation, projection and 3D box shape metadata.
//!
//! # Frames and transform direction
//!
//! * **sensor frame** — the frame a detection is expressed in (LiDAR or camera
//!   optical frame).
//! * **reference frame** — the vehicle body frame at the reference time. The
//!   static calibrations [`Calibration::lidar_to_reference`] and
//!   [`Calibration::camera_to_reference`] map sensor coordinates into it:
//!   `p_reference = R p_sensor + t`.
//! * **tracking frame** — the frame the core tracker operates in. When no ego
//!   pose is supplied it *is* the reference frame. When ego poses are supplied
//!   ([`PoseTrack`]) the tracking frame is the world frame and each
//!   measurement at time `t` is mapped by `pose(t) ∘ sensor_to_reference`.
//!
//! Poses are never extrapolated. [`PoseTrack::pose_at`] returns an error when
//! the requested time falls outside the recorded range or inside a gap larger
//! than the configured maximum, so a stale pose can never be substituted for a
//! missing one.
//!
//! The camera model is a rectified pinhole. Distortion is **not** handled; the
//! caller must supply rectified images or pre-undistorted detections.
//! `KITTI` rectified cameras satisfy this.

use factor_tracker_core::{BBox2D, FrameId, RigidTransform3D, Timestamp};
use nalgebra::{Matrix3, Matrix4, UnitQuaternion, Vector2, Vector3};

use crate::error::{EngineError, EngineResult};

/// Pinhole camera intrinsics for a rectified image.
///
/// Projection uses the camera optical convention: `x` right, `y` down, `z`
/// forward; `u = fx*x/z + skew*y/z + cx`, `v = fy*y/z + cy`.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in pixels along x.
    pub fx: f64,
    /// Focal length in pixels along y.
    pub fy: f64,
    /// Principal point x in pixels.
    pub cx: f64,
    /// Principal point y in pixels.
    pub cy: f64,
    /// Skew (usually 0 for rectified cameras).
    pub skew: f64,
    /// Image width in pixels.
    pub image_width: u32,
    /// Image height in pixels.
    pub image_height: u32,
}

impl CameraIntrinsics {
    /// Validates the intrinsics.
    pub fn validate(&self) -> EngineResult<()> {
        if !self.fx.is_finite() || self.fx <= 0.0 {
            return Err(EngineError::config("fx must be finite and > 0"));
        }
        if !self.fy.is_finite() || self.fy <= 0.0 {
            return Err(EngineError::config("fy must be finite and > 0"));
        }
        if !self.cx.is_finite() || !self.cy.is_finite() || !self.skew.is_finite() {
            return Err(EngineError::config("principal point / skew must be finite"));
        }
        if self.image_width == 0 || self.image_height == 0 {
            return Err(EngineError::config("image size must be > 0"));
        }
        Ok(())
    }

    /// The `3x3` intrinsic matrix `K`.
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.fx, self.skew, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0,
        )
    }

    /// Projects a point in the camera frame. Returns `None` when `z <= 0`.
    pub fn project(&self, point_camera: &Vector3<f64>) -> Option<Vector2<f64>> {
        if point_camera.z <= 0.0 || !point_camera.iter().all(|v| v.is_finite()) {
            return None;
        }
        let inv_z = 1.0 / point_camera.z;
        Some(Vector2::new(
            self.fx * point_camera.x * inv_z + self.skew * point_camera.y * inv_z + self.cx,
            self.fy * point_camera.y * inv_z + self.cy,
        ))
    }
}

/// Static sensor-to-reference calibration and camera model.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Name of the tracking frame.
    pub tracking_frame: FrameId,
    /// Name of the LiDAR sensor frame.
    pub lidar_frame: FrameId,
    /// Name of the camera optical frame.
    pub camera_frame: FrameId,
    /// LiDAR sensor frame to reference frame.
    pub lidar_to_reference: RigidTransform3D,
    /// Camera optical frame to reference frame.
    pub camera_to_reference: RigidTransform3D,
    /// Rectified pinhole intrinsics.
    pub intrinsics: CameraIntrinsics,
    /// Near plane in metres; geometry closer than this is rejected.
    pub near_plane: f64,
}

impl Calibration {
    /// Validates the calibration.
    pub fn validate(&self) -> EngineResult<()> {
        self.intrinsics.validate()?;
        if !self.near_plane.is_finite() || self.near_plane <= 0.0 {
            return Err(EngineError::config("near_plane must be finite and > 0"));
        }
        if self.lidar_frame == self.camera_frame {
            return Err(EngineError::config(
                "lidar_frame and camera_frame must differ",
            ));
        }
        Ok(())
    }
}

/// Returns `a ∘ b` (apply `b` first, then `a`).
pub fn compose(a: &RigidTransform3D, b: &RigidTransform3D) -> RigidTransform3D {
    let rotation = a.rotation() * b.rotation();
    let translation = a.rotation() * b.translation() + a.translation();
    RigidTransform3D::new(rotation, translation).expect("composition of valid rigid transforms")
}

/// Returns the inverse of a rigid transform.
pub fn inverse(transform: &RigidTransform3D) -> RigidTransform3D {
    let rotation = transform.rotation().transpose();
    let translation = -(rotation * transform.translation());
    RigidTransform3D::new(rotation, translation).expect("inverse of a valid rigid transform")
}

/// Orthonormalizes a rotation matrix (polar decomposition) and forces a
/// right-handed frame. Used for calibration matrices read from text files,
/// whose floating point values are only approximately orthonormal.
pub fn orthonormalize_rotation(rotation: &Matrix3<f64>) -> EngineResult<Matrix3<f64>> {
    if !rotation.iter().all(|v| v.is_finite()) {
        return Err(EngineError::config(
            "rotation matrix contains non-finite values",
        ));
    }
    let svd = rotation.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| EngineError::config("rotation decomposition failed"))?;
    let vt = svd
        .v_t
        .ok_or_else(|| EngineError::config("rotation decomposition failed"))?;
    let mut result = u * vt;
    if result.determinant() < 0.0 {
        let mut u_flipped = u;
        u_flipped.column_mut(2).scale_mut(-1.0);
        result = u_flipped * vt;
    }
    Ok(result)
}

/// Builds a validated rigid transform from a `4x4` matrix whose top-left `3x3`
/// block is an (approximately) orthonormal rotation.
pub fn rigid_from_matrix4(matrix: &Matrix4<f64>) -> EngineResult<RigidTransform3D> {
    if !matrix.iter().all(|v| v.is_finite()) {
        return Err(EngineError::config(
            "transform matrix contains non-finite values",
        ));
    }
    let rotation = matrix.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]);
    let rotation = orthonormalize_rotation(&rotation)?;
    RigidTransform3D::new(rotation, translation)
        .map_err(|error| EngineError::config(format!("invalid rigid transform: {error}")))
}

/// Builds a validated rigid transform from a `3x4` row-major slice
/// (`[R | t]`), as used by the KITTI calibration files.
pub fn rigid_from_row_major_3x4(values: &[f64]) -> EngineResult<RigidTransform3D> {
    if values.len() != 12 {
        return Err(EngineError::config(
            "expected 12 values for a 3x4 transform",
        ));
    }
    let rotation = Matrix3::new(
        values[0], values[1], values[2], values[4], values[5], values[6], values[8], values[9],
        values[10],
    );
    let translation = Vector3::new(values[3], values[7], values[11]);
    let rotation = orthonormalize_rotation(&rotation)?;
    RigidTransform3D::new(rotation, translation)
        .map_err(|error| EngineError::config(format!("invalid rigid transform: {error}")))
}

/// A time-indexed track of ego poses (reference frame to world frame).
#[derive(Debug, Clone)]
pub struct PoseTrack {
    samples: Vec<(Timestamp, RigidTransform3D)>,
    max_gap_ns: i64,
}

impl PoseTrack {
    /// Creates a pose track from timestamped poses.
    ///
    /// Samples are sorted by timestamp; duplicate timestamps are rejected.
    /// `max_gap_ns` bounds the interpolation gap and must be positive.
    pub fn new(
        mut samples: Vec<(Timestamp, RigidTransform3D)>,
        max_gap_ns: i64,
    ) -> EngineResult<Self> {
        if max_gap_ns <= 0 {
            return Err(EngineError::config("max_gap_ns must be > 0"));
        }
        samples.sort_by_key(|(timestamp, _)| timestamp.as_nanos());
        for window in samples.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(EngineError::config("duplicate pose timestamp"));
            }
        }
        Ok(PoseTrack {
            samples,
            max_gap_ns,
        })
    }

    /// Number of pose samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// The recorded samples, sorted by timestamp.
    pub fn samples(&self) -> &[(Timestamp, RigidTransform3D)] {
        &self.samples
    }

    /// Returns `true` when there are no pose samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// First recorded timestamp.
    pub fn first_timestamp(&self) -> Option<Timestamp> {
        self.samples.first().map(|(timestamp, _)| *timestamp)
    }

    /// Last recorded timestamp.
    pub fn last_timestamp(&self) -> Option<Timestamp> {
        self.samples.last().map(|(timestamp, _)| *timestamp)
    }

    /// Returns the interpolated pose at `timestamp`.
    ///
    /// No extrapolation is performed: times outside `[first, last]` or inside a
    /// gap larger than `max_gap_ns` yield [`EngineError::PoseUnavailable`].
    pub fn pose_at(&self, timestamp: Timestamp) -> EngineResult<RigidTransform3D> {
        if self.samples.is_empty() {
            return Err(EngineError::PoseUnavailable {
                reason: "pose track is empty".to_string(),
            });
        }
        if timestamp < self.samples[0].0 || timestamp > self.samples[self.samples.len() - 1].0 {
            return Err(EngineError::PoseUnavailable {
                reason: format!(
                    "timestamp {} outside pose range [{}, {}] (no extrapolation)",
                    timestamp.as_nanos(),
                    self.samples[0].0.as_nanos(),
                    self.samples[self.samples.len() - 1].0.as_nanos()
                ),
            });
        }
        let index = match self
            .samples
            .binary_search_by_key(&timestamp.as_nanos(), |(t, _)| t.as_nanos())
        {
            Ok(exact) => return Ok(self.samples[exact].1.clone()),
            Err(insert) => insert,
        };
        // `index` is the first sample strictly greater than `timestamp`.
        let (t0, p0) = &self.samples[index - 1];
        let (t1, p1) = &self.samples[index];
        let gap = t1
            .checked_duration_since(*t0)
            .map_err(|_| EngineError::PoseUnavailable {
                reason: "pose timestamp difference overflowed".to_string(),
            })?;
        if gap > self.max_gap_ns {
            return Err(EngineError::PoseUnavailable {
                reason: format!("pose gap {gap} ns exceeds max_gap_ns {}", self.max_gap_ns),
            });
        }
        let elapsed =
            timestamp
                .checked_duration_since(*t0)
                .map_err(|_| EngineError::PoseUnavailable {
                    reason: "pose timestamp difference overflowed".to_string(),
                })?;
        let s = elapsed as f64 / gap as f64;
        interpolate_pose(p0, p1, s)
    }
}

fn interpolate_pose(
    a: &RigidTransform3D,
    b: &RigidTransform3D,
    s: f64,
) -> EngineResult<RigidTransform3D> {
    let translation = a.translation() + (b.translation() - a.translation()) * s;
    // `from_rotation_matrix` can return a quaternion whose norm is off by
    // ~1e-9, which is enough to fail the core's orthonormality check after
    // conversion back; renormalize and re-orthonormalize instead of trusting it.
    let quaternion = |r: &Matrix3<f64>| {
        UnitQuaternion::new_normalize(
            UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(*r))
                .into_inner(),
        )
    };
    let q = UnitQuaternion::new_normalize(
        quaternion(a.rotation())
            .slerp(&quaternion(b.rotation()), s)
            .into_inner(),
    );
    let rotation = orthonormalize_rotation(&q.to_rotation_matrix().into_inner())?;
    RigidTransform3D::new(rotation, translation).map_err(|error| EngineError::PoseUnavailable {
        reason: format!("interpolated pose is invalid: {error}"),
    })
}

/// Box dimensions in metres, KITTI ordering (`height`, `width`, `length`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxDimensions {
    /// Height (KITTI `h`).
    pub height: f64,
    /// Width (KITTI `w`).
    pub width: f64,
    /// Length (KITTI `l`).
    pub length: f64,
}

impl BoxDimensions {
    /// Validates the dimensions.
    pub fn validate(&self) -> EngineResult<()> {
        if !self.height.is_finite()
            || !self.width.is_finite()
            || !self.length.is_finite()
            || self.height <= 0.0
            || self.width <= 0.0
            || self.length <= 0.0
        {
            return Err(EngineError::data("box dimensions must be finite and > 0"));
        }
        Ok(())
    }
}

/// Projection-only box shape metadata attached to a track.
///
/// The eight corners are stored as offsets from the object center, expressed in
/// the tracking frame at capture time. Orientation and dimensions are **not**
/// part of the tracked state; they are metadata used for camera projection.
#[derive(Debug, Clone, PartialEq)]
pub struct Box3DShape {
    /// Dimensions.
    pub dimensions: BoxDimensions,
    /// Corner offsets in the tracking frame relative to the object center.
    pub corner_offsets: [Vector3<f64>; 8],
}

impl Box3DShape {
    /// Builds a shape from a KITTI-style box in the sensor frame.
    ///
    /// `rotation_y` is the KITTI yaw about the sensor `y` axis. The result
    /// stores corner offsets relative to the object center, expressed in the
    /// **same sensor frame**. Use [`Box3DShape::transformed`] to rotate the
    /// offsets into another frame.
    pub fn from_kitti_camera(dimensions: BoxDimensions, rotation_y: f64) -> EngineResult<Self> {
        dimensions.validate()?;
        if !rotation_y.is_finite() {
            return Err(EngineError::data("rotation_y must be finite"));
        }
        let (l, w, h) = (dimensions.length, dimensions.width, dimensions.height);
        let x_corners = [
            l / 2.0,
            l / 2.0,
            -l / 2.0,
            -l / 2.0,
            l / 2.0,
            l / 2.0,
            -l / 2.0,
            -l / 2.0,
        ];
        let y_corners = [0.0, 0.0, 0.0, 0.0, -h, -h, -h, -h];
        let z_corners = [
            w / 2.0,
            -w / 2.0,
            -w / 2.0,
            w / 2.0,
            w / 2.0,
            -w / 2.0,
            -w / 2.0,
            w / 2.0,
        ];
        let (sin, cos) = rotation_y.sin_cos();
        let mut corner_offsets = [Vector3::zeros(); 8];
        for i in 0..8 {
            // Rotate about the sensor y axis.
            let x = cos * x_corners[i] + sin * z_corners[i];
            let y = y_corners[i];
            let z = -sin * x_corners[i] + cos * z_corners[i];
            corner_offsets[i] = Vector3::new(x, y, z);
        }
        Ok(Box3DShape {
            dimensions,
            corner_offsets,
        })
    }

    /// Builds an axis-aligned box (no yaw) centered on the object.
    ///
    /// The eight offsets are the `(±length/2, ±width/2, ±height/2)`
    /// combinations in the frame's own axes.
    pub fn axis_aligned(dimensions: BoxDimensions) -> EngineResult<Self> {
        dimensions.validate()?;
        let (hl, hw, hh) = (
            dimensions.length / 2.0,
            dimensions.width / 2.0,
            dimensions.height / 2.0,
        );
        let signs = [-1.0, 1.0];
        let mut corner_offsets = [Vector3::zeros(); 8];
        let mut index = 0;
        for sx in signs {
            for sy in signs {
                for sz in signs {
                    corner_offsets[index] = Vector3::new(sx * hl, sy * hw, sz * hh);
                    index += 1;
                }
            }
        }
        Ok(Box3DShape {
            dimensions,
            corner_offsets,
        })
    }

    /// Builds a shape from explicit corner offsets (relative to the object
    /// center) in the sensor frame.
    pub fn from_corner_offsets(
        dimensions: BoxDimensions,
        offsets: [Vector3<f64>; 8],
    ) -> EngineResult<Self> {
        dimensions.validate()?;
        if !offsets.iter().all(|o| o.iter().all(|v| v.is_finite())) {
            return Err(EngineError::data("corner offsets must be finite"));
        }
        Ok(Box3DShape {
            dimensions,
            corner_offsets: offsets,
        })
    }

    /// Flattened corner offsets (8 corners x 3 components).
    pub fn flattened_offsets(&self) -> Vec<f64> {
        self.corner_offsets
            .iter()
            .flat_map(|offset| offset.iter().copied())
            .collect()
    }

    /// Rotates the corner offsets into another frame (no translation, since
    /// offsets are relative to the object center).
    pub fn transformed(&self, rotation: &Matrix3<f64>) -> Box3DShape {
        let mut corner_offsets = [Vector3::zeros(); 8];
        for (i, offset) in self.corner_offsets.iter().enumerate() {
            corner_offsets[i] = rotation * offset;
        }
        Box3DShape {
            dimensions: self.dimensions,
            corner_offsets,
        }
    }

    /// Returns the eight corners for a given center.
    pub fn corners_at(&self, center: &Vector3<f64>) -> [Vector3<f64>; 8] {
        let mut corners = [Vector3::zeros(); 8];
        for (i, offset) in self.corner_offsets.iter().enumerate() {
            corners[i] = center + offset;
        }
        corners
    }
}

/// Why a projected box was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionRejection {
    /// Every corner is behind the camera.
    BehindCamera,
    /// The box straddles the near plane, so its projection is not well defined.
    StraddlesNearPlane,
    /// The projected box does not overlap the image.
    OutOfView,
    /// The shape had no corners (defensive; should not occur).
    EmptyShape,
}

impl ProjectionRejection {
    /// Stable lowercase name for diagnostics and serialization.
    pub const fn as_str(self) -> &'static str {
        match self {
            ProjectionRejection::BehindCamera => "behind_camera",
            ProjectionRejection::StraddlesNearPlane => "straddles_near_plane",
            ProjectionRejection::OutOfView => "out_of_view",
            ProjectionRejection::EmptyShape => "empty_shape",
        }
    }
}

/// A successfully projected box.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedBox {
    /// Axis-aligned image box (clipped to the image bounds).
    pub bbox: BBox2D,
    /// Whether the unclipped projection extended outside the image.
    pub clipped: bool,
    /// Minimum camera-frame depth among the corners, in metres.
    pub min_depth: f64,
}

/// Projects eight camera-frame corners to an axis-aligned image box.
///
/// Returns a [`ProjectionRejection`] when the box is behind the camera,
/// straddles the near plane, or falls entirely outside the image. The unclipped
/// projection is clipped to the image before being returned.
pub fn project_box_corners(
    intrinsics: &CameraIntrinsics,
    near_plane: f64,
    corners_camera: &[Vector3<f64>; 8],
) -> Result<ProjectedBox, ProjectionRejection> {
    if corners_camera.is_empty() {
        return Err(ProjectionRejection::EmptyShape);
    }
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for corner in corners_camera {
        if !corner.iter().all(|v| v.is_finite()) {
            return Err(ProjectionRejection::EmptyShape);
        }
        min_z = min_z.min(corner.z);
        max_z = max_z.max(corner.z);
    }
    if max_z <= near_plane {
        return if max_z <= 0.0 {
            Err(ProjectionRejection::BehindCamera)
        } else {
            Err(ProjectionRejection::StraddlesNearPlane)
        };
    }
    if min_z <= near_plane {
        return Err(ProjectionRejection::StraddlesNearPlane);
    }

    let mut min_u = f64::INFINITY;
    let mut min_v = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for corner in corners_camera {
        let projected = intrinsics
            .project(corner)
            .ok_or(ProjectionRejection::StraddlesNearPlane)?;
        min_u = min_u.min(projected.x);
        min_v = min_v.min(projected.y);
        max_u = max_u.max(projected.x);
        max_v = max_v.max(projected.y);
    }

    let width = intrinsics.image_width as f64;
    let height = intrinsics.image_height as f64;
    let clipped = min_u < 0.0 || min_v < 0.0 || max_u > width || max_v > height;
    let clipped_min_u = min_u.max(0.0);
    let clipped_min_v = min_v.max(0.0);
    let clipped_max_u = max_u.min(width);
    let clipped_max_v = max_v.min(height);
    if clipped_min_u >= clipped_max_u || clipped_min_v >= clipped_max_v {
        return Err(ProjectionRejection::OutOfView);
    }
    Ok(ProjectedBox {
        bbox: BBox2D::from_corners(clipped_min_u, clipped_min_v, clipped_max_u, clipped_max_v),
        clipped,
        min_depth: min_z,
    })
}
