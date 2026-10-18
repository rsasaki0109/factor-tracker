//! Adapter for the KITTI object-detection/tracking text formats.
//!
//! # Dataset choice and licence
//!
//! KITTI is chosen because camera and LiDAR share a sequence, the calibration
//! and timestamps are documented, ego poses are distributed with the odometry
//! benchmark, and the label format is plain text with a tracking-id variant.
//! The KITTI datasets themselves are **not** redistributed here and require
//! accepting the dataset terms on `cvlibs.net` before download. See
//! `docs/datasets.md`.
//!
//! # Convention used by this adapter
//!
//! KITTI 3D labels and 3D detection results are expressed in the **camera0
//! rectified** frame. This adapter re-expresses them in the **velodyne** frame,
//! which is used as the engine's reference/tracking frame:
//!
//! ```text
//! T_velo_from_cam = inverse(R_rect_00 * Tr_velo_to_cam)
//! camera_to_reference = inverse(T_velo_from_cam)   // cam0 <- velo
//! lidar_to_reference  = identity                    // already velodyne
//! ```
//!
//! Camera *detections* are 2D image boxes and are used directly. Camera
//! distortion is not modelled; KITTI rectified images are assumed.
//!
//! Ego poses from `poses.txt` are camera0-to-world; the adapter composes them
//! with `camera_to_reference` so the [`crate::geometry::PoseTrack`] maps the
//! velodyne frame to the world frame. The poses are external inputs: no
//! ego-motion estimation is performed or evaluated here.

use std::collections::BTreeMap;
use std::path::Path;

use nalgebra::{Matrix3, Matrix4, Vector3};
use perception_core::{ClassLabel, FrameId, RigidTransform3D, SensorId, Timestamp};

use crate::error::{EngineError, EngineResult};
use crate::geometry::{
    compose, inverse, Box3DShape, BoxDimensions, Calibration, CameraIntrinsics, PoseTrack,
};
use crate::input::{CameraBatch, CameraDetection, EngineBatch, LidarBatch, LidarDetection};
use crate::io::{
    CameraDetectionRecord, EngineInputRecord, LidarDetectionRecord, ShapeMetadataRecord,
    ENGINE_SCHEMA_VERSION,
};

/// Tracking frame used when ego poses are supplied.
pub const KITTI_WORLD_FRAME: &str = "world";

/// Frame id of the rectified projecting camera (`P2` / `image_02` for KITTI).
pub const KITTI_CAMERA_FRAME: &str = "cam_rect";

/// Maps KITTI class names to tracking-space ids.
#[derive(Debug, Clone)]
pub struct KittiClassMap {
    map: BTreeMap<String, u32>,
}

impl Default for KittiClassMap {
    fn default() -> Self {
        let mut map = BTreeMap::new();
        map.insert("DontCare".to_string(), 0);
        map.insert("Car".to_string(), 1);
        map.insert("Pedestrian".to_string(), 2);
        map.insert("Cyclist".to_string(), 3);
        map.insert("Van".to_string(), 4);
        map.insert("Truck".to_string(), 5);
        map.insert("Person_sitting".to_string(), 6);
        map.insert("Tram".to_string(), 7);
        map.insert("Misc".to_string(), 8);
        KittiClassMap { map }
    }
}

impl KittiClassMap {
    /// Returns the tracking class id for a KITTI class name.
    pub fn id(&self, name: &str) -> Option<u32> {
        self.map.get(name).copied()
    }
}

/// Parsed KITTI `calib.txt`.
#[derive(Debug, Clone)]
pub struct KittiCalibration {
    /// `P_rect_00` (or `P2`) as a row-major `3x4` of 12 values.
    pub p_rect: [f64; 12],
    /// `R_rect_00` as a row-major `4x4` of 16 values (a `3x3` input is padded).
    pub r_rect: [f64; 16],
    /// `Tr_velo_to_cam` as a row-major `3x4` of 12 values.
    pub tr_velo_to_cam: [f64; 12],
}

impl KittiCalibration {
    /// Parses a KITTI calibration file.
    pub fn parse(text: &str) -> EngineResult<Self> {
        let mut p_rect = None;
        let mut p_fallback = None;
        let mut r_rect = None;
        let mut tr_velo = None;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Object/raw calibration files use `key: values`; the tracking
            // benchmark files write `R_rect` and `Tr_velo_cam` without a colon.
            let (key, rest) = match line.split_once(':') {
                Some((key, rest)) => (key.trim(), rest),
                None => match line.split_once(char::is_whitespace) {
                    Some((key, rest)) => (key, rest),
                    None => continue,
                },
            };
            // Only known keys are parsed, so entries such as `calib_time` in the
            // raw-data files are ignored.
            let slot = match key {
                "P_rect_00" | "P2" => &mut p_rect,
                "P0" => &mut p_fallback,
                "R_rect_00" | "R0_rect" | "R_rect" => &mut r_rect,
                "Tr_velo_to_cam" | "Tr_velo_cam" => &mut tr_velo,
                _ => continue,
            };
            let values: Vec<f64> = rest
                .split_whitespace()
                .map(|token| {
                    token
                        .parse::<f64>()
                        .map_err(|_| EngineError::data(format!("bad number '{token}' in calib")))
                })
                .collect::<EngineResult<Vec<_>>>()?;
            *slot = Some(values);
        }
        let p_rect = p_rect.as_deref().map(array12).transpose()?;
        let p_fallback = p_fallback.as_deref().map(array12).transpose()?;
        let r_rect = r_rect.as_deref().map(rect_4x4).transpose()?;
        let tr_velo = tr_velo.as_deref().map(array12).transpose()?;
        Ok(KittiCalibration {
            p_rect: p_rect
                .or(p_fallback)
                .ok_or_else(|| EngineError::data("calib is missing P_rect_00/P2/P0"))?,
            r_rect: r_rect.ok_or_else(|| EngineError::data("calib is missing R_rect_00"))?,
            tr_velo_to_cam: tr_velo
                .ok_or_else(|| EngineError::data("calib is missing Tr_velo_to_cam"))?,
        })
    }

    /// The camera-from-velodyne transform `R_rect * Tr_velo_to_cam`.
    pub fn velo_to_cam(&self) -> EngineResult<RigidTransform3D> {
        let r = Matrix4::from_row_slice(&self.r_rect);
        let tr = Matrix4::new(
            self.tr_velo_to_cam[0],
            self.tr_velo_to_cam[1],
            self.tr_velo_to_cam[2],
            self.tr_velo_to_cam[3],
            self.tr_velo_to_cam[4],
            self.tr_velo_to_cam[5],
            self.tr_velo_to_cam[6],
            self.tr_velo_to_cam[7],
            self.tr_velo_to_cam[8],
            self.tr_velo_to_cam[9],
            self.tr_velo_to_cam[10],
            self.tr_velo_to_cam[11],
            0.0,
            0.0,
            0.0,
            1.0,
        );
        crate::geometry::rigid_from_matrix4(&(r * tr))
    }

    /// Offset of the projecting camera from rectified camera 0, expressed in
    /// the rectified frame.
    ///
    /// `P_rect = K [I | t]`, so `t = K^-1 p4` where `p4` is the fourth column.
    /// It is zero for camera 0 and about `(0.06, 0, 0)` m for KITTI `P2`.
    pub fn camera_offset(&self) -> Vector3<f64> {
        let p = &self.p_rect;
        let (fx, skew, cx, fy, cy) = (p[0], p[1], p[2], p[5], p[6]);
        let tz = p[11];
        let ty = (p[7] - cy * tz) / fy;
        let tx = (p[3] - skew * ty - cx * tz) / fx;
        Vector3::new(tx, ty, tz)
    }

    /// Builds the engine [`Calibration`] with the velodyne frame as reference.
    ///
    /// The camera frame is the rectified frame of the camera whose projection
    /// matrix was read (`P2` when present), so the translation column of the
    /// projection matrix becomes part of `camera_to_reference`. The returned
    /// transform is velodyne to rectified camera 0, the frame of KITTI 3D
    /// labels, detections and poses.
    pub fn to_calibration(
        &self,
        image_width: u32,
        image_height: u32,
        near_plane: f64,
    ) -> EngineResult<(Calibration, RigidTransform3D)> {
        let velo_to_cam = self.velo_to_cam()?;
        let cam_to_velo = inverse(&velo_to_cam);
        // X_cam0 = X_cam - t for a point X_cam in the projecting camera frame.
        let projecting_to_cam0 = RigidTransform3D::new(Matrix3::identity(), -self.camera_offset())
            .map_err(|error| EngineError::data(format!("bad camera offset: {error}")))?;
        let intrinsics = CameraIntrinsics {
            fx: self.p_rect[0],
            fy: self.p_rect[5],
            cx: self.p_rect[2],
            cy: self.p_rect[6],
            skew: self.p_rect[1],
            image_width,
            image_height,
        };
        let calibration = Calibration {
            tracking_frame: FrameId::new("velodyne"),
            lidar_frame: FrameId::new("velodyne"),
            camera_frame: FrameId::new(KITTI_CAMERA_FRAME),
            lidar_to_reference: RigidTransform3D::identity(),
            camera_to_reference: compose(&cam_to_velo, &projecting_to_cam0),
            intrinsics,
            near_plane,
        };
        Ok((calibration, velo_to_cam))
    }
}

fn array12(values: &[f64]) -> EngineResult<[f64; 12]> {
    if values.len() != 12 {
        return Err(EngineError::data("expected 12 calibration values"));
    }
    let mut out = [0.0; 12];
    out.copy_from_slice(values);
    Ok(out)
}

/// Accepts a rectifying rotation as a `3x3` (as distributed by KITTI) or an
/// already padded `4x4`, and returns the row-major `4x4`.
fn rect_4x4(values: &[f64]) -> EngineResult<[f64; 16]> {
    match values.len() {
        16 => {
            let mut out = [0.0; 16];
            out.copy_from_slice(values);
            Ok(out)
        }
        9 => {
            let mut out = [0.0; 16];
            for row in 0..3 {
                out[row * 4..row * 4 + 3].copy_from_slice(&values[row * 3..row * 3 + 3]);
            }
            out[15] = 1.0;
            Ok(out)
        }
        _ => Err(EngineError::data(
            "expected 9 or 16 values for the rectifying rotation",
        )),
    }
}

/// Parses `times.txt` (seconds per line) into timestamps.
pub fn parse_times(text: &str) -> EngineResult<Vec<Timestamp>> {
    let mut timestamps = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let seconds: f64 = line
            .parse()
            .map_err(|_| EngineError::data(format!("bad time '{line}'")))?;
        if !seconds.is_finite() {
            return Err(EngineError::data("non-finite time"));
        }
        timestamps.push(Timestamp::from_nanos((seconds * 1.0e9).round() as i64));
    }
    Ok(timestamps)
}

/// Parses `poses.txt` (one row-major `3x4` per line).
pub fn parse_poses(text: &str) -> EngineResult<Vec<RigidTransform3D>> {
    let mut poses = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|token| {
                token
                    .parse::<f64>()
                    .map_err(|_| EngineError::data(format!("bad pose value '{token}'")))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        poses.push(crate::geometry::rigid_from_row_major_3x4(&values)?);
    }
    Ok(poses)
}

/// Builds a pose track mapping the reference (velodyne) frame to world.
///
/// KITTI poses are camera0-to-world, so they are composed with the
/// velodyne-to-camera0 transform (`velo_to_cam`).
pub fn build_pose_track(
    timestamps: &[Timestamp],
    camera_poses: &[RigidTransform3D],
    velo_to_cam: &RigidTransform3D,
    max_gap_ns: i64,
) -> EngineResult<PoseTrack> {
    if timestamps.len() != camera_poses.len() {
        return Err(EngineError::data(
            "number of timestamps and poses must match",
        ));
    }
    let samples = timestamps
        .iter()
        .zip(camera_poses)
        .map(|(timestamp, pose)| (*timestamp, crate::geometry::compose(pose, velo_to_cam)))
        .collect();
    PoseTrack::new(samples, max_gap_ns)
}

/// A parsed 3D detection/label (camera frame).
#[derive(Debug, Clone, PartialEq)]
pub struct KittiObject3d {
    /// Tracking class id.
    pub class_id: Option<u32>,
    /// Dimensions `(height, width, length)` in metres.
    pub dimensions: [f64; 3],
    /// Location `(x, y, z)` in the camera0 rectified frame.
    pub location: [f64; 3],
    /// Yaw about the camera `y` axis.
    pub rotation_y: f64,
    /// Detector score (1.0 for labels).
    pub score: f64,
    /// Tracking id, for label files only.
    pub track_id: Option<u32>,
}

/// A parsed 2D detection/label (image frame).
#[derive(Debug, Clone, PartialEq)]
pub struct KittiObject2d {
    /// Tracking class id.
    pub class_id: Option<u32>,
    /// `[x_min, y_min, x_max, y_max]` in pixels.
    pub bbox: [f64; 4],
    /// Detector score (1.0 for labels).
    pub score: f64,
    /// Tracking id, for label files only.
    pub track_id: Option<u32>,
}

fn parse_f64(tokens: &[&str], index: usize, context: &str) -> EngineResult<f64> {
    tokens
        .get(index)
        .ok_or_else(|| EngineError::data(format!("{context}: missing field {index}")))?
        .parse::<f64>()
        .map_err(|_| EngineError::data(format!("{context}: bad number at field {index}")))
}

/// Parses KITTI 3D detection lines: `type trunc occ alpha x1..y2 h w l x y z ry [score]`.
pub fn parse_objects_3d(text: &str, classes: &KittiClassMap) -> EngineResult<Vec<KittiObject3d>> {
    let mut objects = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 15 {
            return Err(EngineError::data(format!(
                "3d object line {}: expected at least 15 fields, found {}",
                line_number + 1,
                tokens.len()
            )));
        }
        let class_id = classes.id(tokens[0]);
        let dimensions = [
            parse_f64(&tokens, 8, "dimensions")?,
            parse_f64(&tokens, 9, "dimensions")?,
            parse_f64(&tokens, 10, "dimensions")?,
        ];
        let location = [
            parse_f64(&tokens, 11, "location")?,
            parse_f64(&tokens, 12, "location")?,
            parse_f64(&tokens, 13, "location")?,
        ];
        let rotation_y = parse_f64(&tokens, 14, "rotation_y")?;
        let score = if tokens.len() > 15 {
            parse_f64(&tokens, 15, "score")?
        } else {
            1.0
        };
        objects.push(KittiObject3d {
            class_id,
            dimensions,
            location,
            rotation_y,
            score,
            track_id: None,
        });
    }
    Ok(objects)
}

/// Parses KITTI 2D detection lines.
///
/// Accepted forms: `type trunc occ alpha x1..y2 [score]`, a 15-field label
/// line (score 1.0) and a full 16-field detection line whose score is the
/// last field.
pub fn parse_objects_2d(text: &str, classes: &KittiClassMap) -> EngineResult<Vec<KittiObject2d>> {
    let mut objects = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 8 {
            return Err(EngineError::data(format!(
                "2d object line {}: expected at least 8 fields, found {}",
                line_number + 1,
                tokens.len()
            )));
        }
        let class_id = classes.id(tokens[0]);
        let bbox = [
            parse_f64(&tokens, 4, "bbox")?,
            parse_f64(&tokens, 5, "bbox")?,
            parse_f64(&tokens, 6, "bbox")?,
            parse_f64(&tokens, 7, "bbox")?,
        ];
        let score = match tokens.len() {
            9 => parse_f64(&tokens, 8, "score")?,
            n if n >= 16 => parse_f64(&tokens, 15, "score")?,
            _ => 1.0,
        };
        objects.push(KittiObject2d {
            class_id,
            bbox,
            score,
            track_id: None,
        });
    }
    Ok(objects)
}

/// Converts a camera-frame 3D object into a LiDAR-frame detection.
///
/// `cam_to_velo` maps camera coordinates to velodyne coordinates; the shape
/// offsets are rotated by its rotation part.
pub fn to_lidar_detection(
    object: &KittiObject3d,
    cam_to_velo: &RigidTransform3D,
    default_covariance: f64,
) -> EngineResult<LidarDetection> {
    let location = Vector3::new(object.location[0], object.location[1], object.location[2]);
    let position = cam_to_velo.transform_position(&location);
    let rotation = *cam_to_velo.rotation();
    let shape = Box3DShape::from_kitti_camera(
        BoxDimensions {
            height: object.dimensions[0],
            width: object.dimensions[1],
            length: object.dimensions[2],
        },
        object.rotation_y,
    )?
    .transformed(&rotation);
    Ok(LidarDetection {
        position,
        covariance: Matrix3::identity() * default_covariance,
        class_label: object.class_id.map(ClassLabel::new),
        score: object.score,
        shape: Some(shape),
        appearance: None,
    })
}

/// Converts a KITTI 2D object into an engine camera detection.
pub fn to_camera_detection(object: &KittiObject2d, detection_id: u64) -> CameraDetection {
    CameraDetection {
        detection_id,
        bbox: perception_core::BBox2D::from_corners(
            object.bbox[0],
            object.bbox[1],
            object.bbox[2],
            object.bbox[3],
        ),
        class_label: object.class_id.map(ClassLabel::new),
        score: object.score,
    }
}

/// Directory layout expected by [`prepare_sequence`].
#[derive(Debug, Clone)]
pub struct KittiSequencePaths {
    /// Calibration file (`calib.txt`).
    pub calib: std::path::PathBuf,
    /// `times.txt`.
    pub times: std::path::PathBuf,
    /// `poses.txt` (optional).
    pub poses: Option<std::path::PathBuf>,
    /// Directory of per-frame 3D detection files (`000000.txt`, ...).
    pub lidar_detections: std::path::PathBuf,
    /// Directory of per-frame 2D detection files.
    pub camera_detections: std::path::PathBuf,
    /// Optional directory of per-frame 3D label files (tracking ids).
    pub lidar_labels: Option<std::path::PathBuf>,
    /// Optional directory of per-frame 2D label files (tracking ids).
    pub camera_labels: Option<std::path::PathBuf>,
    /// LiDAR sensor id used in the engine input.
    pub lidar_sensor_id: u32,
    /// Camera sensor id used in the engine input.
    pub camera_sensor_id: u32,
}

fn frame_file(dir: &Path, index: usize) -> std::path::PathBuf {
    dir.join(format!("{index:06}.txt"))
}

fn read_file(path: &Path) -> EngineResult<String> {
    Ok(std::fs::read_to_string(path)?)
}

/// Result of preparing a KITTI sequence.
pub struct KittiPrepared {
    /// Engine [`Calibration`].
    pub calibration: Calibration,
    /// Optional pose track (reference -> world).
    pub poses: Option<PoseTrack>,
    /// Engine input records, sorted by arrival then event time.
    pub inputs: Vec<EngineInputRecord>,
    /// Ground-truth records.
    pub truth: Vec<crate::io::TruthRecord>,
}

/// Converts a KITTI sequence directory into engine inputs and ground truth.
///
/// Detection frames are read from `lidar_detections` / `camera_detections`;
/// label directories are optional and only used for evaluation ground truth.
/// The arrival time defaults to the event time; injected delays are applied
/// later by the caller so the conversion stays deterministic.
pub fn prepare_sequence(
    paths: &KittiSequencePaths,
    image_width: u32,
    image_height: u32,
    near_plane: f64,
    default_covariance: f64,
    pose_max_gap_ns: i64,
) -> EngineResult<KittiPrepared> {
    let classes = KittiClassMap::default();
    let calibration_text = read_file(&paths.calib)?;
    let kitti_calib = KittiCalibration::parse(&calibration_text)?;
    let (mut calibration, velo_to_cam) =
        kitti_calib.to_calibration(image_width, image_height, near_plane)?;
    let cam_to_velo = inverse(&velo_to_cam);
    let timestamps = parse_times(&read_file(&paths.times)?)?;

    let poses = match &paths.poses {
        Some(path) => {
            let camera_poses = parse_poses(&read_file(path)?)?;
            if camera_poses.len() < timestamps.len() {
                return Err(EngineError::data(
                    "pose file has fewer poses than times entries",
                ));
            }
            Some(build_pose_track(
                &timestamps,
                &camera_poses[..timestamps.len()],
                &velo_to_cam,
                pose_max_gap_ns,
            )?)
        }
        None => None,
    };
    // With ego poses, tracks (and therefore ground truth) live in the world frame.
    if poses.is_some() {
        calibration.tracking_frame = FrameId::new(KITTI_WORLD_FRAME);
    }

    let mut inputs = Vec::new();
    let mut truth = Vec::new();

    for (index, timestamp) in timestamps.iter().enumerate() {
        let lidar_path = frame_file(&paths.lidar_detections, index);
        if lidar_path.exists() {
            let objects = parse_objects_3d(&read_file(&lidar_path)?, &classes)?;
            let detections = objects
                .iter()
                .map(|object| to_lidar_detection(object, &cam_to_velo, default_covariance))
                .collect::<EngineResult<Vec<_>>>()?;
            inputs.push(EngineInputRecord::Lidar {
                schema_version: ENGINE_SCHEMA_VERSION,
                sensor_id: paths.lidar_sensor_id,
                sequence: index as u64,
                event_time_ns: timestamp.as_nanos(),
                arrival_time_ns: timestamp.as_nanos(),
                frame_id: "velodyne".to_string(),
                detections: detections
                    .iter()
                    .map(|detection| LidarDetectionRecord {
                        position: detection.position.iter().copied().collect(),
                        covariance: row_major_3(&detection.covariance),
                        class: detection.class_label.map(ClassLabel::value),
                        score: detection.score,
                        appearance: None,
                        shape: detection.shape.as_ref().map(|shape| ShapeMetadataRecord {
                            height: shape.dimensions.height,
                            width: shape.dimensions.width,
                            length: shape.dimensions.length,
                            rotation_y: 0.0,
                            corners: Some(shape.flattened_offsets()),
                        }),
                    })
                    .collect(),
            });
        }

        let camera_path = frame_file(&paths.camera_detections, index);
        if camera_path.exists() {
            let objects = parse_objects_2d(&read_file(&camera_path)?, &classes)?;
            inputs.push(EngineInputRecord::Camera {
                schema_version: ENGINE_SCHEMA_VERSION,
                sensor_id: paths.camera_sensor_id,
                sequence: index as u64,
                event_time_ns: timestamp.as_nanos(),
                arrival_time_ns: timestamp.as_nanos(),
                frame_id: KITTI_CAMERA_FRAME.to_string(),
                detections: objects
                    .iter()
                    .enumerate()
                    .map(|(id, object)| CameraDetectionRecord {
                        detection_id: id as u64,
                        bbox: object.bbox.to_vec(),
                        class: object.class_id,
                        score: object.score,
                    })
                    .collect(),
            });
        }

        if let Some(labels_dir) = &paths.lidar_labels {
            let path = frame_file(labels_dir, index);
            if path.exists() {
                let labels = parse_tracking_labels_3d(&read_file(&path)?, &classes)?;
                let cam_to_tracking = match &poses {
                    Some(poses) => compose(&poses.pose_at(*timestamp)?, &cam_to_velo),
                    None => cam_to_velo.clone(),
                };
                let objects = labels
                    .iter()
                    .map(|(track_id, object)| crate::eval::GroundTruthObject {
                        truth_id: *track_id,
                        class: object.class_id.unwrap_or(0),
                        position: cam_to_tracking
                            .transform_position(&Vector3::new(
                                object.location[0],
                                object.location[1],
                                object.location[2],
                            ))
                            .iter()
                            .copied()
                            .collect(),
                    })
                    .collect();
                truth.push(crate::io::TruthRecord::Truth3d {
                    schema_version: ENGINE_SCHEMA_VERSION,
                    timestamp_ns: timestamp.as_nanos(),
                    objects,
                });
            }
        }
    }

    Ok(KittiPrepared {
        calibration,
        poses,
        inputs,
        truth,
    })
}

/// Parses KITTI 3D tracking labels: `frame track_id type trunc occ alpha ...`.
///
/// `DontCare` regions carry track id `-1` and are skipped.
pub fn parse_tracking_labels_3d(
    text: &str,
    classes: &KittiClassMap,
) -> EngineResult<Vec<(u32, KittiObject3d)>> {
    let mut objects = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 16 {
            return Err(EngineError::data(
                "3d tracking label: expected at least 16 fields",
            ));
        }
        let track_id = tokens[1]
            .parse::<i64>()
            .map_err(|_| EngineError::data("bad tracking id"))?;
        if track_id < 0 || tokens[2] == "DontCare" {
            continue;
        }
        let track_id =
            u32::try_from(track_id).map_err(|_| EngineError::data("tracking id out of range"))?;
        let class_id = classes.id(tokens[2]);
        let object = KittiObject3d {
            class_id,
            dimensions: [
                parse_f64(&tokens, 10, "dimensions")?,
                parse_f64(&tokens, 11, "dimensions")?,
                parse_f64(&tokens, 12, "dimensions")?,
            ],
            location: [
                parse_f64(&tokens, 13, "location")?,
                parse_f64(&tokens, 14, "location")?,
                parse_f64(&tokens, 15, "location")?,
            ],
            rotation_y: if tokens.len() > 16 {
                parse_f64(&tokens, 16, "rotation_y")?
            } else {
                0.0
            },
            score: 1.0,
            track_id: Some(track_id),
        };
        objects.push((track_id, object));
    }
    Ok(objects)
}

fn row_major_3(m: &Matrix3<f64>) -> Vec<f64> {
    (0..3)
        .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
        .collect()
}

/// Builds a LiDAR batch directly (used by tests and fixtures).
pub fn lidar_batch(
    sensor_id: u32,
    sequence: u64,
    timestamp: Timestamp,
    detections: Vec<LidarDetection>,
) -> EngineBatch {
    EngineBatch::Lidar(LidarBatch {
        sensor_id: SensorId::new(sensor_id),
        sequence,
        event_time: timestamp,
        frame_id: FrameId::new("velodyne"),
        detections,
    })
}

/// Builds a camera batch directly (used by tests and fixtures).
pub fn camera_batch(
    sensor_id: u32,
    sequence: u64,
    timestamp: Timestamp,
    detections: Vec<CameraDetection>,
) -> EngineBatch {
    EngineBatch::Camera(CameraBatch {
        sensor_id: SensorId::new(sensor_id),
        sequence,
        event_time: timestamp,
        frame_id: FrameId::new(KITTI_CAMERA_FRAME),
        detections,
    })
}
