//! A small, deterministic synthetic camera + LiDAR fixture.
//!
//! This is **synthetic data** used to exercise the engine end to end when a
//! real dataset cannot be downloaded. It is not a substitute for real-data
//! validation. The fixture is written in the engine JSONL schema plus a setup
//! file, so the same `replay`/`evaluate`/`report` path runs on it.

use std::path::{Path, PathBuf};

use factor_tracker_core::{BBox2D, Timestamp};
use nalgebra::{Matrix3, Vector3};

use crate::error::{EngineError, EngineResult};
use crate::geometry::{inverse, Box3DShape, BoxDimensions, Calibration, CameraIntrinsics};
use crate::io::{
    write_jsonl, CameraDetectionRecord, EngineInputRecord, LidarDetectionRecord,
    ShapeMetadataRecord, TruthRecord, ENGINE_SCHEMA_VERSION,
};
use crate::setup::{transform_to_vec16, EngineSetup, IntrinsicsSetup, PoseSetup};

const IMAGE_WIDTH: u32 = 1242;
const IMAGE_HEIGHT: u32 = 375;
const FRAME_PERIOD_NS: i64 = 100_000_000;
const FRAMES: usize = 40;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1.0e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }

    fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

struct FixtureObject {
    truth_id: u32,
    class: u32,
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    dimensions: BoxDimensions,
}

/// Generated fixture data.
pub struct FixtureData {
    /// Engine setup (calibration + poses).
    pub setup: EngineSetup,
    /// Engine inputs.
    pub inputs: Vec<EngineInputRecord>,
    /// Ground truth.
    pub truth: Vec<TruthRecord>,
}

/// The camera-from-velodyne rotation used by the fixture.
fn cam_from_velo_rotation() -> Matrix3<f64> {
    Matrix3::new(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0)
}

fn dimensions(h: f64, w: f64, l: f64) -> BoxDimensions {
    BoxDimensions {
        height: h,
        width: w,
        length: l,
    }
}

/// Generates the fixture for a seed.
pub fn generate(seed: u64) -> EngineResult<FixtureData> {
    let mut rng = SplitMix64(seed ^ 0xC0FF_EE00);
    let intrinsics = CameraIntrinsics {
        fx: 700.0,
        fy: 700.0,
        cx: IMAGE_WIDTH as f64 / 2.0,
        cy: IMAGE_HEIGHT as f64 / 2.0,
        skew: 0.0,
        image_width: IMAGE_WIDTH,
        image_height: IMAGE_HEIGHT,
    };
    let near_plane = 0.3;

    let cam_rotation = cam_from_velo_rotation();
    let cam_from_velo = factor_tracker_core::RigidTransform3D::new(cam_rotation, Vector3::zeros())
        .map_err(|error| EngineError::data(format!("fixture transform: {error}")))?;

    let objects = vec![
        FixtureObject {
            truth_id: 1,
            class: 1,
            position: Vector3::new(20.0, -1.5, 0.0),
            velocity: Vector3::new(1.0, 0.0, 0.0),
            dimensions: dimensions(1.5, 1.8, 4.0),
        },
        FixtureObject {
            truth_id: 2,
            class: 2,
            position: Vector3::new(12.0, 2.0, 0.0),
            velocity: Vector3::new(-0.5, 0.0, 0.0),
            dimensions: dimensions(1.7, 0.8, 0.8),
        },
        FixtureObject {
            truth_id: 3,
            class: 1,
            position: Vector3::new(35.0, 1.5, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            dimensions: dimensions(1.6, 1.9, 4.2),
        },
    ];

    let mut inputs = Vec::new();
    let mut truth = Vec::new();
    let mut poses = Vec::new();

    for frame in 0..FRAMES {
        let timestamp = Timestamp::from_nanos(frame as i64 * FRAME_PERIOD_NS);
        let t = timestamp.as_nanos() as f64 / 1.0e9;
        let ego_world = Vector3::new(5.0 * t, 0.0, 0.0);
        let pose = factor_tracker_core::RigidTransform3D::new(Matrix3::identity(), ego_world)
            .map_err(|error| EngineError::data(format!("fixture pose: {error}")))?;
        poses.push(PoseSetup {
            timestamp_ns: timestamp.as_nanos(),
            transform: transform_to_vec16(&pose),
        });
        let world_to_velo = inverse(&pose);
        let cam_from_world = crate::geometry::compose(&cam_from_velo, &world_to_velo);

        let mut lidar_detections = Vec::new();
        let mut camera_detections = Vec::new();
        let mut truth_objects = Vec::new();
        let mut camera_truth: Vec<crate::eval::CameraDetectionTruth> = Vec::new();

        for object in &objects {
            let center_world = object.position + object.velocity * t;
            truth_objects.push(crate::eval::GroundTruthObject {
                truth_id: object.truth_id,
                class: object.class,
                position: center_world.iter().copied().collect(),
            });
            let shape_world = Box3DShape::axis_aligned(object.dimensions)?;

            // LiDAR observation in the velodyne frame.
            if !rng.chance(0.05) {
                let center_velo = center_world - ego_world;
                let noisy =
                    center_velo + Vector3::new(rng.normal(), rng.normal(), rng.normal()) * 0.05;
                lidar_detections.push(LidarDetectionRecord {
                    position: noisy.iter().copied().collect(),
                    covariance: vec![0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
                    class: Some(object.class),
                    score: 0.9,
                    appearance: None,
                    shape: Some(ShapeMetadataRecord {
                        height: object.dimensions.height,
                        width: object.dimensions.width,
                        length: object.dimensions.length,
                        rotation_y: 0.0,
                        corners: Some(shape_world.flattened_offsets()),
                    }),
                });
            }

            // Camera observation: project the world box into the image.
            let corners_world = shape_world.corners_at(&center_world);
            let mut corners_camera = [Vector3::zeros(); 8];
            for (i, corner) in corners_world.iter().enumerate() {
                corners_camera[i] = cam_from_world.transform_position(corner);
            }
            if let Ok(projected) =
                crate::geometry::project_box_corners(&intrinsics, near_plane, &corners_camera)
            {
                if !rng.chance(0.08) {
                    let noise = 1.5;
                    let bbox = BBox2D::from_corners(
                        projected.bbox.x_min + noise * rng.normal(),
                        projected.bbox.y_min + noise * rng.normal(),
                        projected.bbox.x_max + noise * rng.normal(),
                        projected.bbox.y_max + noise * rng.normal(),
                    );
                    let detection_id = camera_detections.len() as u64;
                    camera_truth.push(crate::eval::CameraDetectionTruth {
                        detection_id,
                        truth_id: object.truth_id,
                    });
                    camera_detections.push(CameraDetectionRecord {
                        detection_id,
                        bbox: vec![bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                        class: Some(object.class),
                        score: 0.85,
                    });
                }
            }
        }

        // Occasional LiDAR false positive.
        if rng.chance(0.03) {
            lidar_detections.push(LidarDetectionRecord {
                position: vec![
                    30.0 + rng.next_f64() * 10.0,
                    -8.0 + rng.next_f64() * 16.0,
                    rng.next_f64(),
                ],
                covariance: vec![0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25],
                class: None,
                score: 0.4,
                appearance: None,
                shape: None,
            });
        }

        inputs.push(EngineInputRecord::Lidar {
            schema_version: ENGINE_SCHEMA_VERSION,
            sensor_id: 1,
            sequence: frame as u64,
            event_time_ns: timestamp.as_nanos(),
            arrival_time_ns: timestamp.as_nanos(),
            frame_id: "velodyne".to_string(),
            detections: lidar_detections,
        });
        inputs.push(EngineInputRecord::Camera {
            schema_version: ENGINE_SCHEMA_VERSION,
            sensor_id: 2,
            sequence: frame as u64,
            event_time_ns: timestamp.as_nanos(),
            arrival_time_ns: timestamp.as_nanos(),
            frame_id: "cam0_rect".to_string(),
            detections: camera_detections,
        });
        truth.push(TruthRecord::Truth3d {
            schema_version: ENGINE_SCHEMA_VERSION,
            timestamp_ns: timestamp.as_nanos(),
            objects: truth_objects,
        });
        if !camera_truth.is_empty() {
            truth.push(TruthRecord::CameraAssociation {
                schema_version: ENGINE_SCHEMA_VERSION,
                timestamp_ns: timestamp.as_nanos(),
                entries: camera_truth,
            });
        }
    }

    let setup = EngineSetup {
        tracking_frame: "world".to_string(),
        lidar_frame: "velodyne".to_string(),
        camera_frame: "cam0_rect".to_string(),
        lidar_to_reference: transform_to_vec16(&factor_tracker_core::RigidTransform3D::identity()),
        // `cam_from_velo` maps velo -> camera; the setup stores the sensor ->
        // reference direction, i.e. camera -> velo.
        camera_to_reference: transform_to_vec16(&inverse(&cam_from_velo)),
        intrinsics: IntrinsicsSetup {
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
            skew: intrinsics.skew,
            image_width: intrinsics.image_width,
            image_height: intrinsics.image_height,
        },
        near_plane,
        poses: Some(poses),
        pose_max_gap_ns: 300_000_000,
    };

    Ok(FixtureData {
        setup,
        inputs,
        truth,
    })
}

/// Paths written by [`write()`].
pub struct FixturePaths {
    /// Setup file.
    pub setup: PathBuf,
    /// Engine inputs.
    pub inputs: PathBuf,
    /// Ground truth.
    pub truth: PathBuf,
}

/// Writes the fixture to `out_dir`.
pub fn write(out_dir: &Path, seed: u64) -> EngineResult<FixturePaths> {
    std::fs::create_dir_all(out_dir)?;
    let data = generate(seed)?;
    let setup_path = out_dir.join("setup.json");
    std::fs::write(&setup_path, serde_json::to_string_pretty(&data.setup)?)?;
    let inputs_path = out_dir.join("inputs.jsonl");
    write_jsonl(&inputs_path, &data.inputs)?;
    let truth_path = out_dir.join("truth.jsonl");
    write_jsonl(&truth_path, &data.truth)?;
    Ok(FixturePaths {
        setup: setup_path,
        inputs: inputs_path,
        truth: truth_path,
    })
}

/// Convenience for tests: builds a calibration directly.
pub fn fixture_calibration(setup: &EngineSetup) -> EngineResult<Calibration> {
    setup.to_calibration()
}
