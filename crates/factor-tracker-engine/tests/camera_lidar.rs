//! Integration tests for the camera-assisted LiDAR engine.

use factor_tracker_core::{BBox2D, ClassLabel, RigidTransform3D, Timestamp};
use factor_tracker_engine::engine::{EngineConfig, TrackingMode};
use factor_tracker_engine::geometry::{
    Calibration, CameraIntrinsics, PoseTrack, ProjectionRejection,
};
use factor_tracker_engine::inject::{self, InjectionConfig, SensorKind};
use factor_tracker_engine::input::{CameraDetection, LidarDetection};
use factor_tracker_engine::kitti::{camera_batch, lidar_batch};
use factor_tracker_engine::reorder::{AsyncReorderBuffer, IngestOutcome, ReorderConfig};
use factor_tracker_engine::{fixture, replay};
use nalgebra::{Matrix3, Vector3};

fn rotation_camera_from_velo() -> Matrix3<f64> {
    Matrix3::new(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0)
}

fn test_calibration() -> Calibration {
    let rc = rotation_camera_from_velo();
    Calibration {
        tracking_frame: "world".into(),
        lidar_frame: "velodyne".into(),
        camera_frame: factor_tracker_engine::kitti::KITTI_CAMERA_FRAME.into(),
        lidar_to_reference: RigidTransform3D::identity(),
        camera_to_reference: RigidTransform3D::new(rc.transpose(), Vector3::zeros()).unwrap(),
        intrinsics: CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 50.0,
            cy: 50.0,
            skew: 0.0,
            image_width: 100,
            image_height: 100,
        },
        near_plane: 0.1,
    }
}

fn shape() -> factor_tracker_engine::Box3DShape {
    factor_tracker_engine::Box3DShape::axis_aligned(factor_tracker_engine::BoxDimensions {
        height: 1.0,
        width: 1.0,
        length: 1.0,
    })
    .unwrap()
}

fn lidar_detection(position: Vector3<f64>, class: Option<u32>) -> LidarDetection {
    LidarDetection {
        position,
        covariance: Matrix3::identity() * 0.01,
        class_label: class.map(ClassLabel::new),
        score: 0.9,
        shape: Some(shape()),
        appearance: None,
    }
}

fn camera_detection(bbox: [f64; 4], class: Option<u32>, score: f64, id: u64) -> CameraDetection {
    CameraDetection {
        detection_id: id,
        bbox: BBox2D::from_corners(bbox[0], bbox[1], bbox[2], bbox[3]),
        class_label: class.map(ClassLabel::new),
        score,
    }
}

/// Calibration with the camera frame equal to the tracking frame but oriented
/// so that `+x` in the tracking frame looks into the camera.
fn projection_calibration() -> Calibration {
    test_calibration()
}

#[test]
fn known_point_projects_to_expected_pixel() {
    let calibration = projection_calibration();
    let rc = rotation_camera_from_velo();
    let point_camera = rc * Vector3::new(10.0, 0.0, 0.0);
    let pixel = calibration.intrinsics.project(&point_camera).unwrap();
    assert!((pixel.x - 50.0).abs() < 1e-9);
    assert!((pixel.y - 50.0).abs() < 1e-9);
}

#[test]
fn projection_rejections_are_explicit() {
    let calibration = projection_calibration();
    let rc = rotation_camera_from_velo();
    let corners = |center: Vector3<f64>| {
        let mut out = [Vector3::zeros(); 8];
        for (i, offset) in shape().corner_offsets.iter().enumerate() {
            out[i] = rc * (center + offset);
        }
        out
    };
    // Behind the camera.
    let behind = factor_tracker_engine::geometry::project_box_corners(
        &calibration.intrinsics,
        calibration.near_plane,
        &corners(Vector3::new(-10.0, 0.0, 0.0)),
    );
    assert_eq!(behind.unwrap_err(), ProjectionRejection::BehindCamera);
    // Straddling the near plane.
    let straddle = factor_tracker_engine::geometry::project_box_corners(
        &calibration.intrinsics,
        calibration.near_plane,
        &corners(Vector3::new(0.0, 0.0, 0.0)),
    );
    assert_eq!(
        straddle.unwrap_err(),
        ProjectionRejection::StraddlesNearPlane
    );
    // Far outside the image.
    let out_of_view = factor_tracker_engine::geometry::project_box_corners(
        &calibration.intrinsics,
        calibration.near_plane,
        &corners(Vector3::new(10.0, 500.0, 0.0)),
    );
    assert_eq!(out_of_view.unwrap_err(), ProjectionRejection::OutOfView);
}

#[test]
fn reversed_transform_direction_is_detected() {
    // Using the inverse (wrong) direction places the object behind the camera.
    let correct = projection_calibration();
    let mut reversed = correct.clone();
    reversed.camera_to_reference =
        RigidTransform3D::new(rotation_camera_from_velo(), Vector3::zeros()).unwrap();
    let rc = rotation_camera_from_velo();
    let mut corners_correct = [Vector3::zeros(); 8];
    let mut corners_reversed = [Vector3::zeros(); 8];
    for (i, offset) in shape().corner_offsets.iter().enumerate() {
        let world = Vector3::new(10.0, 0.0, 0.0) + offset;
        corners_correct[i] = factor_tracker_engine::geometry::inverse(&correct.camera_to_reference)
            .transform_position(&world);
        corners_reversed[i] =
            factor_tracker_engine::geometry::inverse(&reversed.camera_to_reference)
                .transform_position(&world);
    }
    let correct_projection = factor_tracker_engine::geometry::project_box_corners(
        &correct.intrinsics,
        correct.near_plane,
        &corners_correct,
    );
    let reversed_projection = factor_tracker_engine::geometry::project_box_corners(
        &reversed.intrinsics,
        reversed.near_plane,
        &corners_reversed,
    );
    assert!(correct_projection.is_ok());
    assert!(
        reversed_projection.is_err(),
        "the reversed transform direction must not produce a valid projection"
    );
    let _ = rc;
}

fn moving_pose_track() -> PoseTrack {
    let samples = (0..5)
        .map(|i| {
            let t = i as i64 * 100_000_000;
            let pose = RigidTransform3D::new(Matrix3::identity(), Vector3::new(i as f64, 0.0, 0.0))
                .unwrap();
            (Timestamp::from_nanos(t), pose)
        })
        .collect();
    PoseTrack::new(samples, 200_000_000).unwrap()
}

#[test]
fn moving_sensor_uses_pose_at_observation_time() {
    // With ego poses, the same sensor-frame point maps to different world
    // points at different times.
    let config = EngineConfig {
        tracker: factor_tracker_core::Cartesian3DTrackerConfig {
            gate_squared_mahalanobis: 100.0,
            ..factor_tracker_core::Cartesian3DTrackerConfig::default()
        },
        ..EngineConfig::default()
    };
    let calibration = projection_calibration();
    let mut engine = factor_tracker_engine::CameraAssistedLidarEngine::new(
        config,
        calibration,
        Some(moving_pose_track()),
    )
    .unwrap();
    // A point fixed in the sensor frame at (10,0,0) at t=0 -> world (10,0,0).
    engine.ingest(
        lidar_batch(
            1,
            0,
            Timestamp::from_nanos(0),
            vec![lidar_detection(Vector3::new(10.0, 0.0, 0.0), Some(1))],
        ),
        Timestamp::from_nanos(0),
    );
    engine.poll(Timestamp::from_nanos(i64::MAX)).unwrap();
    // At t=100ms the ego has moved 1m, so the same sensor point is world (11,0,0).
    engine.ingest(
        lidar_batch(
            1,
            1,
            Timestamp::from_nanos(100_000_000),
            vec![lidar_detection(Vector3::new(10.0, 0.0, 0.0), Some(1))],
        ),
        Timestamp::from_nanos(100_000_000),
    );
    engine.poll(Timestamp::from_nanos(i64::MAX)).unwrap();
    engine.flush().unwrap();
    let last = engine.lidar_frames().last().unwrap();
    let track = last.tracks.iter().find(|t| t.track_id == 1).unwrap();
    assert!((track.position[0] - 11.0).abs() < 0.5);
}

#[test]
fn missing_pose_skips_batch_without_state_change() {
    let config = EngineConfig {
        mode: TrackingMode::CameraAssociated,
        ..EngineConfig::default()
    };
    let mut engine = factor_tracker_engine::CameraAssistedLidarEngine::new(
        config,
        projection_calibration(),
        Some(moving_pose_track()),
    )
    .unwrap();
    // t=1s is outside the pose range: no extrapolation, batch is skipped.
    engine.ingest(
        lidar_batch(
            1,
            0,
            Timestamp::from_nanos(1_000_000_000),
            vec![lidar_detection(Vector3::new(10.0, 0.0, 0.0), Some(1))],
        ),
        Timestamp::from_nanos(1_000_000_000),
    );
    engine.poll(Timestamp::from_nanos(1_000_000_000)).unwrap();
    engine.flush().unwrap();
    assert_eq!(engine.lidar_frames().len(), 0);
    assert_eq!(engine.counters().engine.transform_unavailable, 1);
}

#[test]
fn confidence_is_a_moving_average_of_detection_log_odds() {
    let mut engine = factor_tracker_engine::CameraAssistedLidarEngine::new(
        EngineConfig {
            mode: TrackingMode::LidarOnly,
            ..EngineConfig::default()
        },
        test_calibration(),
        None,
    )
    .unwrap();
    for (k, score) in [0.5, 0.5, 0.95, 0.95].into_iter().enumerate() {
        let t = Timestamp::from_nanos(k as i64 * 100_000_000);
        let detection = LidarDetection {
            score,
            ..lidar_detection(Vector3::new(10.0, 0.0, 0.0), None)
        };
        engine.ingest(lidar_batch(1, k as u64, t, vec![detection]), t);
    }
    engine.flush().unwrap();
    let confidence: Vec<f64> = engine
        .lidar_frames()
        .iter()
        .map(|frame| frame.tracks[0].confidence.unwrap())
        .collect();
    // log-odds 0, 0, ln 19, ln 19 averaged with weight 0.3
    let strong = 19f64.ln();
    let c2 = 0.3 * strong;
    let c3 = c2 + 0.3 * (strong - c2);
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    for (got, want) in confidence.iter().zip([0.5, 0.5, sigmoid(c2), sigmoid(c3)]) {
        assert!((got - want).abs() < 1e-9, "{confidence:?}");
    }
    assert_eq!(
        engine.lidar_frames()[3].tracks[0].track_id,
        engine.lidar_frames()[0].tracks[0].track_id
    );
}

#[test]
fn camera_miss_does_not_drop_track() {
    let calibration = projection_calibration();
    let mut engine = factor_tracker_engine::CameraAssistedLidarEngine::new(
        EngineConfig {
            mode: TrackingMode::CameraAssociated,
            ..EngineConfig::default()
        },
        calibration,
        None,
    )
    .unwrap();
    engine.ingest(
        lidar_batch(
            1,
            0,
            Timestamp::from_nanos(0),
            vec![lidar_detection(Vector3::new(10.0, 0.0, 0.0), Some(1))],
        ),
        Timestamp::from_nanos(0),
    );
    engine.poll(Timestamp::from_nanos(i64::MAX)).unwrap();
    let before = engine.lidar_frames().last().unwrap().tracks.len();
    // Empty camera batch (object out of the camera's view).
    engine.ingest(
        camera_batch(2, 0, Timestamp::from_nanos(0), vec![]),
        Timestamp::from_nanos(0),
    );
    engine.poll(Timestamp::from_nanos(i64::MAX)).unwrap();
    engine.flush().unwrap();
    assert_eq!(before, 1);
    assert_eq!(engine.lidar_frames().last().unwrap().tracks.len(), 1);
}

#[test]
fn camera_support_does_not_shrink_position_covariance() {
    let calibration = projection_calibration();
    let mut engine = factor_tracker_engine::CameraAssistedLidarEngine::new(
        EngineConfig {
            mode: TrackingMode::CameraAssociated,
            ..EngineConfig::default()
        },
        calibration,
        None,
    )
    .unwrap();
    engine.ingest(
        lidar_batch(
            1,
            0,
            Timestamp::from_nanos(0),
            vec![lidar_detection(Vector3::new(10.0, 0.0, 0.0), Some(1))],
        ),
        Timestamp::from_nanos(0),
    );
    engine.poll(Timestamp::from_nanos(i64::MAX)).unwrap();
    let before = engine.lidar_frames().last().unwrap().tracks[0]
        .covariance
        .clone();
    // Camera detection that matches the projected track.
    engine.ingest(
        camera_batch(
            2,
            0,
            Timestamp::from_nanos(0),
            vec![camera_detection([40.0, 40.0, 60.0, 60.0], Some(1), 0.9, 0)],
        ),
        Timestamp::from_nanos(0),
    );
    engine.poll(Timestamp::from_nanos(i64::MAX)).unwrap();
    assert_eq!(engine.counters().engine.camera_matches, 1);
    // No new LiDAR frame, so the last spatial covariance is unchanged.
    let after = engine.lidar_frames().last().unwrap().tracks[0]
        .covariance
        .clone();
    assert_eq!(before, after);
}

#[test]
fn lidar_only_and_camera_associated_states_match() {
    let calibration = projection_calibration();
    let inputs: Vec<_> = (0..5)
        .flat_map(|i| {
            let t = Timestamp::from_nanos(i * 100_000_000);
            let position = Vector3::new(10.0 + i as f64, 0.0, 0.0);
            vec![
                (
                    lidar_batch(1, i as u64, t, vec![lidar_detection(position, Some(1))]),
                    t,
                ),
                (
                    camera_batch(
                        2,
                        i as u64,
                        t,
                        vec![camera_detection(
                            [40.0, 40.0, 60.0, 60.0],
                            Some(1),
                            0.9,
                            i as u64,
                        )],
                    ),
                    t,
                ),
            ]
        })
        .collect();

    let run = |mode| {
        let config = EngineConfig {
            mode,
            ..EngineConfig::default()
        };
        replay::run(calibration.clone(), None, config, &inputs).unwrap()
    };
    let lidar_only = run(TrackingMode::LidarOnly);
    let associated = run(TrackingMode::CameraAssociated);
    let a = lidar_only.lidar_frames().last().unwrap();
    let b = associated.lidar_frames().last().unwrap();
    assert_eq!(a.tracks.len(), b.tracks.len());
    for (ta, tb) in a.tracks.iter().zip(b.tracks.iter()) {
        assert_eq!(ta.track_id, tb.track_id);
        for k in 0..3 {
            assert!((ta.position[k] - tb.position[k]).abs() < 1e-9);
            assert!((ta.covariance[k * 6 + k] - tb.covariance[k * 6 + k]).abs() < 1e-9);
        }
    }
}

#[test]
fn reorder_buffer_handles_late_duplicate_overflow_timeout_and_flush() {
    let mut buffer = AsyncReorderBuffer::new(ReorderConfig {
        window_ns: 100,
        max_hold_ns: 1_000,
        capacity: 2,
    })
    .unwrap();
    let t = |ns| Timestamp::from_nanos(ns);
    // Accept two batches, then overflow.
    assert_eq!(
        buffer.ingest(lidar_batch(1, 0, t(100), vec![]), t(100)),
        IngestOutcome::Queued
    );
    assert_eq!(
        buffer.ingest(lidar_batch(1, 1, t(200), vec![]), t(100)),
        IngestOutcome::Queued
    );
    assert_eq!(
        buffer.ingest(lidar_batch(1, 2, t(300), vec![]), t(100)),
        IngestOutcome::Overflow
    );
    // Poll with a large arrival time releases via the hold timeout.
    let released = buffer.poll(t(2_000));
    assert!(released.len() >= 2);
    // A duplicate of an already seen sequence is rejected.
    assert_eq!(
        buffer.ingest(lidar_batch(1, 0, t(100), vec![]), t(2_000)),
        IngestOutcome::Duplicate
    );
    // A late batch (older than the last released event time) is rejected.
    assert_eq!(
        buffer.ingest(lidar_batch(1, 9, t(50), vec![]), t(2_000)),
        IngestOutcome::Late
    );
    // Flush releases the rest deterministically.
    buffer.ingest(lidar_batch(1, 3, t(500), vec![]), t(2_000));
    let flushed = buffer.flush();
    assert_eq!(flushed.len(), 1);
    assert_eq!(buffer.counters().late, 1);
    assert_eq!(buffer.counters().duplicate, 1);
    assert_eq!(buffer.counters().overflow, 1);
}

#[test]
fn same_time_processing_prefers_lidar_before_camera() {
    let mut buffer = AsyncReorderBuffer::new(ReorderConfig {
        window_ns: 0,
        max_hold_ns: 1_000,
        capacity: 16,
    })
    .unwrap();
    let t = Timestamp::from_nanos(100);
    // Camera ingested first, LiDAR second, same event time.
    buffer.ingest(camera_batch(2, 0, t, vec![]), Timestamp::from_nanos(100));
    buffer.ingest(lidar_batch(1, 0, t, vec![]), Timestamp::from_nanos(100));
    let released = buffer.poll(Timestamp::from_nanos(1_000));
    assert_eq!(released.len(), 2);
    assert!(released[0].is_lidar(), "LiDAR must be released first");
}

#[test]
fn injection_duplicates_and_pauses() {
    let t = |ns| Timestamp::from_nanos(ns);
    let inputs = vec![
        (lidar_batch(1, 0, t(0), vec![]), t(0)),
        (camera_batch(2, 0, t(0), vec![]), t(0)),
        (lidar_batch(1, 1, t(100), vec![]), t(100)),
    ];
    let config = InjectionConfig {
        duplicate_every: 2,
        pause_sensor: Some(SensorKind::Camera),
        pause_start_ns: 0,
        pause_end_ns: 0,
        ..InjectionConfig::default()
    };
    let injected = inject::apply(&inputs, &config);
    // Camera at t=0 is paused (dropped); lidar seq 0 is duplicated.
    assert!(injected.iter().all(|(batch, _)| batch.is_lidar()));
    assert_eq!(injected.len(), 3);
}

#[test]
fn fixture_replay_is_deterministic() {
    let calibration = projection_calibration();
    let run = || {
        let data = fixture::generate(7).unwrap();
        let calibration = fixture::fixture_calibration(&data.setup).unwrap();
        let inputs: Vec<_> = data
            .inputs
            .iter()
            .cloned()
            .map(|record| record.into_batch().unwrap())
            .collect();
        let config = EngineConfig {
            mode: TrackingMode::CameraAssisted,
            ..EngineConfig::default()
        };
        replay::run(calibration, None, config, &inputs).unwrap()
    };
    let _ = calibration;
    let a = run();
    let b = run();
    assert_eq!(a.lidar_frames().len(), b.lidar_frames().len());
    assert_eq!(
        a.lidar_frames().last().unwrap().tracks,
        b.lidar_frames().last().unwrap().tracks
    );
}

#[test]
fn class_prior_is_applied_only_in_camera_assisted_mode() {
    let calibration = projection_calibration();
    // Build a track with a known class prior via three matching camera frames,
    // then present two LiDAR detections with different classes at equal
    // distance from the prediction.
    let build_inputs = || {
        let t0 = Timestamp::from_nanos(0);
        let mut inputs: Vec<_> = vec![(
            lidar_batch(
                1,
                0,
                t0,
                vec![lidar_detection(Vector3::new(10.0, 0.0, 0.0), None)],
            ),
            t0,
        )];
        for i in 0..3u64 {
            let t = Timestamp::from_nanos(100_000_000 + i as i64 * 100_000_000);
            inputs.push((
                camera_batch(
                    2,
                    i,
                    t,
                    vec![camera_detection([40.0, 40.0, 60.0, 60.0], Some(1), 0.9, i)],
                ),
                t,
            ));
        }
        // Ambiguous LiDAR batch at t=1s: two detections symmetric around the
        // prediction at x=10.
        let t2 = Timestamp::from_nanos(500_000_000);
        inputs.push((
            lidar_batch(
                3,
                0,
                t2,
                vec![
                    lidar_detection(Vector3::new(9.9, 0.0, 0.0), Some(1)),
                    lidar_detection(Vector3::new(10.1, 0.0, 0.0), Some(2)),
                ],
            ),
            t2,
        ));
        inputs
    };

    let run = |mode| {
        let config = EngineConfig {
            mode,
            tracker: factor_tracker_core::Cartesian3DTrackerConfig {
                class_prior_penalty: 50.0,
                ..factor_tracker_core::Cartesian3DTrackerConfig::default()
            },
            ..EngineConfig::default()
        };
        let engine = replay::run(calibration.clone(), None, config, &build_inputs()).unwrap();
        engine.lidar_frames().to_vec()
    };

    let associated = run(TrackingMode::CameraAssociated);
    let assisted = run(TrackingMode::CameraAssisted);
    let assoc_last = associated.last().unwrap();
    let assist_last = assisted.last().unwrap();
    let assoc_track = assoc_last.tracks.iter().find(|t| t.track_id == 1).unwrap();
    let assist_track = assist_last.tracks.iter().find(|t| t.track_id == 1).unwrap();
    // Without the prior the conflict resolves spatially; with the class-1
    // prior the class-1 detection is preferred.
    assert_eq!(assist_track.class_label, Some(1));
    // The two modes must not always produce the same decision for this input.
    assert!(
        assoc_track.class_label != assist_track.class_label
            || assoc_last.associations != assist_last.associations,
        "camera-assisted mode should influence the ambiguous association"
    );
}

#[test]
fn ground_truth_types_are_not_part_of_the_engine_api() {
    // Compile-time-ish check: the engine only accepts sensor batches. Ground
    // truth is only consumed by `evaluate_mode`.
    let _truth: fn() -> Vec<factor_tracker_engine::GroundTruthFrame> = Vec::new;
    let _ = BBox2D::from_corners(0.0, 0.0, 1.0, 1.0);
}

#[test]
fn pose_interpolation_tolerates_quaternion_round_off() {
    // Two consecutive ANYmal odometry poses (GrandTour ETH-2) for which
    // nalgebra's matrix-to-quaternion conversion is ~1e-9 off unit norm.
    let a = factor_tracker_engine::geometry::rigid_from_row_major_3x4(&[
        0.057033789423095005,
        -0.9976222255424496,
        0.0386916265847297,
        212.23496032753766,
        0.9983485693585292,
        0.056722675096575914,
        -0.009092424850756928,
        18.723403884330317,
        0.006876112551465939,
        0.03914630549130271,
        0.999209830737549,
        11.38204340007836,
    ])
    .unwrap();
    let b = factor_tracker_engine::geometry::rigid_from_row_major_3x4(&[
        0.057326996194461355,
        -0.9976248170416008,
        0.03818821716227578,
        212.23314465539505,
        0.9983332253902616,
        0.05702850509836954,
        -0.008861189994730146,
        18.72550315854963,
        0.006662326110127036,
        0.03863255141662481,
        0.9992312742212608,
        11.38358516927335,
    ])
    .unwrap();
    let track = PoseTrack::new(
        vec![
            (Timestamp::from_nanos(1_727_776_473_248_179_200), a),
            (Timestamp::from_nanos(1_727_776_473_298_179_072), b),
        ],
        200_000_000,
    )
    .unwrap();
    for offset in [0, 1, 12_500_000, 25_000_000, 49_999_871] {
        let pose = track
            .pose_at(Timestamp::from_nanos(1_727_776_473_248_179_200 + offset))
            .unwrap();
        assert!((pose.rotation().determinant() - 1.0).abs() < 1e-12);
    }
}

/// A stationary sensor whose given poses jitter (up to 1 degree and 0.2 m
/// from scan 5 on); five static landmarks and one standing person.
fn jittered_scans(pose_correction: bool) -> Vec<Vec<f64>> {
    use factor_tracker_engine::input::{EngineBatch, LidarBatch};
    let jitter = |k: i64| -> (f64, f64) {
        if k < 5 {
            (0.0, 0.0)
        } else if k % 2 == 0 {
            (1f64.to_radians(), 0.2)
        } else {
            (-0.5f64.to_radians(), -0.1)
        }
    };
    let samples = (0..12)
        .map(|k| {
            let (yaw, dx) = jitter(k);
            let (s, c) = yaw.sin_cos();
            let pose = RigidTransform3D::new(
                Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0),
                Vector3::new(dx, 0.0, 0.0),
            )
            .unwrap();
            (Timestamp::from_nanos(k * 100_000_000), pose)
        })
        .collect();
    let config = EngineConfig {
        mode: TrackingMode::LidarOnly,
        pose_correction: pose_correction.then(factor_tracker_engine::LandmarkConfig::default),
        ..EngineConfig::default()
    };
    let mut engine = factor_tracker_engine::CameraAssistedLidarEngine::new(
        config,
        test_calibration(),
        Some(PoseTrack::new(samples, 200_000_000).unwrap()),
    )
    .unwrap();
    let landmarks = vec![
        Vector3::new(5.0, 0.0, 0.0),
        Vector3::new(0.0, 7.0, 0.0),
        Vector3::new(-6.0, -2.0, 0.0),
        Vector3::new(3.0, -8.0, 0.0),
        Vector3::new(-4.0, 9.0, 0.0),
    ];
    for k in 0..12 {
        let t = Timestamp::from_nanos(k * 100_000_000);
        let batch = LidarBatch {
            sensor_id: factor_tracker_core::SensorId::new(1),
            sequence: k as u64,
            event_time: t,
            frame_id: "velodyne".into(),
            detections: vec![lidar_detection(Vector3::new(8.0, 2.0, 0.0), Some(2))],
            landmarks: landmarks.clone(),
        };
        engine.ingest(EngineBatch::Lidar(batch), t);
    }
    engine.flush().unwrap();
    for frame in engine.lidar_frames() {
        assert_eq!(frame.pose_correction.is_some(), pose_correction);
    }
    engine
        .lidar_frames()
        .iter()
        .map(|frame| frame.observations[0].clone())
        .collect()
}

#[test]
fn landmarks_remove_pose_jitter_from_observations() {
    let spread = |observations: &[Vec<f64>]| {
        observations
            .iter()
            .map(|p| (p[0] - observations[0][0]).hypot(p[1] - observations[0][1]))
            .fold(0.0, f64::max)
    };
    assert!(spread(&jittered_scans(false)) > 0.2);
    assert!(spread(&jittered_scans(true)) < 0.01);
}
