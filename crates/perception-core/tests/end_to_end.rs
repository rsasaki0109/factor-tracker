//! End-to-end integration tests for the two trackers.
//!
//! These tests deliberately avoid any premise that identity is always
//! recoverable. They only assert stable IDs in scenarios where the objects are
//! well separated and the association problem is unambiguous. Persistent
//! identity through crossings or full occlusion is *not* guaranteed.

use nalgebra::{Matrix3, Matrix4, Vector3};
use perception_core::{
    BBox2D, BBox2DObservation, BBox2DTracker, BBox2DTrackerConfig, Cartesian3DTracker,
    Cartesian3DTrackerConfig, ClassLabel, FrameId, ObservationBatch2D, ObservationBatch3D,
    Position3DObservation, RigidTransform3D, SensorId, Timestamp, TrackLifecycleState,
    TrackerConfig,
};

fn base_common() -> TrackerConfig {
    TrackerConfig {
        confirm_min_observation_times: 2,
        tentative_timeout_ns: 1_000_000_000,
        confirmed_to_lost_ns: 500_000_000,
        lost_to_delete_ns: 2_000_000_000,
        new_track_score_threshold: Some(0.3),
        max_contributing_sensors: 8,
    }
}

fn obs2d(cx: f64, cy: f64, w: f64, h: f64) -> BBox2DObservation {
    BBox2DObservation {
        bbox: BBox2D::from_center_size(cx, cy, w, h),
        score: 0.9,
        class_label: Some(ClassLabel::new(1)),
        covariance: Matrix4::<f64>::identity(),
    }
}

#[test]
fn two_d_tracks_well_separated_objects_through_loss_and_outlier() {
    let config = BBox2DTrackerConfig {
        common: base_common(),
        gate_min_iou: 0.1,
        ..BBox2DTrackerConfig::default()
    };
    let mut tracker = BBox2DTracker::new(config).unwrap();

    // Three horizontal lanes, well separated, moving at constant velocity.
    let lanes = [(100.0, 50.0), (400.0, 50.0), (700.0, 50.0)];
    let mut last_snapshots = Vec::new();
    for frame in 0..40 {
        let t = frame as i64 * 33_333_333;
        let mut observations = Vec::new();
        for (lane, (x0, y0)) in lanes.iter().enumerate() {
            // Object 1 is temporarily undetected between frames 15 and 20.
            if lane == 1 && (15..=20).contains(&frame) {
                continue;
            }
            let x = x0 + 100.0 * (t as f64 / 1.0e9);
            observations.push(obs2d(x, *y0, 60.0, 60.0));
        }
        // A one-off outlier far away from every lane.
        if frame == 25 {
            observations.push(obs2d(1000.0, 1000.0, 20.0, 20.0));
        }
        last_snapshots = tracker
            .process_batch(&ObservationBatch2D {
                sensor_id: SensorId::new(1),
                sequence: frame,
                timestamp: Timestamp::from_nanos(t),
                frame_id: FrameId::new("camera"),
                observations,
            })
            .unwrap();
    }

    // The outlier became its own short-lived track and has been deleted by the
    // end; the three real tracks remain.
    let confirmed: Vec<_> = last_snapshots
        .iter()
        .filter(|snapshot| snapshot.lifecycle == TrackLifecycleState::Confirmed)
        .collect();
    assert_eq!(confirmed.len(), 3, "snapshots: {last_snapshots:?}");
    let mut ids: Vec<u64> = confirmed.iter().map(|s| s.track_id.value()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[test]
fn empty_batch_from_another_sensor_does_not_drop_track() {
    let config = BBox2DTrackerConfig {
        common: base_common(),
        ..BBox2DTrackerConfig::default()
    };
    let mut tracker = BBox2DTracker::new(config).unwrap();

    tracker
        .process_batch(&ObservationBatch2D {
            sensor_id: SensorId::new(1),
            sequence: 0,
            timestamp: Timestamp::from_nanos(0),
            frame_id: FrameId::new("camera"),
            observations: vec![obs2d(100.0, 100.0, 50.0, 50.0)],
        })
        .unwrap();
    tracker
        .process_batch(&ObservationBatch2D {
            sensor_id: SensorId::new(1),
            sequence: 1,
            timestamp: Timestamp::from_nanos(100_000_000),
            frame_id: FrameId::new("camera"),
            observations: vec![obs2d(105.0, 100.0, 50.0, 50.0)],
        })
        .unwrap();

    // Sensor 2 reports an empty batch at the same time. It must not transition
    // the confirmed track to Lost.
    let snapshots = tracker
        .process_batch(&ObservationBatch2D {
            sensor_id: SensorId::new(2),
            sequence: 0,
            timestamp: Timestamp::from_nanos(150_000_000),
            frame_id: FrameId::new("camera"),
            observations: vec![],
        })
        .unwrap();
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].lifecycle, TrackLifecycleState::Confirmed);
}

fn obs3d(position: Vector3<f64>, sigma: f64) -> Position3DObservation {
    Position3DObservation {
        position,
        covariance: Matrix3::identity() * sigma * sigma,
        class_label: Some(ClassLabel::new(1)),
        appearance: None,
    }
}

#[test]
fn three_d_fuses_two_sensors_with_different_rates() {
    let config = Cartesian3DTrackerConfig {
        common: base_common(),
        accel_spectral_density: 0.5,
        gate_squared_mahalanobis: 100.0,
        ..Cartesian3DTrackerConfig::default()
    };
    let mut tracker = Cartesian3DTracker::new(config).unwrap();

    // Sensor 2 observes in its own local frame and is rigidly mounted.
    let sensor2_transform =
        RigidTransform3D::new(Matrix3::identity(), Vector3::new(5.0, 0.0, 0.0)).unwrap();

    let dt = 100_000_000; // 10 Hz base clock
    let mut last = Vec::new();
    for step in 0..30 {
        let t = Timestamp::from_nanos(step * dt);
        let truth = Vector3::new(step as f64 * 1.0, 2.0, 0.5);

        // Sensor 1: every step, low noise.
        last = tracker
            .process_batch(&ObservationBatch3D {
                sensor_id: SensorId::new(1),
                sequence: step as u64,
                timestamp: t,
                frame_id: FrameId::new("world"),
                observations: vec![obs3d(truth, 0.2)],
            })
            .unwrap();

        // Sensor 2: every third step, higher noise, local frame, and paused
        // during steps 12..=18.
        let paused = (12..=18).contains(&step);
        if step % 3 == 0 && !paused {
            let local = truth - sensor2_transform.translation();
            let world = sensor2_transform.transform_position(&local);
            let covariance = sensor2_transform
                .transform_covariance(&(Matrix3::identity() * 0.5))
                .unwrap();
            last = tracker
                .process_batch(&ObservationBatch3D {
                    sensor_id: SensorId::new(2),
                    sequence: (step / 3) as u64,
                    timestamp: t,
                    frame_id: FrameId::new("world"),
                    observations: vec![Position3DObservation {
                        position: world,
                        covariance,
                        class_label: Some(ClassLabel::new(1)),
                        appearance: None,
                    }],
                })
                .unwrap();
        }
    }

    let confirmed: Vec<_> = last
        .iter()
        .filter(|s| s.lifecycle == TrackLifecycleState::Confirmed)
        .collect();
    assert_eq!(confirmed.len(), 1);
    assert_eq!(confirmed[0].contributing_sensors.len(), 2);
    assert!((confirmed[0].position.x - 29.0).abs() < 0.5);
}

#[test]
fn transform_round_trip_preserves_position_and_covariance() {
    let rotation = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let transform = RigidTransform3D::new(rotation, Vector3::new(1.0, -2.0, 3.0)).unwrap();
    let inverse = RigidTransform3D::new(
        rotation.transpose(),
        -(rotation.transpose() * Vector3::new(1.0, -2.0, 3.0)),
    )
    .unwrap();

    let position = Vector3::new(0.5, 1.5, -0.5);
    let covariance = Matrix3::new(1.0, 0.1, 0.0, 0.1, 2.0, 0.0, 0.0, 0.0, 3.0);

    let forward = transform.transform_position(&position);
    let back = inverse.transform_position(&forward);
    assert!((back - position).norm() < 1e-12);

    let forward_cov = transform.transform_covariance(&covariance).unwrap();
    let back_cov = inverse.transform_covariance(&forward_cov).unwrap();
    assert!((back_cov - covariance).norm() < 1e-12);
}
