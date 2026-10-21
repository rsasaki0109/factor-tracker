//! KITTI adapter tests using small synthetic files in the KITTI text format.

use std::path::PathBuf;

use factor_tracker_engine::kitti::{
    parse_objects_2d, parse_objects_3d, parse_times, parse_tracking_labels_3d, prepare_sequence,
    KittiCalibration, KittiClassMap, KittiSequencePaths,
};

fn temp_dir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "factor-tracker-kitti-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const CALIB: &str = "\
P0: 700 0 621 0 0 700 187.5 0 0 0 1 0
R0_rect: 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0
";

#[test]
fn calibration_maps_velodyne_to_camera_correctly() {
    let calib = KittiCalibration::parse(CALIB).unwrap();
    let (calibration, velo_to_cam) = calib.to_calibration(1242, 375, 0.3).unwrap();
    // A point 10 m in front of the camera is at (10, 0, 0) in the velodyne
    // frame (x forward).
    let camera_point = nalgebra::Vector3::new(0.0, 0.0, 10.0);
    let velo_point = calibration
        .camera_to_reference
        .transform_position(&camera_point);
    assert!((velo_point.x - 10.0).abs() < 1e-9);
    assert!(velo_point.y.abs() < 1e-9);
    assert!(velo_point.z.abs() < 1e-9);
    // The forward transform is consistent.
    let back = velo_to_cam.transform_position(&velo_point);
    assert!((back - camera_point).norm() < 1e-9);
    // And it projects to the principal point.
    let pixel = calibration.intrinsics.project(&camera_point).unwrap();
    assert!((pixel.x - 621.0).abs() < 1e-9);
    assert!((pixel.y - 187.5).abs() < 1e-9);
}

#[test]
fn parses_detection_and_label_lines() {
    let classes = KittiClassMap::default();
    let objects = parse_objects_3d(
        "Car 0 0 0 600 150 700 250 1.5 1.8 4.0 0.5 0.2 10.0 0.1 0.9",
        &classes,
    )
    .unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].class_id, Some(1));
    assert!((objects[0].location[2] - 10.0).abs() < 1e-9);

    let objects_2d = parse_objects_2d("Pedestrian 0 0 0 10 20 30 40 0.8", &classes).unwrap();
    assert_eq!(objects_2d[0].class_id, Some(2));
    assert_eq!(objects_2d[0].bbox, [10.0, 20.0, 30.0, 40.0]);

    let labels = parse_tracking_labels_3d(
        "0 7 Car 0 0 0 600 150 700 250 1.5 1.8 4.0 0.5 0.2 10.0 0.1",
        &classes,
    )
    .unwrap();
    assert_eq!(labels[0].0, 7);
    assert_eq!(labels[0].1.class_id, Some(1));
}

#[test]
fn prepares_a_sequence_directory() {
    let dir = temp_dir("prepare");
    let calib = dir.join("calib.txt");
    std::fs::write(&calib, CALIB).unwrap();
    let times = dir.join("times.txt");
    std::fs::write(&times, "0.0\n0.1\n").unwrap();
    let lidar_dir = dir.join("detections_lidar");
    let camera_dir = dir.join("detections_camera");
    std::fs::create_dir_all(&lidar_dir).unwrap();
    std::fs::create_dir_all(&camera_dir).unwrap();
    std::fs::write(
        lidar_dir.join("000000.txt"),
        "Car 0 0 0 600 150 700 250 1.5 1.8 4.0 0.5 0.2 10.0 0.1 0.9\n",
    )
    .unwrap();
    std::fs::write(
        camera_dir.join("000000.txt"),
        "Car 0 0 0 600 150 700 250 0.9\n",
    )
    .unwrap();

    let paths = KittiSequencePaths {
        calib,
        times,
        poses: None,
        lidar_detections: lidar_dir,
        camera_detections: camera_dir,
        lidar_labels: None,
        camera_labels: None,
        lidar_sensor_id: 1,
        camera_sensor_id: 2,
    };
    let prepared = prepare_sequence(&paths, 1242, 375, 0.3, 0.25, 200_000_000).unwrap();
    assert_eq!(prepared.inputs.len(), 2);
    let (batch, arrival) = prepared.inputs[0].clone().into_batch().unwrap();
    assert!(batch.is_lidar());
    assert_eq!(arrival.as_nanos(), 0);
    // Camera (0.5, 0.2, 10) is velodyne (10, -0.5, -0.2) for this calibration.
    let factor_tracker_engine::input::EngineBatch::Lidar(lidar) = batch else {
        unreachable!()
    };
    let position = lidar.detections[0].position;
    assert!((position - nalgebra::Vector3::new(10.0, -0.5, -0.2)).norm() < 1e-9);
    assert_eq!(parse_times("0.0\n0.1\n").unwrap().len(), 2);
}

/// Tracking-benchmark layout: `R_rect`/`Tr_velo_cam` without a colon, a `3x3`
/// rectifying rotation and a `P2` with a non-zero translation column.
const TRACKING_CALIB: &str = "\
P0: 700 0 621 0 0 700 187.5 0 0 0 1 0
P2: 700 0 621 42 0 700 187.5 0.7 0 0 1 0.005
R_rect 1 0 0 0 1 0 0 0 1
Tr_velo_cam 0 -1 0 0 0 0 -1 0 1 0 0 0
Tr_imu_velo 1 0 0 0 0 1 0 0 0 0 1 0
";

#[test]
fn parses_tracking_calibration_and_projects_through_p2() {
    let calib = KittiCalibration::parse(TRACKING_CALIB).unwrap();
    let offset = calib.camera_offset();
    // t = K^-1 p4
    assert!((offset.z - 0.005).abs() < 1e-12);
    assert!((offset.y - (0.7 - 187.5 * 0.005) / 700.0).abs() < 1e-12);
    assert!((offset.x - (42.0 - 621.0 * 0.005) / 700.0).abs() < 1e-12);

    let (calibration, velo_to_cam0) = calib.to_calibration(1242, 375, 0.3).unwrap();
    let cam0_point = nalgebra::Vector3::new(1.5, -0.5, 12.0);
    let velo_point =
        factor_tracker_engine::geometry::inverse(&velo_to_cam0).transform_position(&cam0_point);
    let camera_point = factor_tracker_engine::geometry::inverse(&calibration.camera_to_reference)
        .transform_position(&velo_point);
    let pixel = calibration.intrinsics.project(&camera_point).unwrap();

    // Reference: x = P2 * [X_cam0; 1].
    let p2 = nalgebra::Matrix3x4::new(
        700.0, 0.0, 621.0, 42.0, 0.0, 700.0, 187.5, 0.7, 0.0, 0.0, 1.0, 0.005,
    );
    let h = p2 * nalgebra::Vector4::new(cam0_point.x, cam0_point.y, cam0_point.z, 1.0);
    assert!((pixel.x - h.x / h.z).abs() < 1e-9);
    assert!((pixel.y - h.y / h.z).abs() < 1e-9);
}

#[test]
fn ignores_non_numeric_raw_calibration_entries() {
    let text = "calib_time: 09-Jan-2012 13:57:47\n".to_string() + CALIB;
    assert!(KittiCalibration::parse(&text).is_ok());
}

#[test]
fn reads_2d_score_from_full_detection_lines() {
    let classes = KittiClassMap::default();
    // 16-field detection line: the score is the last field.
    let detection = parse_objects_2d(
        "Car -1 -1 -10 10 20 30 40 -1 -1 -1 -1000 -1000 -1000 -10 0.75",
        &classes,
    )
    .unwrap();
    assert!((detection[0].score - 0.75).abs() < 1e-12);
    // 15-field label line used as an oracle detection: no score, so 1.0.
    let label = parse_objects_2d(
        "Car 0 0 0 10 20 30 40 1.5 1.8 4.0 0.5 0.2 10.0 0.1",
        &classes,
    )
    .unwrap();
    assert!((label[0].score - 1.0).abs() < 1e-12);
}

#[test]
fn skips_dont_care_tracking_labels() {
    let classes = KittiClassMap::default();
    let labels = parse_tracking_labels_3d(
        "0 -1 DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10\n\
         0 3 Car 0 0 0 600 150 700 250 1.5 1.8 4.0 0.5 0.2 10.0 0.1\n",
        &classes,
    )
    .unwrap();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].0, 3);
}
