//! Evaluation of replay results against optional ground truth.
//!
//! Ground truth is only used here. It is never passed to a tracker.
//!
//! The metrics are deliberately simple and are named after what they actually
//! compute. They are not HOTA, IDF1 or MOTA. Matching condition: at each
//! evaluated timestamp, tracks and ground-truth objects are matched one-to-one
//! by Euclidean distance with a hard distance gate (the same gate is used for
//! every mode).

use std::collections::{BTreeMap, HashMap};

use factor_tracker_core::linear_assignment;
use serde::{Deserialize, Serialize};

use crate::diagnostics::AllCounters;
use crate::engine::{CameraFrameOutput, LidarFrameOutput, TrackingMode};

/// A ground-truth object at one time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundTruthObject {
    /// Ground-truth object id.
    pub truth_id: u32,
    /// Class id in the tracking class space.
    pub class: u32,
    /// Position `[x, y, z]` in the tracking frame, metres.
    pub position: Vec<f64>,
}

/// Ground truth at one timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundTruthFrame {
    /// Timestamp.
    pub timestamp_ns: i64,
    /// Objects.
    pub objects: Vec<GroundTruthObject>,
}

/// One evaluation-only detection-to-truth mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraDetectionTruth {
    /// Camera detection id.
    pub detection_id: u64,
    /// Ground-truth object id.
    pub truth_id: u32,
}

/// Optional evaluation-only mappings from camera detection ids to truth ids.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraAssociationTruth {
    /// Camera batch timestamp.
    pub timestamp_ns: i64,
    /// Mappings present at this timestamp.
    pub entries: Vec<CameraDetectionTruth>,
}

impl CameraAssociationTruth {
    /// Looks up the truth id for a detection.
    pub fn truth_for(&self, detection_id: u64) -> Option<u32> {
        self.entries
            .iter()
            .find(|entry| entry.detection_id == detection_id)
            .map(|entry| entry.truth_id)
    }
}

/// Evaluation configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalConfig {
    /// Hard matching gate in metres.
    pub match_distance_m: f64,
    /// Only confirmed tracks are matched (tentative and lost tracks are what
    /// a consumer would not act on). When `false`, every output track counts.
    pub confirmed_only: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        EvalConfig {
            match_distance_m: 2.0,
            confirmed_only: true,
        }
    }
}

/// A robot-centric evaluation zone: ground truth outside it is dropped before
/// matching, so metrics describe what matters to the robot (nearby objects,
/// ahead of it) rather than the whole sensor range.
///
/// The zone is defined in the reference frame (x forward, y left, as for the
/// KITTI velodyne, the CODa Ouster and the GrandTour base frame).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalZone {
    /// Maximum horizontal distance from the reference origin, metres.
    pub max_range_m: f64,
    /// Maximum absolute bearing from the +x axis, radians (`PI` = all around).
    pub max_bearing_rad: f64,
}

impl EvalZone {
    /// Whether a reference-frame position is inside the zone.
    pub fn contains(&self, position: &nalgebra::Vector3<f64>) -> bool {
        position.x.hypot(position.y) <= self.max_range_m
            && position.y.atan2(position.x).abs() <= self.max_bearing_rad
    }
}

/// Keeps only the ground truth inside `zone`. Truth positions are in the
/// tracking frame; with `poses` (reference to world) they are mapped back to
/// the reference frame at their own timestamp, without poses the tracking
/// frame is the reference frame. Frames whose pose is unavailable are dropped.
pub fn filter_truth_to_zone(
    truth: &[GroundTruthFrame],
    poses: Option<&crate::geometry::PoseTrack>,
    zone: &EvalZone,
) -> Vec<GroundTruthFrame> {
    truth
        .iter()
        .filter_map(|frame| {
            let world_to_reference = match poses {
                Some(poses) => Some(crate::geometry::inverse(
                    &poses
                        .pose_at(factor_tracker_core::Timestamp::from_nanos(
                            frame.timestamp_ns,
                        ))
                        .ok()?,
                )),
                None => None,
            };
            let objects = frame
                .objects
                .iter()
                .filter(|object| {
                    let Some(p) = vector3(&object.position) else {
                        return false;
                    };
                    let p = nalgebra::Vector3::new(p[0], p[1], p[2]);
                    let p = match &world_to_reference {
                        Some(t) => t.transform_position(&p),
                        None => p,
                    };
                    zone.contains(&p)
                })
                .cloned()
                .collect();
            Some(GroundTruthFrame {
                timestamp_ns: frame.timestamp_ns,
                objects,
            })
        })
        .collect()
}

/// Metrics for one tracking mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModeMetrics {
    /// Mode.
    pub mode: TrackingMode,
    /// LiDAR frames evaluated.
    pub frames_evaluated: u64,
    /// Ground-truth object-timestamps considered.
    pub gt_object_times: u64,
    /// Object-timestamps with a matched track.
    pub matched_object_times: u64,
    /// Matched fraction in `[0, 1]`.
    pub track_coverage: f64,
    /// Position RMSE over matched pairs (metres), if any.
    pub position_rmse: Option<f64>,
    /// Mean absolute position error (metres), if any.
    pub position_mean_abs: Option<f64>,
    /// Median absolute position error (metres), if any.
    pub position_median_abs: Option<f64>,
    /// 95th percentile absolute position error (metres), if any.
    pub position_p95_abs: Option<f64>,
    /// Number of track-id changes for a persistent ground-truth object between
    /// consecutive matched timestamps.
    pub id_switches: u64,
    /// Matched pairs whose track class equals the truth class.
    pub class_correct: u64,
    /// Matched pairs where the track had a known class.
    pub class_total: u64,
    /// Accepted camera associations that agree with the evaluation mapping.
    pub camera_assoc_correct: u64,
    /// Accepted camera associations for which a truth mapping existed.
    pub camera_assoc_total: u64,
    /// Camera association accuracy, if measurable.
    pub camera_assoc_accuracy: Option<f64>,
    /// Diagnostic counters.
    pub counters: AllCounters,
}

/// Returns the matched `(track_index, truth_index, distance)` triples for one
/// timestamp.
fn match_frame(
    tracks: &[(u64, [f64; 3])],
    objects: &[(u32, [f64; 3])],
    max_distance: f64,
) -> Vec<(usize, usize, f64)> {
    let n_tracks = tracks.len();
    let n_objects = objects.len();
    let mut costs = vec![vec![f64::INFINITY; n_objects]; n_tracks];
    for (i, (_, track)) in tracks.iter().enumerate() {
        for (j, (_, object)) in objects.iter().enumerate() {
            let distance = ((track[0] - object[0]).powi(2)
                + (track[1] - object[1]).powi(2)
                + (track[2] - object[2]).powi(2))
            .sqrt();
            if distance <= max_distance {
                costs[i][j] = distance;
            }
        }
    }
    let assignment = match linear_assignment(&costs, n_objects, max_distance, 0.0) {
        Ok(assignment) => assignment,
        Err(_) => return Vec::new(),
    };
    assignment
        .pairs
        .iter()
        .map(|&(i, j)| (i, j, costs[i][j]))
        .collect()
}

fn vector3(values: &[f64]) -> Option<[f64; 3]> {
    if values.len() != 3 {
        return None;
    }
    Some([values[0], values[1], values[2]])
}

/// Evaluates one mode's replay output against ground truth.
pub fn evaluate_mode(
    lidar_frames: &[LidarFrameOutput],
    camera_frames: &[CameraFrameOutput],
    truth: &[GroundTruthFrame],
    camera_truth: &[CameraAssociationTruth],
    counters: AllCounters,
    config: &EvalConfig,
) -> ModeMetrics {
    let mode = lidar_frames
        .first()
        .map(|frame| frame.mode)
        .or_else(|| camera_frames.first().map(|frame| frame.mode))
        .unwrap_or(TrackingMode::LidarOnly);

    let truth_by_time: BTreeMap<i64, &GroundTruthFrame> = truth
        .iter()
        .map(|frame| (frame.timestamp_ns, frame))
        .collect();

    // Track-id -> truth-id, rebuilt per timestamp, plus the last known mapping
    // for camera-time lookups.
    let mut last_track_truth: HashMap<u64, u32> = HashMap::new();

    let mut position_errors: Vec<f64> = Vec::new();
    let mut gt_object_times: u64 = 0;
    let mut matched_object_times: u64 = 0;
    let mut class_correct: u64 = 0;
    let mut class_total: u64 = 0;

    // Per truth object: the sequence of matched track ids across time.
    let mut truth_track_history: BTreeMap<u32, Vec<(i64, u64)>> = BTreeMap::new();
    let mut truth_presence: BTreeMap<u32, u64> = BTreeMap::new();

    let mut frames_evaluated: u64 = 0;
    for frame in lidar_frames {
        let Some(truth_frame) = truth_by_time.get(&frame.timestamp_ns) else {
            continue;
        };
        frames_evaluated += 1;
        let tracks: Vec<(u64, [f64; 3])> = frame
            .tracks
            .iter()
            .filter(|track| {
                !config.confirmed_only
                    || track.lifecycle
                        == factor_tracker_core::TrackLifecycleState::Confirmed.as_str()
            })
            .filter_map(|track| vector3(&track.position).map(|p| (track.track_id, p)))
            .collect();
        // A pose-corrected frame was tracked in the corrected frame: map the
        // truth (given with the uncorrected poses) into it.
        let correct = |p: [f64; 3]| match frame.pose_correction.as_deref() {
            Some(m) if m.len() == 16 => {
                let row = |r: usize| {
                    m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3]
                };
                [row(0), row(1), row(2)]
            }
            _ => p,
        };
        let (objects, object_classes): (Vec<(u32, [f64; 3])>, Vec<u32>) = truth_frame
            .objects
            .iter()
            .filter_map(|object| {
                vector3(&object.position).map(|p| ((object.truth_id, correct(p)), object.class))
            })
            .unzip();
        gt_object_times += objects.len() as u64;
        for object in &truth_frame.objects {
            *truth_presence.entry(object.truth_id).or_insert(0) += 1;
        }

        let matches = match_frame(&tracks, &objects, config.match_distance_m);
        for &(i, j, distance) in &matches {
            position_errors.push(distance);
            matched_object_times += 1;
            let (track_id, _) = tracks[i];
            let (truth_id, _) = objects[j];
            truth_track_history
                .entry(truth_id)
                .or_default()
                .push((frame.timestamp_ns, track_id));
            last_track_truth.insert(track_id, truth_id);

            if let Some(track) = frame.tracks.iter().find(|t| t.track_id == track_id) {
                if let Some(class) = track.class_label {
                    class_total += 1;
                    if class == object_classes[j] {
                        class_correct += 1;
                    }
                }
            }
        }
    }

    let id_switches = truth_track_history
        .values()
        .map(|history| {
            history
                .windows(2)
                .filter(|window| window[0].1 != window[1].1)
                .count() as u64
        })
        .sum();

    let mut position_errors_sorted = position_errors.clone();
    position_errors_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let position_rmse = if position_errors.is_empty() {
        None
    } else {
        Some(
            (position_errors.iter().map(|e| e * e).sum::<f64>() / position_errors.len() as f64)
                .sqrt(),
        )
    };
    let position_mean_abs = if position_errors.is_empty() {
        None
    } else {
        Some(position_errors.iter().sum::<f64>() / position_errors.len() as f64)
    };
    let position_median_abs = if position_errors_sorted.is_empty() {
        None
    } else {
        Some(position_errors_sorted[position_errors_sorted.len() / 2])
    };
    let position_p95_abs = if position_errors_sorted.is_empty() {
        None
    } else {
        let index = ((position_errors_sorted.len() as f64 * 0.95) as usize)
            .min(position_errors_sorted.len() - 1);
        Some(position_errors_sorted[index])
    };

    let track_coverage = if gt_object_times == 0 {
        0.0
    } else {
        matched_object_times as f64 / gt_object_times as f64
    };

    // Camera association accuracy.
    let camera_truth_by_time: BTreeMap<i64, &CameraAssociationTruth> = camera_truth
        .iter()
        .map(|item| (item.timestamp_ns, item))
        .collect();
    let mut camera_assoc_correct = 0u64;
    let mut camera_assoc_total = 0u64;
    for frame in camera_frames {
        let Some(truth_frame) = camera_truth_by_time.get(&frame.timestamp_ns) else {
            continue;
        };
        for m in &frame.matches {
            if !m.accepted {
                continue;
            }
            let Some(track_id) = m.track_id else {
                continue;
            };
            let Some(detection_truth) = truth_frame.truth_for(m.detection_id) else {
                continue;
            };
            let Some(track_truth) = last_track_truth.get(&track_id) else {
                continue;
            };
            camera_assoc_total += 1;
            if detection_truth == *track_truth {
                camera_assoc_correct += 1;
            }
        }
    }
    let camera_assoc_accuracy = if camera_assoc_total == 0 {
        None
    } else {
        Some(camera_assoc_correct as f64 / camera_assoc_total as f64)
    };

    ModeMetrics {
        mode,
        frames_evaluated,
        gt_object_times,
        matched_object_times,
        track_coverage,
        position_rmse,
        position_mean_abs,
        position_median_abs,
        position_p95_abs,
        id_switches,
        class_correct,
        class_total,
        camera_assoc_correct,
        camera_assoc_total,
        camera_assoc_accuracy,
        counters,
    }
}

/// Result of comparing several modes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// One entry per mode.
    pub modes: Vec<ModeMetrics>,
}

/// Helper to look up the track-id -> truth-id mapping from a set of lidar
/// frames (used by tests).
pub fn track_truth_map(
    frames: &[LidarFrameOutput],
    truth: &[GroundTruthFrame],
    max_distance: f64,
) -> BTreeMap<u64, u32> {
    let truth_by_time: BTreeMap<i64, &GroundTruthFrame> =
        truth.iter().map(|f| (f.timestamp_ns, f)).collect();
    let mut map = BTreeMap::new();
    for frame in frames {
        let Some(truth_frame) = truth_by_time.get(&frame.timestamp_ns) else {
            continue;
        };
        let tracks: Vec<(u64, [f64; 3])> = frame
            .tracks
            .iter()
            .filter_map(|t| vector3(&t.position).map(|p| (t.track_id, p)))
            .collect();
        let objects: Vec<(u32, [f64; 3])> = truth_frame
            .objects
            .iter()
            .filter_map(|o| vector3(&o.position).map(|p| (o.truth_id, p)))
            .collect();
        for (i, j, _) in match_frame(&tracks, &objects, max_distance) {
            map.insert(tracks[i].0, objects[j].0);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TrackOutput;

    fn track(track_id: u64, position: [f64; 3], class_label: Option<u32>) -> TrackOutput {
        TrackOutput {
            track_id,
            lifecycle: "confirmed".to_string(),
            state_timestamp_ns: 0,
            last_spatial_update_ns: Some(0),
            last_camera_support_ns: None,
            age_since_spatial_update_ns: 0,
            prediction_only: false,
            position: position.to_vec(),
            velocity: vec![0.0; 3],
            covariance: vec![0.0; 36],
            class_label,
            class_prior: None,
            contributing_sensors: vec![1],
            observation_times_count: 1,
        }
    }

    #[test]
    fn zone_keeps_nearby_objects_ahead_of_the_robot() {
        let zone = EvalZone {
            max_range_m: 5.0,
            max_bearing_rad: 45f64.to_radians(),
        };
        let object = |truth_id, x: f64, y: f64| GroundTruthObject {
            truth_id,
            class: 2,
            position: vec![x, y, 0.0],
        };
        let frames = vec![GroundTruthFrame {
            timestamp_ns: 0,
            objects: vec![
                object(1, 3.0, 1.0),  // ahead, near
                object(2, 3.0, 4.0),  // near but 53 degrees to the left
                object(3, -2.0, 0.0), // behind
                object(4, 9.0, 0.0),  // ahead but too far
            ],
        }];
        let kept = filter_truth_to_zone(&frames, None, &zone);
        let ids: Vec<u32> = kept[0].objects.iter().map(|o| o.truth_id).collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn class_accuracy_compares_against_the_truth_class_not_the_truth_id() {
        let frame = LidarFrameOutput {
            mode: TrackingMode::LidarOnly,
            timestamp_ns: 0,
            tracks: vec![
                track(1, [0.0, 0.0, 0.0], Some(4)),
                track(2, [10.0, 0.0, 0.0], Some(7)),
            ],
            observations: Vec::new(),
            associations: Vec::new(),
            unmatched_track_ids: Vec::new(),
            pose_correction: None,
        };
        let truth = GroundTruthFrame {
            timestamp_ns: 0,
            objects: vec![
                GroundTruthObject {
                    truth_id: 7,
                    class: 4,
                    position: vec![0.0, 0.0, 0.0],
                },
                GroundTruthObject {
                    truth_id: 4,
                    class: 2,
                    position: vec![10.0, 0.0, 0.0],
                },
            ],
        };
        let metrics = evaluate_mode(
            &[frame],
            &[],
            &[truth],
            &[],
            AllCounters::default(),
            &EvalConfig::default(),
        );
        assert_eq!(metrics.class_total, 2);
        assert_eq!(metrics.class_correct, 1);
    }

    #[test]
    fn truth_is_mapped_into_a_pose_corrected_frame() {
        // The scan was tracked in a frame shifted by +1.5 m in x.
        let frame = LidarFrameOutput {
            mode: TrackingMode::LidarOnly,
            timestamp_ns: 0,
            tracks: vec![track(1, [1.5, 0.0, 0.0], None)],
            observations: Vec::new(),
            associations: Vec::new(),
            unmatched_track_ids: Vec::new(),
            pose_correction: Some(vec![
                1.0, 0.0, 0.0, 1.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
        };
        let truth = GroundTruthFrame {
            timestamp_ns: 0,
            objects: vec![GroundTruthObject {
                truth_id: 1,
                class: 2,
                position: vec![0.0, 0.0, 0.0],
            }],
        };
        let metrics = evaluate_mode(
            &[frame],
            &[],
            &[truth],
            &[],
            AllCounters::default(),
            &EvalConfig::default(),
        );
        assert_eq!(metrics.matched_object_times, 1);
        assert!(metrics.position_rmse.unwrap() < 1e-9);
    }
}
