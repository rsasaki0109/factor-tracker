# Evaluation and the three modes

## Modes

The same input recording is replayed in three modes so that "labelled" and
"changed processing" can be distinguished:

| Mode | Camera association | Class evidence | Class prior applied | Position updated by camera |
| --- | --- | --- | --- | --- |
| `lidar_only` | no | no | no | no |
| `camera_associated` | yes | recorded | no | no |
| `camera_assisted` | yes | recorded | yes | no |

All modes share the same LiDAR detections, calibration, transforms, filter
settings, evaluation interval and injected latency/misses. The camera never
updates the 3D position, so **position accuracy is not assumed to improve** in
the camera modes.

## Injection experiments

`inject::apply` deterministically injects camera delay, LiDAR delay,
arrival-order reversal, duplicate batches and sensor pauses into the same input
recording. These are software experiments; they do not represent every real
sensor failure.

## Metrics

All metrics are computed by `perception_engine::eval`. Matching condition: at
each evaluated timestamp, tracks and ground-truth objects are matched one-to-one
by Euclidean distance with a hard gate (`EvalConfig::match_distance_m`, default
2 m). The same gate is used for every mode.

| Metric | Definition |
| --- | --- |
| `track_coverage` | matched object-timestamps / ground-truth object-timestamps |
| `position_rmse` | RMSE of Euclidean position error over matched pairs (m) |
| `position_mean_abs` | mean absolute position error over matched pairs (m) |
| `position_median_abs` | median absolute position error (m) |
| `position_p95_abs` | 95th percentile absolute position error (m) |
| `id_switches` | for each truth object, the number of times the matched track id changes between consecutive matched timestamps |
| `class_correct` / `class_total` | matched pairs where the track's positional class equals the truth class, among pairs where the track has a known class |
| `camera_assoc_accuracy` | accepted camera associations that agree with the evaluation detection-to-truth mapping, over those for which a mapping exists |
| counters | reorder and engine counters (late, duplicate, overflow, transform_unavailable, association_rejected, ...) |

These are simple, explicitly defined metrics. They are **not** HOTA, IDF1 or
MOTA, and no official metric name is used. Metrics that cannot be measured for a
given recording are reported as unavailable (`null` / `n/a`).

## Reported results

`cam-demo` prints and reports the three modes. With the synthetic fixture
(seed 42, no injected faults), the observed results are:

- `track_coverage = 1.00` for all three modes,
- identical 3D position metrics for all three modes (the camera does not update
  position),
- `camera_assoc_accuracy = 1.00` for `camera_associated` and
  `camera_assisted` and not measured (`n/a`) for `lidar_only`,
- `class_acc = 1.00` for all three modes (class is a LiDAR observation and the
  fixture's LiDAR classes are correct).

Earlier versions reported a class accuracy below 1.0 here. That was an
evaluation bug: the track class was compared with the truth *id* instead of the
truth class.

In the synthetic fixture the LiDAR association is unambiguous, so
`camera_assisted` does not change the tracking result relative to
`camera_associated`. The engine test suite contains a separate, constructed
ambiguous example where the camera class prior changes which LiDAR detection is
associated (see `crates/perception-engine/tests/camera_lidar.rs`). That
constructed example is a mechanism check only and is **not** evidence of a
general real-world accuracy improvement.

### KITTI tracking, oracle detections

All 21 training sequences of the KITTI tracking benchmark (8,008 frames,
47,262 labelled object-timestamps), converted with
`scripts/kitti/convert_tracking.py --oracle` and evaluated with
`scripts/kitti/evaluate_sequences.sh`. The LiDAR and camera "detections" are the
ground-truth labels themselves, so this is a check of the adapter, the
calibration/projection and pose handling, and the tracker on real geometry and
timing. It is **not** a detection or tracking benchmark result.

| mode | coverage | position RMSE (m) | id switches | class accuracy | camera match rate |
| --- | --- | --- | --- | --- | --- |
| lidar_only | 1.000 | 0.176 | 98 | 1.000 | n/a |
| camera_associated | 1.000 | 0.176 | 98 | 1.000 | 0.994 |
| camera_assisted | 1.000 | 0.176 | 98 | 1.000 | 0.994 |

- Camera match rate: accepted camera associations over camera detections
  (46,958 / 47,262). Projected label boxes and labelled image boxes agree, which
  checks `P2`, the rectification and the velodyne/camera transforms.
- The non-zero position RMSE is filter lag (constant-velocity Kalman filter),
  since the inputs are exact.
- The ID switches happen even with perfect detections and are concentrated in
  a few sequences (0008: 44, 0018: 17, 0020: 14). They are a baseline to
  investigate, not yet explained.

Detector-based results will be added once `scripts/runpod/` has been run.

## Reproducing

```sh
cargo run -p perception-cli -- cam-demo --out-dir cam-demo-out
# comparison table: cam-demo-out/comparison.json
# report:           cam-demo-out/report.html
```

To inject faults:

```sh
cargo run -p perception-cli -- cam-demo --out-dir cam-demo-out \
    --camera-delay-ns 150000000 --lidar-delay-ns 0 \
    --reverse-window 3 --duplicate-every 7 \
    --pause-sensor camera --pause-start-ns 1000000000 --pause-end-ns 2000000000
```

The corresponding counters appear in `comparison.json` and the report's
diagnostics panel.
