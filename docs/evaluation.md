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

`evaluate_sequences.sh` replays with `--initial-velocity-variance 100`
(an initial velocity standard deviation of 10 m/s; see "Initial velocity
variance" below) and the engine's default matching cascade (see "Matching
cascade").

| mode | coverage | position RMSE (m) | id switches | class accuracy | camera match rate |
| --- | --- | --- | --- | --- | --- |
| lidar_only | 1.000 | 0.106 | 15 | 1.000 | n/a |
| camera_associated | 1.000 | 0.106 | 15 | 1.000 | 0.993 |
| camera_assisted | 1.000 | 0.106 | 15 | 1.000 | 0.993 |

- Camera match rate: accepted camera associations over camera detections.
  Projected label boxes and labelled image boxes agree, which checks `P2`, the
  rectification and the velodyne/camera transforms.
- The non-zero position RMSE is filter lag (constant-velocity Kalman filter),
  since the inputs are exact.
- The remaining ID switches (0013: 8, 0004: 2, 0020: 2, 0005, 0009, 0018: 1
  each) were inspected one by one:
  - pedestrians walking less than 1 m apart swap identities (0013): real
    association ambiguity for a position-only tracker;
  - objects reappearing after more than 2 s of occlusion get a new track
    (0004, 0009), because the lost track has already been deleted;
  - label artifacts: in 0018 an oncoming car at 16 m/s stops within one frame
    as it leaves the image, and in 0005 a parked car jumps to 14 m/s within one
    frame. No tracker setting should follow those.

#### Initial velocity variance

With the library default (`initial_velocity_variance = 4`, i.e. 2 m/s) the
same run has 98 ID switches and an RMSE of 0.176 m, 44 of them in sequence
0008. A new track starts at zero velocity; with a velocity prior of 2 m/s
against a 0.5 m position noise the second observation barely moves the
velocity, so a car at 18 m/s in the world frame outruns the 0.999 gate after
two frames, a new track is spawned and the cycle repeats every three frames.
Sweep over all 21 sequences (oracle detections, `lidar_only`):

| initial velocity variance (m²/s²) | 4 | 25 | 100 | 225 | 400 |
| --- | --- | --- | --- | --- | --- |
| id switches | 98 | 31 | 18 | 17 | 20 |
| position RMSE (m) | 0.176 | 0.116 | 0.107 | 0.106 | 0.106 |

`replay` exposes `--initial-velocity-variance`, `--accel-spectral-density`,
`--gate-squared-mahalanobis` and `--recent-track-window-ns`. The library
default is unchanged; it suits the synthetic scenarios and slow objects. (The
sweep above predates the matching cascade.)

#### Matching cascade

Association cost is the squared Mahalanobis distance. A track that coasts
without observations grows its covariance, so its distance to an observation
shrinks: when two tracks sit on one object, the coasting one takes the
observation back, the other starts coasting, and the two alternate. The
engine therefore associates in two stages (the "matching cascade" of
DeepSORT): tracks observed within `recent_track_window_ns` (engine default
150 ms, i.e. the previous LiDAR scan at 10 Hz) first, the remaining tracks
against the leftover observations. In `perception-core` the option defaults to
off.

| oracle detections | without cascade | with cascade |
| --- | --- | --- |
| KITTI tracking, 21 sequences (`lidar_only` id switches) | 18 | 15 |
| CODa sequence 20, frames 8640-8800 (`lidar_only` id switches) | 158 | 58 |

### CODa (robot), oracle LiDAR detections, YOLO11x camera detections

CODa sequence 20, frames 8640-8800 (16 s): a Clearpath Husky robot driving
through a crowd on the UT Austin campus, 91 labelled dynamic objects per frame
on average (203 identities, mostly pedestrians). LiDAR detections are the
CODa 3D boxes (oracle); camera detections are YOLO11x (COCO) on the rectified
left camera. Converted with `scripts/coda/convert_coda.py` (boxes beyond 80 m
dropped: one pedestrian label in this window drifts kilometres away).

| mode | coverage | position RMSE (m) | id switches | class accuracy | camera match rate |
| --- | --- | --- | --- | --- | --- |
| lidar_only | 1.000 | 0.089 | 58 | 1.000 | n/a |
| camera_associated | 1.000 | 0.089 | 58 | 1.000 | 0.980 |
| camera_assisted | 1.000 | 0.089 | 58 | 1.000 | 0.980 |

- Camera match rate: accepted associations over YOLO detections in view of a
  projected track (1,938 of 1,977); this checks the CODa LiDAR-camera
  calibration path (`R_rect * T_os1_cam0`, rectified projection).
- About half of the 58 switches are two pedestrians closer than ~0.6 m
  exchanging tracks in the same frame, i.e. the ambiguity appearance (ReID)
  would address; the rest are tracks re-spawned after an abrupt label jump.

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
