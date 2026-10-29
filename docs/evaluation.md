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

All metrics are computed by `factor_tracker_engine::eval`. Matching condition: at
each evaluated timestamp, tracks and ground-truth objects are matched one-to-one
by Euclidean distance with a hard gate (`EvalConfig::match_distance_m`, default
2 m; `evaluate --match-distance-m`). Only **confirmed** tracks are matched
(`EvalConfig::confirmed_only`, default; `evaluate --all-tracks` matches every
output track): tentative and lost tracks are not what a consumer acts on, and
counting them inflates coverage. The same settings are used for every mode.

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

- `track_coverage = 0.97` for all three modes (the first frames of each
  object, before its track is confirmed, are not matched),
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
associated (see `crates/factor-tracker-engine/tests/camera_lidar.rs`). That
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
cascade"); evaluation uses the default 2 m gate and confirmed tracks only.

| mode | coverage | position RMSE (m) | id switches | class accuracy | camera match rate |
| --- | --- | --- | --- | --- | --- |
| lidar_only | 0.981 | 0.107 | 15 | 1.000 | n/a |
| camera_associated | 0.981 | 0.107 | 15 | 1.000 | 0.993 |
| camera_assisted | 0.981 | 0.107 | 15 | 1.000 | 0.993 |

- Coverage is below 1 with perfect detections because a track only counts once
  it is confirmed.
- Camera match rate: accepted camera associations over camera detections.
  Projected label boxes and labelled image boxes agree, which checks `P2`, the
  rectification and the velodyne/camera transforms.
- The non-zero position RMSE is filter lag (constant-velocity Kalman filter),
  since the inputs are exact.
- The remaining ID switches (0013: 8, 0004: 2, 0020: 2, 0005, 0009, 0019: 1
  each) fall into three groups when inspected one by one:
  - pedestrians walking less than 1 m apart swap identities (0013): real
    association ambiguity for a position-only tracker;
  - objects reappearing after more than 2 s of occlusion get a new track
    (0004, 0009), because the lost track has already been deleted;
  - label artifacts, e.g. in 0005 a parked car jumps to 14 m/s within one
    frame (and in 0018 an oncoming car at 16 m/s stops within one frame as it
    leaves the image). No tracker setting should follow those.

#### Initial velocity variance

With the library default (`initial_velocity_variance = 4`, i.e. 2 m/s) the
KITTI run had 98 ID switches and an RMSE of 0.176 m, 44 of them in sequence
0008. A new track starts at zero velocity; with a velocity prior of 2 m/s
against a 0.5 m position noise the second observation barely moves the
velocity, so a car at 18 m/s in the world frame outruns the 0.999 gate after
two frames, a new track is spawned and the cycle repeats every three frames.
Sweep over all 21 sequences (oracle detections, `lidar_only`; measured before
the matching cascade and with every output track matched):

| initial velocity variance (m²/s²) | 4 | 25 | 100 | 225 | 400 |
| --- | --- | --- | --- | --- | --- |
| id switches | 98 | 31 | 18 | 17 | 20 |
| position RMSE (m) | 0.176 | 0.116 | 0.107 | 0.106 | 0.106 |

Pedestrians need the opposite: on CODa (below) a 10 m/s prior lets new tracks
gate far and take neighbours' detections, and `--initial-velocity-variance 1
--accel-spectral-density 0.5` gives fewer switches than the vehicle setting
(real detections, score >= 0.4, 1 m gate, before tuning the cascade window:
597 with 100 / 1.0, 466 with 1 / 0.5). The motion prior is a per-deployment
setting, not a constant.

`replay` exposes `--initial-velocity-variance`, `--accel-spectral-density`,
`--gate-squared-mahalanobis` and `--recent-track-window-ns`. The library
defaults are unchanged.

#### Matching cascade

Association cost is the squared Mahalanobis distance. A track that coasts
without observations grows its covariance, so its distance to an observation
shrinks: when two tracks sit on one object, the coasting one takes the
observation back, the other starts coasting, and the two alternate. The
engine therefore associates in two stages (the "matching cascade" of
DeepSORT): tracks observed within `recent_track_window_ns` first, the
remaining tracks against the leftover observations. In `factor-tracker-core` the
option defaults to off; the engine uses 350 ms (at 10 Hz: tolerate up to two
missed scans before a track loses priority).

The window matters once detections are missed. With a window of one scan a
track that misses a single detection drops to the second stage and its next
detection can go to a neighbour. CODa sequence 20 window, pedestrian motion
prior, 1 m gate, `lidar_only` ID switches:

| cascade window | off | 150 ms | 250 ms | 350 ms | 550 ms | 1050 ms |
| --- | --- | --- | --- | --- | --- | --- |
| oracle detections | 111 | 29 | 28 | 31 | 38 | 58 |
| real detections, score >= 0.3 | 666 | 545 | 557 | 529 | 534 | 605 |
| real detections, score >= 0.4 | 439 | 466 | 377 | 370 | 357 | 384 |
| real detections, score >= 0.5 | 383 | 431 | 342 | 328 | 295 | 307 |

KITTI tracking (oracle, 21 sequences): 18 without the cascade, 15 with it (150
and 350 ms alike).

### CODa (robot)

CODa sequence 20, frames 8640-8800 (16 s): a Clearpath Husky robot driving
through a crowd on the UT Austin campus, 91 labelled dynamic objects per frame
on average (203 identities, mostly pedestrians). Camera detections are
YOLO11x (COCO) on the rectified left camera. Converted with
`scripts/coda/convert_coda.py` (boxes beyond 80 m dropped: one pedestrian label
in this window drifts kilometres away) and evaluated with

```sh
REPLAY_ARGS="--initial-velocity-variance 1 --accel-spectral-density 0.5" \
EVAL_ARGS="--match-distance-m 1" scripts/kitti/evaluate_sequences.sh coda-seq coda-res 20
```

The 1 m gate is used because the median distance from a pedestrian to the
nearest other pedestrian in this window is 1.2 m; a 2 m gate would match
tracks to the wrong person.

#### Oracle LiDAR detections (the CODa 3D boxes)

| mode | coverage | position RMSE (m) | id switches | class accuracy | camera match rate |
| --- | --- | --- | --- | --- | --- |
| lidar_only | 0.987 | 0.103 | 31 | 1.000 | n/a |
| camera_associated | 0.987 | 0.103 | 31 | 1.000 | 0.979 |
| camera_assisted | 0.987 | 0.103 | 31 | 1.000 | 0.979 |

The camera match rate checks the CODa LiDAR-camera calibration path
(`R_rect * T_os1_cam0`, rectified projection). About half of the remaining
switches are two pedestrians closer than ~0.6 m exchanging tracks.

#### Real LiDAR detections (CODa PV-RCNN, first stage, CPU)

`scripts/coda/detect_lidar_cpu.py` runs the published CODa checkpoint
`coda128_allclass_bestoracle.pth` without CUDA: the sparse 3D convolution
backbone is reimplemented with gather/scatter in PyTorch, and only the first
stage (CenterHead proposals, rotated BEV NMS at 0.1) is run; the RoI
refinement stage needs custom CUDA operators and is skipped. About 20 s per
scan on one CPU core.

Caveats: CODa splits frames randomly, and 116 of the 161 frames in this window
are in the CODa *training* split, so detector numbers are optimistic. The
first stage alone tracks slightly better than the full two-stage model on
the development windows (scripts/runpod/README.md).

Detector (pedestrians, score >= 0.4, one-to-one matching within 0.5 m in the
bird's-eye view, objects within 50 m, all 161 frames): precision 0.84, recall
0.79.

Tracking with LiDAR detections of score >= 0.4:

| mode | coverage | position RMSE (m) | id switches | class accuracy | camera match rate |
| --- | --- | --- | --- | --- | --- |
| lidar_only | 0.939 | 0.239 | 370 | 0.994 | n/a |
| camera_associated | 0.939 | 0.239 | 370 | 0.994 | 0.993 |
| camera_assisted | 0.939 | 0.239 | 372 | 0.993 | 0.993 |

With a fifth of the people missed in any scan, a position-only tracker in a
crowd confuses neighbours: 370 ID switches over 203 identities and 16 s, an
order of magnitude more than with oracle detections. The largest groups are a
confirmed track taking over a neighbour's detection while the neighbour's own
track is still alive (about two thirds), and re-acquisition after one or more
missed scans. Appearance (ReID embeddings from the camera) and a
detector-aware lifecycle are the next steps; see `ROADMAP.md`.

#### Appearance (ReID) on CODa

`scripts/reid/embed_detections.py` projects each LiDAR detection into the
camera, matches it to a YOLO box of the same frame (IoU >= 0.3) and stores a
DINOv2 ViT-S/14 embedding of the crop (140x70, frozen, no ReID training) in the
detection's `appearance` field; the tracker adds
`appearance_weight * (1 - cos)` to the association cost and keeps a moving
average per track (`factor-tracker-core`, off by default).

- The features separate people: cosine distance between the same person in
  consecutive frames has median 0.165 (90th percentile 0.28), between
  neighbours closer than 1.5 m median 0.33 (oracle detections, 1,303 / 442
  pairs).
- But only 13 % of the LiDAR detections get an embedding: the Ouster sees
  360 degrees, the one forward camera about 80, and YOLO finds roughly 12 of the
  ~90 people per frame. Even at 0-5 m only 20 % have one.
- ID switches (real detections, score >= 0.4, same settings as the table
  above): 370 without appearance, 373 / 372 / 364 / 387 with weights 2 / 5 / 10
  / 20. No measurable change; two thirds of the switches happen inside the
  camera's view, mostly between people it does not detect or crops it cannot
  separate.

So the mechanism is in place and measured, and the bottleneck is coverage,
not the cost: appearance needs cameras around the robot (GrandTour has five)
and person detections for most of the LiDAR objects before it can move this
number.

### Robot-centric evaluation (CODa)

For a robot, an identity mistake 25 m away matters far less than one in front
of it. `evaluate --zone-range-m R --zone-half-angle-deg A --setup setup.json`
scores only ground truth within `R` metres and `A` degrees of the robot's +x
axis (positions mapped back to the robot frame with the ego poses). The
front zone below (8 m, +-40 degrees) is roughly the forward camera's view and
the robot's path; real LiDAR detections (score >= 0.4), pedestrian motion
prior, 1 m gate, `lidar_only`, ID switches (coverage):

| | 5 m, +-40 deg | 8 m, +-40 deg | 8 m, all around | whole range |
| --- | --- | --- | --- | --- |
| oracle detections | 0 | 2 | 2 | 31 |
| real detections | 12 | 43 (0.993) | 65 (0.959) | 370 (0.939) |
| + appearance, weight 10 | 12 | 37 (0.994) | 59 (0.960) | 364 (0.938) |
| + birth suppression 1.0 m | 16 | 41 (0.987) | 62 (0.949) | 316 (0.930) |
| + both | 14 | **26** (0.984) | **41** (0.949) | 314 (0.929) |

The front zone has more switches per object than the whole range (63 vs 25
per 1,000 object-frames) although the detector is best there (pedestrians
0-8 m ahead: recall 0.91, precision 0.89, median position error 0.09 m).

What was tried and what it showed:

- **Simulated detector errors on the oracle boxes.** Position noise
  (sigma 0.18 m) and 20 % random misses together give 5 front-zone / 95 total
  switches, far from the real 43 / 370. 19 % extra detections placed 0.4-1 m
  from real people give 849 total; placed at random, 199. Most real false
  positives are of the first kind: 62 % lie 0.5-1.2 m from a person, and 45 %
  are a second detection of a person who is already detected.
- **Birth suppression** (`min_birth_distance_m`, off by default): an
  unassociated detection closer than the distance to a confirmed track does
  not start a new one. -15 % switches over the whole range on CODa; on KITTI
  cars (oracle) it costs 15 -> 16 switches and 0.981 -> 0.969 coverage, so it
  is a setting for pedestrian crowds, not a default.
- **Appearance + birth suppression** is the only combination that moved the
  front zone in this window (43 -> 26-30 for birth 1.0-1.2 m and weights
  5-10); birth 0.8 m or weight 20 lose the gain. See the multi-window check
  below before using it.
- **No effect or worse** (code not kept unless noted): applying appearance
  only to ambiguous pairs (no measurable difference, reverted); a LiDAR-only
  identity cue from the point-cloud height in the box (same person |dh|
  median 0.02 m vs neighbours 0.08 m, but noisy; worse at every weight
  tested); dropping LiDAR detections the camera does not confirm (removes
  1,737 true and 476 false pedestrians: YOLO misses people in the crowd);
  centre-distance NMS between pedestrian detections (radius 0.5 -> 0.8 m:
  precision 0.89 -> 0.91, recall 0.82 -> 0.75; no consistent gain in
  switches); merging confirmed
  tracks that stay within 0.3-0.6 m (removes real neighbours, reverted);
  merging pedestrian + scooter boxes of riders (38 merges, no effect).

### Multi-window check (CODa)

The window above was the one the settings were found on, so the same grid was
run on four more 16 s windows (sequence 20 at frames 4820, 9330 and 12700;
sequence 13, another day, at 9700) with `scripts/coda/run_window.sh` and
summed with `scripts/eval/summarize_windows.py`. ID switches summed over the
five windows (coverage), and in how many windows the setting was better /
equal / worse than the baseline:

| real detections | 5 m, +-40 deg | 8 m, +-40 deg | 8 m, all around | whole range |
| --- | --- | --- | --- | --- |
| baseline | 14 (0.996) | 63 (0.989) | 147 (0.977) | 1091 (0.944) |
| + appearance, weight 10 | 20 (0.997), 0/3/2 | 61 (0.990), 2/1/2 | 146 (0.978), 1/2/2 | 1088 (0.943), 4/0/1 |
| + birth suppression 1.0 m | 18 (0.985), 0/4/1 | 63 (0.977), 1/3/1 | 131 (0.968), 3/2/0 | 965 (0.930), 4/0/1 |
| + both | 15 (0.969), 1/3/1 | **45** (0.970), 2/2/1 | **109** (0.965), 3/2/0 | 960 (0.929), 4/0/1 |
| + both, birth 1.2 m | 15 (0.961), 1/3/1 | 49 (0.966), 2/2/1 | 116 (0.961), 4/1/0 | **932** (0.921), 4/0/1 |

With oracle detections the same settings do harm: birth suppression raises
the 8 m all-around switches from 16 to 22 and lowers coverage by about two
points, because it also blocks real people who walk up next to someone.

What this supports:

- With a noisy detector, appearance + birth suppression (1.0 m, weight 10)
  reduces switches within 8 m around the robot by 26 %, never worse in any
  window, and by 12 % over the whole range.
- The forward-zone gain (-29 %) comes mostly from one window (43 -> 26); in
  the other four the forward zone has 0-9 switches per window and moves by
  at most 2 either way.
- The cost is coverage, worst close in: within 5 m ahead it drops from 0.996
  to 0.969 (people near the robot are tracked later). For a robot that has to
  stop for people, that is the number to watch.
- Appearance alone does nothing measurable; it only helps once birth
  suppression stops the duplicate tracks it would otherwise have to resolve.

#### Persistent births

The coverage loss comes from real people who walk up next to someone: their
detections are suppressed for as long as they stay close. With
`birth_persistence` (`--birth-persistence N`) a suppressed detection is kept
as a candidate; if it reappears within 0.5 m of the previous sighting, at
most 300 ms later, on N batches in a row, a track is started and the earlier
sightings count toward confirmation. A second box on a person flickers and
moves with the person, so it rarely gets there; a neighbour does after N
frames. Same five windows, appearance weight 10:

| real detections | 5 m, +-40 deg | 8 m, +-40 deg | 8 m, all around | whole range |
| --- | --- | --- | --- | --- |
| baseline | 14 (0.996) | 63 (0.989) | 147 (0.977) | 1091 (0.944) |
| birth 1.0 m, suppressed for good | 15 (0.969) | 45 (0.970) | 109 (0.965) | 960 (0.929) |
| birth 1.0 m, persistence 3 | 11 (0.986) | 46 (0.983) | 118 (0.973) | 1025 (0.939) |
| **birth 1.5 m, persistence 3** | **11** (0.986), 1/4/0 | **42** (0.983), 2/2/1 | **116** (0.972), 3/2/0 | 983 (0.936), 4/0/1 |
| birth 1.5 m, persistence 3, no appearance | 16 (0.986) | 53 (0.984) | 128 (0.972) | 973 (0.936) |

| oracle detections | 5 m, +-40 deg | 8 m, all around | whole range |
| --- | --- | --- | --- |
| baseline | 0 (0.998) | 16 (0.994) | 381 (0.986) |
| birth 1.0 m, suppressed for good | 1 (0.973) | 22 (0.972) | 380 (0.962) |
| birth 1.5 m, persistence 3 | 0 (0.998) | 16 (0.993) | 359 (0.982) |

On KITTI cars (oracle, all 21 sequences, `lidar_only`): baseline 15 switches
/ 0.981 coverage, birth 1.0 m suppressed for good 16 / 0.969, birth 1.5 m
suppressed for good 27 / 0.953, birth 1.5 m with persistence 3 15 / 0.980.

So with persistence the larger birth distance is safe: it no longer costs
anything with good detections, and with real detections it keeps most of the
switch reduction (-33 % ahead, -21 % around the robot) while within 5 m ahead
coverage is 0.986 instead of 0.969. Results are flat for birth 1.5-2.0 m and
persistence 3-4. Appearance matters here: without it the forward-zone gain
halves. The forward gain still comes mostly from one window (43 -> 22).

Suggested settings for a robot among pedestrians with a noisy LiDAR detector
and one forward camera: `--min-birth-distance-m 1.5 --birth-persistence 3
--appearance-weight 10`. They stay off by default in the library.

#### Camera detector: YOLO11x vs YOLO26x

The numbers above use YOLO11x for the camera boxes that select the ReID
crops. Re-running the five windows with YOLO26x (Ultralytics 8.4, COCO) gives
2.6 % more person boxes but the same share of LiDAR pedestrians with an
embedding (19.9 % vs 20.3 %). By position of the LiDAR pedestrian:

| | out of camera view | in view, < 8 m | in view, 8-15 m | in view, > 15 m |
| --- | --- | --- | --- | --- |
| LiDAR pedestrians | 23,353 | 3,691 | 7,516 | 9,657 |
| with embedding, YOLO11x | 1 % | 72 % | 51 % | 22 % |
| with embedding, YOLO26x | 1 % | 72 % | 51 % | 23 % |

The camera detector is not the limit: half of the people are outside the one
forward camera, and in view close range the misses are people occluded by
others in the crowd. ID switches with YOLO26x embeddings (birth 1.5 m,
persistence 3, weight 10, real detections): 16 / 50 / 121 / 1002 for the
four zones, against 11 / 42 / 116 / 983 with YOLO11x; appearance alone gets
11 / 52 / 137 / 1088 instead of 20 / 61 / 146 / 1088. Swapping the detector
moves single configurations by up to about 10 switches in the forward zones,
which is the size of the differences between neighbouring settings: treat
those as noise and only the consistent effects (birth suppression with
persistence, appearance on top of it) as findings. The scripts now default
to YOLO26x.

#### Better appearance inputs (negative results)

Offline, with the CODa instance labels: for pairs of observations one to ten
frames apart, how well does a cue separate the same person from a different
person within 2 m (ROC AUC of the distance; 0.5 = no information)?

| cue | all pairs | position-ambiguous pairs (0.3-1.0 m apart) |
| --- | --- | --- |
| position | 0.767 | 0.722 |
| camera embedding (DINOv2) | 0.827 | 0.776 |
| camera, crop < 5 % occluded | 0.853 | |
| camera, crop 30 % or more occluded | 0.732 | |
| LiDAR shape (point fraction and spread in 8 height bins, top height) | 0.754 | 0.713 |

Occluded crops are clearly worse, and the shape descriptor (CODa's released
scans carry no intensity, so geometry only) is about as informative as
position but available for 73 % of pedestrian detections instead of 20 %.
Neither helped the tracker on the five windows (real detections, birth 1.5 m,
persistence 3):

- **Skipping crops occluded more than 30 %** by closer camera boxes
  (`embed_detections.py --max-occlusion 0.3`, off by default): 16 % fewer
  embeddings; switches 15 / 47 / 120 / 974 for the four zones against
  11 / 42 / 116 / 983, i.e. no change beyond noise. The track's moving
  average already damps single bad crops.
- **LiDAR shape descriptor as the appearance vector**: weight 3 gives
  14 / 47 / 129 / 988 (no gain), weight 10 makes everything worse
  (whole range 1091 -> 1171, with oracle detections 381 -> 436). The cue
  separates people on average but is too noisy frame to frame, and errors
  near people are exactly where it is applied.

### Fixed-lag re-association (FGO)

`--fgo-lag N` (replay, stream, and the ROS 2 node's `fgo_lag`) passes the
engine's LiDAR frames through `factor_tracker_engine::fgo::FixedLagRefiner`: over
the last N scans it re-decides which detection belongs to which person by the
trajectories' factor-graph cost (constant-velocity motion and position
factors; the optimum's cost is the Kalman prediction-error decomposition)
plus birth, miss and clutter costs, by local search from the engine's own
association (tail swaps, merges, splits, single detections to clutter).
Scans older than the window are frozen, so output is final; each frame comes
out N scans late. Positions are re-estimated causally.

Why: on CODa the base tracker's switches are almost never a lost track
restarting (fragments, ~0 %); about half are a track moving to another person
(in half of those both people are detected) and 30 % a person with two tracks
(`scripts/fgo/switch_types.py`).

Five CODa windows (real detections) and GrandTour ETH-2, `lidar_only`,
pedestrian prior, 1 m gate; ID switches (coverage):

| setting | CODa all range | CODa within 8 m | ETH-2 |
| --- | --- | --- | --- |
| base | 1091 (0.941) | 147 | 22 (0.990) |
| `--fgo-lag 10` (1 s) | **984** (0.939) | **112** | 22 (0.989) |
| `--fgo-lag 20` (2 s) | 982 (0.939) | 121 | 22 (0.990) |

- -10 % over the whole range and -24 % within 8 m of the robot at 1 s lag,
  coverage unchanged. Per window (all range / within 8 m): 293/22 ->
  276/25, 8/0 -> 10/0, 118/27 -> 116/22, 370/65 -> 308/43, 302/33 -> 274/22;
  most of the gain is in the three dense windows, and two counts get slightly
  worse. No effect on ETH-2 (its 22 switches are mostly labels that end
  in the crowd, see `benchmarks/grandtour_eth2`).
- The costs were chosen on these same sets (`scripts/fgo`); the differences
  between nearby settings are a few percent, but a held-out set is still
  needed.
- About 3-5 ms per scan in the densest CODa window (~300 tracks over 16 s),
  against ~2 ms without; real time at 10 Hz.
- The refiner trades latency for identity: the immediate output is the
  engine's. For stopping in front of people use the immediate output; for
  following or counting people, the lagged one.
- A Python prototype (`scripts/fgo/window.py`) that saw each engine track's
  later detections reported better lagged numbers (around 940); that was
  optimistic, the Rust refiner only sees scans up to the current one.

### Standard metrics against baseline trackers (CODa)

`scripts/eval/mot_metrics.py` scores HOTA, IDF1 and MOTA with a
distance-based similarity (s = 1 - d / 1 m; the outputs carry positions, not
boxes), and `scripts/eval/baselines.py` re-implements AB3DMOT- and
SimpleTrack-style trackers on the engine's own detections, so every tracker
sees the same inputs (CODa PV-RCNN, CPU, score >= 0.4). These inputs and the
truth include every CODa dynamic class the detector reports (pedestrians,
scooters, cyclists, ...); `scripts/eval/coda_tracker_eval.sh` now scores
pedestrians only, where the engine and AB3DMOT-style score 0.618 / 0.618 on
dev and 0.768 / 0.764 on held-out windows (`docs/detector_training.md`).
All settings were
chosen on the five development windows above; the held-out windows
(sequences 3, 4 and 9, 160 scans each, not used before) were scored once
with those settings.

| tracker (setting chosen on dev) | dev HOTA | dev IDF1 | held-out HOTA | held-out IDF1 | held-out AssA |
| --- | --- | --- | --- | --- | --- |
| engine, previous defaults | 0.559 | 0.690 | 0.666 | 0.798 | 0.713 |
| engine, updated tracks only, gate 5.99 | 0.596 | 0.714 | 0.688 | **0.813** | 0.740 |
| + FGO (`--fgo-lag 10`) | 0.607 | 0.744 | 0.695 | 0.806 | 0.757 |
| engine, distance association (1 m, no cascade) | **0.615** | **0.755** | **0.710** | **0.827** | **0.786** |
| + FGO (`--fgo-lag 10`) | 0.613 | 0.746 | 0.704 | 0.806 | 0.776 |
| AB3DMOT-style (0.8 m gate, 8 scans) | 0.614 | 0.741 | 0.705 | 0.806 | 0.771 |
| AB3DMOT-style + FGO | 0.611 | 0.740 | 0.701 | 0.798 | 0.766 |
| SimpleTrack-style | 0.578 | 0.649 | 0.666 | 0.743 | 0.718 |

What this supports:

- Reporting coasting predictions cost the engine most of its gap: they
  count as false positives. With updated-only output and a tighter gate it is
  close to a tuned AB3DMOT-style tracker, but not ahead of it on HOTA, on
  either split.
- The rest of the gap was the association cost. A squared Mahalanobis cost
  favours tracks with a large covariance (one that has coasted, or one just
  started with an uncertain velocity), so they take detections from
  well-observed neighbours; a larger Mahalanobis gate makes it worse (dev HOTA
  0.598 at 9.21, 0.586 at 20). With `--max-association-distance-m 1.0` the
  cost is the horizontal distance to the predicted position with a fixed 1 m
  gate. The matching cascade, which existed to work around the same bias,
  then only gets in the way (`--recent-track-window-ns 0`; with it, held-out
  HOTA is 0.679). Distance 0.8 m and 1.0 m, and the motion-noise settings,
  differ by at most 0.003 on dev; 1.0 m was chosen there, and the held-out
  windows were scored with it afterwards. The ROS 2 node uses this setting
  by default; the engine default stays Mahalanobis, since a fixed metric gate
  suits people but not fast vehicles.
- With the distance cost the engine is level with the AB3DMOT-style tracker
  on dev (0.615 / 0.614) and slightly ahead on the held-out windows (0.710 /
  0.705, IDF1 0.827 / 0.806). FGO no longer helps: its cost is the Kalman
  likelihood, the model the distance cost moved away from. Re-tuning it on dev
  (birth 10-40, sigma 0.15-0.25, clutter, miss, radius, lag 10-30) and
  ablating its moves keeps dev HOTA within 0.612-0.615: swaps cut ID
  switches (1037 -> 854) but lower IDF1 (0.753 -> 0.746), merges alone give
  IDF1 0.757 and HOTA 0.615, split and drop change nothing. With positions
  only, re-deciding over a window adds nothing measurable to the distance
  association.
- Assigning every detection the identity of the truth object it matches
  (perfect association on the same detections) scores dev HOTA 0.659, or
  0.697 when unmatched detections are also dropped. Association has about
  0.045 HOTA left on these detections; most remaining ID switches are
  neighbours exchanging tracks in crowds.
- FGO cuts ID switches by about 10 % and raises the engine's AssA on the
  held-out windows (0.740 -> 0.757), but does not move HOTA or IDF1 past the
  baseline, and slightly lowers AB3DMOT-style's.
- Every reasonable tracker lands within about 0.01 HOTA of the others on
  these detections; DetA (about 0.63 dev, 0.65 held-out) limits HOTA as much
  as association does. Beating the baselines clearly needs better detections,
  not only a better tracker.
- The held-out windows are less crowded (32-58 ID switches in total against
  ~850-1000 on dev), so they separate trackers less.

### FGO and box shape on pedestrian-only scoring (CODa)

Re-measured with `scripts/eval/coda_tracker_eval.sh` (pedestrians only, same
detections; engine with distance association):

| setting | dev HOTA / IDF1 / ID switches | held-out HOTA / IDF1 / ID switches |
| --- | --- | --- |
| engine | 0.618 / 0.760 / 934 | 0.768 / 0.900 / 18 |
| + FGO, `--fgo-lag 10` | 0.615 / 0.751 / 806 | 0.761 / 0.881 / 20 |
| + FGO, `--fgo-lag 20` | 0.615 / 0.750 / 814 | 0.761 / 0.881 / 20 |
| + box size as an appearance cue, weight 0.5 (best on dev) | 0.619 / 0.760 / 928 | 0.768 / 0.900 / 20 |

- FGO cuts dev switches by 14 % but lowers IDF1: its swaps undo about as many
  right identities as wrong ones. Replacing the Kalman filter by a factor graph
  would not change the filtered positions: for this linear-Gaussian model the
  latest state of the factor graph equals the Kalman estimate (FGO already
  computes its trajectory cost that way).
- Box size and heading barely separate neighbours where position does not.
  Over the eight windows, with a person's last five boxes as template and a
  neighbour within 1.5 m (`scripts/eval/shape_cue.py`): position picks the right
  detection 98.8 % of the time; in the 363 cases where it does not, size is
  right 57 % and heading 51 % (chance). In close encounters the detector's
  boxes merge or split, so their size is unreliable exactly when it would be
  needed. As an association cue (size standardized, cosine appearance cost)
  it changes nothing beyond noise; larger weights make it worse (dev HOTA
  0.617 at weight 2, 0.595 at 4).

### Upper bounds and detector NMS for crowds (CODa, pedestrians)

Upper bounds on the same eight windows (`scripts/eval/upper_bounds.py`):

| | dev HOTA | held-out HOTA |
| --- | --- | --- |
| engine (distance association, NMS 0.1, score >= 0.4) | 0.618 | 0.768 |
| perfect association on the same detections | 0.697 | 0.790 |
| same, unmatched detections dropped | 0.731 | 0.815 |
| perfect detections (truth boxes), engine | 0.848 | 0.955 |
| perfect detections (truth boxes), AB3DMOT-style | 0.853 | 0.953 |

The detector's bird's-eye-view NMS suppresses any box overlapping a
higher-scoring one by more than IoU 0.1, which can remove a person standing
next to another. The detector was run once without suppression
(`detect_lidar_cpu.py --nms-threshold 1.0 --score-threshold 0.1`) and NMS
re-applied offline (`scripts/eval/nms_variants.py`; IoU 0.1 reproduces the
detector's own output exactly). Dev recall / precision at score >= 0.4: IoU
0.1 0.801 / 0.804, 0.2 0.825 / 0.793, 0.3 0.837 / 0.773, 0.5 0.857 / 0.720;
circle NMS (centre distance) is no better. Looser NMS finds more people but
also duplicate boxes, and on its own lowers HOTA (IoU 0.3, score >= 0.5: dev
0.609). Suppressing births within `--min-birth-distance-m` of a confirmed
track absorbs the duplicates. Grid on dev (NMS IoU 0.2 / 0.3 / 0.5, score
0.4 / 0.5, birth distance 0.8 / 1.0 / 1.2 m): best is IoU 0.3, score >= 0.5,
0.8 m.

| setting (chosen on dev) | dev HOTA / IDF1 / ID switches | held-out HOTA / IDF1 / ID switches |
| --- | --- | --- |
| engine, NMS 0.1, score >= 0.4 | 0.618 / 0.760 / 934 | 0.768 / 0.900 / 18 |
| engine + birth suppression 0.8 m, NMS 0.3, score >= 0.5 | 0.628 / 0.772 / 773 | 0.777 / 0.902 / 18 |
| AB3DMOT-style, NMS 0.1, score >= 0.4 (its best on dev) | 0.618 / 0.746 / 987 | 0.764 / 0.882 / 28 |
| AB3DMOT-style, NMS 0.3, score >= 0.5 | 0.600 / 0.709 / 1332 | 0.752 / 0.859 / 57 |

`scripts/eval/coda_tracker_eval.sh` prints the birth-suppressed engine as
`engine_birth`.

### Where the remaining error is (CODa, pedestrians)

Same input as the best row above (NMS IoU 0.3, score >= 0.5), dev windows:

| | HOTA | DetA | AssA | IDF1 |
| --- | --- | --- | --- | --- |
| engine + birth suppression 0.8 m | 0.628 | 0.650 | 0.613 | 0.772 |
| perfect association on the same detections (`scripts/eval/upper_bounds.py`) | 0.703 | 0.651 | 0.761 | 0.893 |
| engine, tracks cut where their person changes (`scripts/eval/oracle_repair.py --split`) | 0.640 | 0.654 | 0.628 | 0.767 |
| engine, fragments of one person joined (`--stitch`) | 0.632 | 0.650 | 0.621 | 0.786 |
| both: perfect swap correction on the engine's own positions | 0.670 | 0.654 | 0.688 | 0.837 |

Detection is already where perfect association would leave it (DetA 0.650
against 0.651); the gap is association, mostly identity swaps in crowds.
About 45 % of the dev switches hand a person to a track that was following
someone else, 23 % to a newly started track, 17 % to a duplicate track that
followed nobody, and 14 % go back and forth between two tracks.

**More of the detector's candidates (negative).** Without NMS the detector
finds more people: within 1 m, 0.901 of the dev people have a detection in
the tracker's input, 0.947 have a candidate at score >= 0.3 and 0.965 at
>= 0.1, and a perfect selection with perfect association would reach dev
HOTA 0.784 instead of 0.735 (`upper_bounds.py --drop-unmatched` on the
un-suppressed candidates at score >= 0.3, and on the tracker's input). But of the 0.3-0.5 score candidates left after
NMS, 6576 are second boxes on a detected person, 4954 clutter and 1859
people otherwise missed, and neither the score nor staying in place
separates the last two. Everything tried stays at or below the baseline
(dev HOTA; baseline 0.626 in these runs, with NMS over pedestrians only;
code not kept):

- low-score candidates only update existing tracks
  (`--new-track-score-threshold 0.5`): 0.603 at score >= 0.3, 0.486 at >= 0.1;
- no NMS, score >= 0.5: 0.548; circle NMS 0.5 m with low-score updates: 0.612;
- low-score updates only within 0.3-0.5 m of a prediction, or a `1 - score`
  term in the cost: 0.612-0.627; low-score updates used for association but
  not reported (decoys against stealing): 0.616-0.627;
- filling a track's gaps of up to 10 scans (interpolation or nearby
  candidates), joining fragments: 0.617-0.628. Each miss removed adds about
  one false positive: many gaps are scans where the person has no label, or
  where the track is on someone else.

**Swaps are a model error, not a search error.** At each transfer (a track
moves from one person to another), `scripts/eval/swap_cost.py` scores the
detections of the two tracks within 10 scans twice with the fixed-lag
refiner's cost (constant-velocity Kalman NLL plus a miss cost): as the engine
associated them, and by person. On dev the right association is the cheaper
one at only 24 % of 679 transfers (16-30 % for position noise 0.1-0.4 m,
process noise 0.1-8 m^2/s^3, no miss cost). No search over associations can
repair the rest, which is why `--fgo-lag` cuts switches but does not raise
HOTA. A velocity measurement per detection would change that: with the truth
velocity plus 0.3 m/s noise as an extra factor the right association is
cheaper at 59 % of the transfers (0.5 m/s: 50 %). A learning-free one,
aligning each detection's upper-body points with the previous scans
(horizontal translation search), has a median error of 0.26 m/s (the Kalman
velocity: 0.19 m/s) with a heavy tail; as a robust factor (outliers capped)
it raises the share to 29 %, where the truth velocity under the same factor
gives 40 % (not kept).

### Labels and poses of the CODa windows

`scripts/eval/label_consistency.py --coda-root CODa` on the eight windows
(teleport: a labelled id moving more than max(1 m, 3 m/s * dt) between its
labelled scans; label tracking: a constant-velocity tracker run on the
labelled positions themselves, scored against the label ids; static
objects: labelled poles, trees, bins, ... in the world frame):

| window | ids | teleports | label tracking HOTA / ID switches | static objects: per-instance std, scan-to-scan 95 % |
| --- | --- | --- | --- | --- |
| dev 13:9700 | 171 | 58 | 0.790 / 159 | 0.12 m, 0.30 m |
| dev 20:12700 | 44 | 3 | 0.996 / 2 | 0.11 m, 0.12 m |
| dev 20:4820 | 113 | 13 | 0.950 / 37 | 0.09 m, 0.15 m |
| dev 20:8640 | 200 | 22 | 0.964 / 36 | 0.09 m, 0.13 m |
| dev 20:9330 | 126 | 51 | 0.882 / 157 | 0.26 m, 0.27 m |
| held-out 3:1800 | 22 | 0 | 1.000 / 0 | 0.06 m, 0.11 m |
| held-out 4:6870 | 27 | 0 | 0.991 / 3 | 0.07 m, 0.10 m |
| held-out 9:8515 | 25 | 0 | 1.000 / 0 | 0.08 m, 0.10 m |

On dev 13:9700 and 20:9330 even the labelled positions cannot be tracked
consistently with their own ids, so association scores there are bounded by
the labels, and on 20:9330 the given poses jitter (static objects move
0.26 m). The held-out windows are clean.

### Pose correction from static landmarks (CODa)

`--pose-correction` aligns each scan's static detections (the detector's
poles, trees, bins, benches, ... at score >= 0.5 within 40 m, written by
`convert_coda.py` as `landmarks`) with a map of the ones seen before and
tracks the scan in the corrected frame (`docs/camera-lidar.md`). Same
detections as above; `coda_tracker_eval.sh` prints it as `engine_pose`:

| setting | dev HOTA / IDF1 / ID switches | held-out HOTA / IDF1 / ID switches |
| --- | --- | --- |
| engine + birth suppression 0.8 m | 0.628 / 0.772 / 773 | 0.777 / 0.902 / 18 |
| + pose correction | 0.636 / 0.793 / 584 | 0.771 / 0.906 / 18 |

It helps where the poses jitter: dev 13:9700 HOTA 0.543 -> 0.556 (switches
258 -> 172) and 20:9330 0.585 -> 0.606 (194 -> 121). Where the poses are
already good it adds its own noise (about 2 cm per scan): other windows
within +-0.006 HOTA, except held-out 4:6870 (-0.010) and 9:8515 (-0.005).
Tuned on dev with a prototype (baseline 0.626): match radius 0.5-1.0 m,
2-5 sightings to confirm a landmark, 2-20 s memory, a 10-200 sighting mean
and landmark scores 0.3-0.7 all give HOTA within +-0.002; starting each scan
from the previous correction matters (from the identity: 0.623). Aligning
to a map of the whole window instead (offline) gives 0.630.

With pose noise added to every scan (`scripts/eval/coda_pose_noise_eval.sh`:
yaw 1 degree and 0.1 m standard deviation, as a jittery odometry would give;
truth moved with the same poses):

| setting (pose noise) | dev HOTA / IDF1 / ID switches | held-out HOTA / IDF1 / ID switches |
| --- | --- | --- |
| engine + birth suppression 0.8 m | 0.538 / 0.692 / 2467 | 0.688 / 0.863 / 120 |
| + pose correction | 0.631 / 0.789 / 644 | 0.767 / 0.905 / 19 |
| AB3DMOT-style, same detections | 0.514 / 0.620 / 3682 | 0.649 / 0.781 / 304 |

The correction recovers the noise-free scores.

```sh
scripts/eval/nms_variants.py det/raw det/nms iou0.3                # detector run without NMS, see above
scripts/eval/coda_tracker_eval.sh det/nms/iou0.3 CODa 0.5 work   # engine_pose row
scripts/eval/coda_pose_noise_eval.sh work 1.0 0.1                 # pose-noise table
scripts/eval/label_consistency.py work/coda*/* --coda-root CODa   # labels and poses
scripts/eval/swap_cost.py work/coda20_9330/20 work/coda20_9330/20/engine_birth.jsonl
```

Estimating ego motion and
moving objects together is SLAMMOT (Wang et al., IJRR 2007); here the poses
are given and only corrected, from the detections the tracker already gets:
no point cloud and no map beyond the last few seconds.

### Fixed-lag smoothing (CODa)

`--smooth-lag N` re-estimates each scan's tracks with the next N scans
(Rauch-Tung-Striebel pass over the engine's filtered states;
`docs/camera-lidar.md`). It changes no association, only positions,
velocities and covariances, at N scans of latency. Velocity is scored by
`scripts/eval/velocity_error.py` (median error against the slope of a line
fitted to the labelled positions within +-5 scans). Same detections as above
(NMS IoU 0.3, score >= 0.5); the lag was chosen on dev and held-out scored once:

| setting | dev HOTA / IDF1 / ID switches | dev velocity median / 90 % (m/s) | held-out HOTA / IDF1 / ID switches | held-out velocity median / 90 % (m/s) |
| --- | --- | --- | --- | --- |
| engine + birth suppression (filter) | 0.628 / 0.772 / 773 | 0.218 / 0.728 | 0.777 / 0.902 / 18 | 0.129 / 0.332 |
| + smoothing, lag 2 (0.2 s) | 0.636 / 0.773 / 780 | 0.188 / 0.610 | | |
| + smoothing, lag 5 (0.5 s) | 0.638 / 0.773 / 767 | 0.155 / 0.527 | 0.782 / 0.905 / 18 | 0.099 / 0.231 |
| + smoothing, lag 10 (1.0 s) | 0.638 / 0.775 / 757 | 0.148 / 0.504 | | |
| + pose correction | 0.636 / 0.793 / 584 | 0.197 / 0.604 | 0.771 / 0.906 / 18 | 0.126 / 0.305 |
| + pose correction + smoothing, lag 5 | 0.640 / 0.793 / 562 | 0.153 / 0.463 | 0.774 / 0.908 / 18 | 0.102 / 0.232 |
| AB3DMOT-style, NMS 0.1, score >= 0.4 (its best) | 0.618 / 0.746 / 987 | 0.227 / 0.738 | 0.764 / 0.882 / 28 | 0.133 / 0.344 |

At 0.5 s the median velocity error drops by 29 % (dev) and 23 % (held-out),
the 90th percentile by 28 % and 30 %; HOTA rises through localization (LocA
0.866 -> 0.871 dev). Longer lags add little. `coda_tracker_eval.sh` prints
the smoothed rows as `engine_smooth` and `engine_pose_smooth`.

### KITTI 3D MOT, PointRCNN pedestrians (AB3DMOT's protocol)

The standard 3D MOT comparison: the KITTI tracking validation split (11
sequences), the PointRCNN detections AB3DMOT ships, and AB3DMOT's evaluator
(`scripts/KITTI/evaluate.py`, 3D IoU 0.25; sAMOTA, MOTA, ID switches,
fragmentations). Both trackers write AB3DMOT's pedestrian output: a track from
its first detection and for up to three missed scans, with the detection's
box size, heading and image box. Running AB3DMOT's own code reproduces its
published pedestrian row (sAMOTA 82.73, MOTA 73.86, 4 ID switches): 83.24,
73.86, 4.

AB3DMOT's settings were tuned on the whole validation split. To compare
fairly, the split was halved: every choice below was made on the tune half
(0001 0006 0010 0013 0016 0018), and the test half (0008 0012 0014 0015
0019) was scored once with the final settings. The earlier rows were scored
on the test half afterwards, only to see whether each step carries over:

| setting | tune sAMOTA / MOTA / IDS | test sAMOTA / MOTA / IDS | whole split sAMOTA / MOTA / IDS |
| --- | --- | --- | --- |
| CODa pedestrian settings, confirmed tracks updated in the scan | 0.718 / 0.630 / 0 | 0.798 / 0.727 / 26 | 0.767 / 0.685 / 29 |
| AB3DMOT's output, no birth suppression, detection std 0.1 m | 0.740 / 0.591 / 0 | 0.851 / 0.745 / 20 | 0.816 / 0.691 / 20 |
| + report a track only while its confidence is >= 0.73 | **0.774 / 0.680 / 1** | 0.861 / 0.760 / 20 | 0.818 / 0.727 / 21 |
| AB3DMOT (tuned on the whole split) | 0.761 / 0.668 / 0 | **0.865 / 0.771 / 5** | **0.832 / 0.739 / 4** |

On the test half factor-tracker trails AB3DMOT by 0.4 sAMOTA and 1.2 MOTA
points, with 20 ID switches to 5 (fragmentations 77 to 57): it associates by
centre distance, AB3DMOT by 3D GIoU of the boxes. Birth suppression, which
absorbs duplicate boxes on CODa, suppresses real neighbours here (348 of the
boxes AB3DMOT writes and the suppressed run did not).

#### Track confidence

The evaluator scores a track by the average score of its boxes and drops
whole tracks below a threshold (the one that maximizes MOTA; sAMOTA sweeps
it). KITTI labels pedestrians only from some distance on, and PointRCNN sees
them earlier with weak scores: in 0013 a person walking towards the car from
47 m is one factor-tracker track over 140 scans, and five AB3DMOT tracks.
Averaged over the unlabelled, weak part, the long track falls below the
threshold and is lost entirely (438 missed boxes on the tune half, 260 for
AB3DMOT's fragments).

Each track therefore carries a confidence (`TrackOutput.confidence`): the
moving average, with weight 0.3 for the newest detection, of the log-odds of
the detection scores that updated it, as a probability.
`--report-min-confidence` reports a track only while its confidence is high
enough; it keeps its id meanwhile. The KITTI result files carry the
confidence as the box score, as log-odds (PointRCNN's own scale, which
AB3DMOT writes). Other choices, on the tune half:

| setting | sAMOTA | MOTA |
| --- | --- | --- |
| detection score, as a probability | 0.740 | 0.591 |
| detection score, as log-odds | 0.755 | 0.630 |
| track score: sum of detection log-odds, log(1 - 0.9) per missed scan (MHT) | 0.710 | 0.642 |
| confidence (weight 0.3), all tracks reported | 0.754 | 0.628 |
| confidence (weight 0.3), reported from log-odds 0.5 / 1 / 1.5 | 0.771 / 0.774 / 0.759 | 0.663 / 0.680 / 0.671 |
| confidence (weight 0.2 / 0.5), reported from log-odds 1 | 0.773 / 0.773 | 0.668 / 0.671 |

The cumulative (MHT) score ranks long false tracks (static clutter) first.
Reporting from 0.73 (log-odds 1) is the setting in the table above.

To reproduce (a checkout of https://github.com/xinshuoweng/AB3DMOT for its
detections and evaluator; its code is not part of this repository):

```sh
scripts/kitti/ab3dmot_benchmark.sh <kitti-root> <ab3dmot-root> work factor_tracker
cd <ab3dmot-root> && python3 scripts/KITTI/evaluate.py factor_tracker 1 3D 0.25
```

The evaluator scores the sequences in `scripts/KITTI/evaluate_tracking.seqmap.val`;
for a half, run it from a directory holding `scripts/KITTI/label` (a link),
a seqmap with those lines, and `results/KITTI` (a link).

## Reproducing

```sh
cargo run -p factor-tracker -- cam-demo --out-dir cam-demo-out
# comparison table: cam-demo-out/comparison.json
# report:           cam-demo-out/report.html
```

To inject faults:

```sh
cargo run -p factor-tracker -- cam-demo --out-dir cam-demo-out \
    --camera-delay-ns 150000000 --lidar-delay-ns 0 \
    --reverse-window 3 --duplicate-every 7 \
    --pause-sensor camera --pause-start-ns 1000000000 --pause-end-ns 2000000000
```

The corresponding counters appear in `comparison.json` and the report's
diagnostics panel.
