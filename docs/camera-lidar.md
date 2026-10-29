# Camera-assisted LiDAR tracking

This document describes the `factor-tracker` SDK layer: the pipeline from
sensor detections and calibration to unified 3D tracks, diagnostics and
comparison/evaluation. The scope is deliberately one camera + one LiDAR. The
name reflects what is implemented: the LiDAR tracks the objects, and the camera
associates and augments them.

## Pipeline

```
sensor detections (LiDAR in the LiDAR frame, camera in image pixels)
    -> bounded arrival-order reorder buffer (event time preserved)
    -> per-observation coordinate normalization (calibration + ego pose at event time)
    -> ego-pose correction from static landmarks (optional, --pose-correction)
    -> LiDAR position tracking (factor-tracker-core Cartesian3DTracker)
    -> fixed-lag smoothing of the LiDAR tracks (optional, --smooth-lag)
    -> geometric camera association (projected 3D track boxes vs 2D detections)
    -> camera class evidence as a *soft* prior on LiDAR association (optional)
    -> unified track output + diagnostics
    -> comparison / evaluation / HTML report
```

`factor-tracker-core` still receives only observations that are already in one
common tracking frame. Sensor frames, calibration, reordering and the camera
live in the engine layer.

## Calibration and frames

Frames:

- **sensor frame** — LiDAR frame (metres) or camera optical frame.
- **reference frame** — vehicle body frame at the reference time.
- **tracking frame** — the frame the core tracker uses. Without ego poses it is
  the reference frame; with ego poses it is the world frame.

`Calibration` must include the camera intrinsics, the image size, the
sensor-to-reference transforms, the tracking frame name and the near plane.
The camera model is a rectified pinhole; distortion is not modelled. KITTI
rectified cameras satisfy this. Unsupported distortion is not silently ignored:
the adapter documents the assumption and rejects geometry it cannot handle.

Ego poses (`PoseTrack`) are interpolated (linear translation, quaternion slerp)
and **never extrapolated**. A requested time outside the recorded range, or
inside a gap larger than `pose_max_gap_ns`, yields `PoseUnavailable` and the
affected batch is skipped with a diagnostic. LiDAR uses the pose at the LiDAR
event time; camera projection uses the pose at the camera event time.

## Ego-pose correction from static landmarks

Given poses can jitter: on two of the five CODa development windows,
labelled poles and trees move by 0.2-0.3 m between scans in the world frame
(`docs/evaluation.md`), and every track inherits those jumps. With
`EngineConfig::pose_correction` (`--pose-correction`), each LiDAR batch's
`landmarks` (static objects in the LiDAR frame: poles, trees, bins, whatever
the detector reports as static; never tracked) correct its pose:

1. the landmarks go to the tracking frame with the pose and calibration, then
   with the previous scan's correction `C`;
2. each one is matched to the nearest confirmed map landmark (seen in 3 scans)
   within 0.7 m;
3. with at least 3 matches, `C` is updated by the horizontal rigid transform
   (rotation about z plus translation) that minimises the Huber-weighted
   distances to the matched map landmarks, solved in closed form (weighted
   Procrustes), twice with re-matching;
4. the scan's detections are tracked in the corrected frame `C ∘ pose`, and
   the map is updated with the corrected landmarks (running mean over at
   most 50 sightings; a landmark unseen for 5 s is dropped).

It is the optimum of a one-scan factor graph (one pose-correction variable,
one robust position factor per matched landmark) on top of the given poses;
without landmarks the correction stays where it is. Each `lidar_frame` then
carries `pose_correction`, the row-major 4x4 transform from the given
tracking frame to the corrected one; `factor-tracker evaluate` and
`scripts/eval/mot_metrics.py` map the truth of that frame through it, which
is the same as scoring in the sensor frame. Camera projection uses the latest LiDAR scan's correction. Input:

```json
{"type":"lidar_batch", "...": "...", "detections":[...], "landmarks":[{"position":[5.2,-3.1,0.4]}]}
```

`scripts/coda/convert_coda.py` writes the detector's static classes (score
>= 0.5, within 40 m) as landmarks. Code: `crates/factor-tracker-engine/src/landmarks.rs`.

## Fixed-lag smoothing

With `--smooth-lag N` each LiDAR frame is emitted N scans late with every
track's position, velocity and covariance re-estimated from the detections of
the next N scans too (`FixedLagSmoother`,
`crates/factor-tracker-engine/src/smoother.rs`): a Rauch-Tung-Striebel
backward pass over the engine's own filtered states, with the tracker's
motion model, so it is the marginal of each track's factor graph over the
window. Nothing is re-associated: ids, lifecycles and observations stay as
they are, and smoothed frames carry `"smoothed": true`. `stream
--smooth-realtime` writes every frame twice, at once and smoothed, so a live
consumer can use either (the ROS 2 node publishes them as `tracks` and
`tracks/smoothed`); `mot_metrics.py` and `velocity_error.py` score the
smoothed ones of such a file unless `--realtime`. It is not combined with
`--fgo-lag`, whose refiner re-estimates the states itself.

## Track confidence

Every track in a `lidar_frame` carries `confidence` in `[0, 1]`: the
exponential moving average of the log-odds of the detection scores that
updated it (weight `EngineConfig::confidence_alpha`, `--confidence-alpha`,
default 0.3 for the newest), as a probability. It follows the detections, so
a person approaching from far away is one track whose confidence rises as
the detector grows sure. `--report-min-confidence` reports a track only while
its confidence is at least the given value (0.73 on KITTI, `docs/evaluation.md`);
the track keeps its id while it is not reported. With the ROS 2 node, pass it
in `engine_args`. Tracks re-estimated by `--fgo-lag` carry none.

## Camera projection and association

For each camera batch at time `t_cam`:

1. The tracker is predicted to `t_cam` (no measurement update).
2. Tracks with valid, non-stale shape metadata and a recent spatial update are
   projected: the 3D box corners are transformed into the camera frame, then
   projected through `K`.
3. The projected axis-aligned box is clipped to the image; behind-camera,
   near-plane-straddling and out-of-view cases are rejected with a reason.
4. Projected boxes and 2D detections are matched one-to-one by IoU with a hard
   IoU gate (Hungarian assignment). Association uses geometry only; camera
   class never hard-gates a spatial correspondence.

Each association record contains the track id, detection id, camera time,
projected box, cost and accept/reject reason, plus per-track projection
rejection reasons.

### What the camera may and may not do

The camera:

- associates 2D detections with projected 3D tracks,
- accumulates class evidence,
- supplies a **soft** class prior to the LiDAR association.

The camera never:

- creates a 3D position from a 2D box,
- updates the 3D position covariance,
- deletes an out-of-view LiDAR track,
- keeps an old 3D position valid indefinitely.

Three timestamps are tracked separately: `state_timestamp`,
`last_spatial_update_ns` (the last LiDAR update) and `last_camera_support_ns`.
The age since the last spatial update is always reported, and projection is
gated on it, so camera support cannot extend the validity of an old position.

## Class evidence and the soft prior

Camera class evidence is per-track, exponentially decayed with a configurable
half-life, and ignored once older than `max_age_ns` or below `min_hits` /
`min_confidence`. The emitted value is a normalized evidence ratio, **not** a
calibrated probability.

In `camera_assisted` mode the engine installs the current evidence as a track's
soft class prior. The core tracker adds
`class_prior_penalty * confidence` to the association cost when a LiDAR
observation's known class contradicts the prior. The spatial Mahalanobis gate
always applies first, so a class prior can break a spatial tie but can never
associate a far-away detection. `camera_associated` mode records evidence but
does not install a prior; `lidar_only` ignores the camera entirely.

Sensor class systems are mapped by `ClassMap`. An empty map is the identity; a
non-empty map is authoritative and unmapped camera classes become Unknown
(`None`) and are never used as evidence.

## Bounded asynchronous input

`AsyncReorderBuffer` keeps event time and arrival time separate. State
estimation always uses event time; arrival time only drives buffering.

- **Watermark**: `max_event_time - window_ns` releases in-order batches.
- **Hold timeout**: a batch is also released once
  `arrival_now - arrival >= max_hold_ns`, so a stalled stream does not deadlock
  the buffer.
- Same-time order is deterministic: LiDAR before camera, then sensor id, then
  sequence.
- Configuration: `window_ns`, `max_hold_ns`, `capacity`.
- Duplicates (same kind/sensor/sequence), late batches (older than the last
  released event time) and overflow (buffer full) are rejected with counters and
  diagnostic events; overflow drops the newly arrived batch.
- `flush` releases everything in event-time order; the tracker's internal time
  is never advanced for display reasons.

Counters: `accepted`, `reordered`, `late`, `duplicate`, `overflow`, `flushed`
(reorder) and `transform_unavailable`, `association_rejected`,
`camera_matches`, `camera_unmatched`, `rejected_invalid` (engine).

## Fault injection

`inject::apply` deterministically injects, into a fixed input recording:

- LiDAR delay, camera delay (arrival time only),
- arrival-order reversal within a window,
- duplicate batches,
- a sensor pause over an event-time window.

These are software fault-injection experiments and do not reproduce every real
sensor failure mode.

## CLI

All commands are listed in `usage.md`. The camera+LiDAR entry point is:

```sh
cargo run -p factor-tracker -- cam-demo --out-dir cam-demo-out
```

which writes the synthetic fixture, replays it in the three modes, evaluates
them and renders `cam-demo-out/report.html`.

Individual steps:

```sh
cargo run -p factor-tracker -- cam-fixture --out-dir cam-demo-out
cargo run -p factor-tracker -- replay --setup cam-demo-out/setup.json \
    --inputs cam-demo-out/inputs.jsonl --mode camera_assisted \
    --output cam-demo-out/outputs.jsonl --diagnostics cam-demo-out/diagnostics.json
cargo run -p factor-tracker -- evaluate --output cam-demo-out/outputs.jsonl \
    --truth cam-demo-out/truth.jsonl --metrics cam-demo-out/metrics.json
cargo run -p factor-tracker -- report --setup cam-demo-out/setup.json \
    --output cam-demo-out/outputs.jsonl --truth cam-demo-out/truth.jsonl \
    --metrics cam-demo-out/comparison.json --diagnostics cam-demo-out/diagnostics.json \
    --html cam-demo-out/report.html
```

Real KITTI data uses the same `replay`/`evaluate`/`report` path after
`prepare`; see `docs/datasets.md`.

## Report contents

The HTML report shows the camera panel (2D detections and projected tracks with
association decisions), the top-down XY LiDAR view (observations, tracks,
lifecycle, truth), the mode comparison table, per-frame unmatched detections and
projection rejections, and the engine diagnostics. Selecting a frame answers
"why was this detection associated" and "why was this observation rejected".
The top-down view is a 2D projection, not a 3D viewer. Real camera images are
not embedded; the camera panel is drawn on a blank canvas at the calibrated
image size.
