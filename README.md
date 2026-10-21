<p align="center">
  <img src="assets/hero.gif" alt="factor-tracker on a Clearpath Husky robot driving through a campus crowd (CODa sequence 20): camera frames with YOLO detections and projected 3D track boxes (left), Ouster OS1-128 scan with pedestrian tracks from a LiDAR detector, trails and uncertainty rings in a bird's-eye view around the robot (right)" width="100%">
</p>

<h1 align="center">Factor Graph Object Tracker</h1>

<p align="center"><code>factor-tracker</code></p>

<p align="center">
  <a href="https://github.com/rsasaki0109/factor-tracker/actions/workflows/ci.yml"><img src="https://github.com/rsasaki0109/factor-tracker/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="License: MIT OR Apache-2.0"></a>
  <img src="https://img.shields.io/badge/MSRV-1.89-orange.svg" alt="MSRV 1.89">
  <img src="https://img.shields.io/badge/dependencies-nalgebra%20%7C%20thiserror-informational.svg" alt="Dependencies: nalgebra, thiserror">
</p>

<p align="center"><em>A Clearpath Husky robot driving through a crowd on the UT Austin campus (<a href="https://github.com/ut-amrl/coda-devkit">CODa</a> sequence 20): real camera frames and Ouster OS1-128 scans. LiDAR detections come from the published CODa PV-RCNN checkpoint (first stage, run on CPU by <code>scripts/coda/detect_lidar_cpu.py</code>), camera detections from YOLO11x; boxes, IDs, trails and uncertainty rings are the engine's <code>camera_assisted</code> tracks in the world frame (CODa SLAM poses). Most frames shown are in CODa's training split, so detections are optimistic; tracking numbers are in <a href="docs/evaluation.md"><code>docs/evaluation.md</code></a>. The animation contains CODa data and is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> (Zhang et al.), unlike the code. Rendered with <code>scripts/hero/render_hero.py</code>; the synthetic-demo image is <a href="assets/hero.svg"><code>assets/hero.svg</code></a>.</em></p>

---

Detector-agnostic multi-object tracking and multi-sensor fusion, written in
Rust. Kalman filtering and assignment per scan, with an optional fixed-lag
factor-graph re-association that re-decides recent associations
(`--fgo-lag`, `docs/evaluation.md`). Short name: `factor-tracker` (crates,
command line tool, ROS 2 package).

`factor-tracker` consumes discrete observations produced by external
detectors or sensor front-ends and performs prediction, data association, state
estimation and track lifecycle management. It outputs objects with consistent
IDs and explicit uncertainty. Sensor fusion is done by updating a **shared
track state** with measurements from several sensors; it is not an average of
independent tracker outputs.

The core is CPU-only and has no GPU, ROS, Python, OpenCV or inference-engine
dependency. Numeric work uses `f64`.

> All data shipped in this repository is **synthetic**, except the README
> animation (CODa, see its caption). Real-data results (KITTI tracking with
> oracle detections; CODa with oracle and with real LiDAR/camera detections)
> are reported, with their caveats, in `docs/evaluation.md`; no other accuracy,
> speed or benchmark number is claimed beyond `docs/algorithms.md`.

### Livox Mid-360 + camera on a quadruped

<p align="center">
  <img src="assets/hero_mid360.gif" alt="factor-tracker on GrandTour ETH-2: ANYmal quadruped with a Livox Mid-360 and a front camera among people in the ETH main hall" width="100%">
</p>

<p align="center"><em>An ANYmal quadruped in the ETH Zurich main hall (<a href="https://grand-tour.leggedrobotics.com">GrandTour</a> ETH-2, MIT licence): Livox Mid-360 (mounted upside down, three sweeps accumulated) and the Alphasense front camera, undistorted. Mid-360 detections come from the CODa PV-RCNN 32-beam checkpoint run on CPU (trained on an Ouster, not on Mid-360 data), camera detections from YOLO11x; camera frames arrive ~50 ms after the LiDAR and are associated at their own timestamps. No 3D ground truth exists for this recording; see <a href="docs/mid360.md"><code>docs/mid360.md</code></a>.</em></p>

## What it is

- A small Rust library (`factor-tracker-core`) with two concrete trackers:
  - `BBox2DTracker`: axis-aligned 2D bounding boxes on **one** camera image
    plane.
  - `Cartesian3DTracker`: 3D positions in **one** common frame, with
    multi-sensor sequential fusion.
- A CLI (`factor-tracker`) that generates reproducible synthetic scenarios,
  reads/writes JSONL observations and tracks, and writes a self-contained HTML
  visualization.

## What it is not

- It is **not** a detector: the engine takes detections as input and does no
  raw image, point-cloud or radar-signal processing and no monocular depth.
  (The evaluation scripts run a LiDAR detector on CODa; see
  `docs/detector_training.md`.)
- It does **not** estimate 3D orientation or extent. `Cartesian3DTracker` is
  position tracking, not full 3D box tracking.
- There is no appearance/ReID model, no EKF/UKF, no JPDA/MHT, no
  track-to-track fusion, no out-of-sequence measurement handling, no
  auto-calibration, no ego-motion estimation, no Python bindings and no GUI
  framework (a ROS 2 node is in `ros2/`). See `ROADMAP.md`.
- It is **not** ByteTrack or DeepSORT. Those systems combine specific detectors,
  appearance models and association strategies. This is a plain linear Kalman
  filter with IoU/Mahalanobis gating and Hungarian assignment, plus lifecycle
  management.

## Implemented features

| Area | Status |
| --- | --- |
| 2D constant-velocity Kalman tracker (`[cx, cy, w, h]` + velocities) | implemented |
| 3D constant-velocity Kalman tracker (`[x, y, z]` + velocities) | implemented |
| Linear Kalman filter with Cholesky solves and Joseph-form updates | implemented |
| Hungarian (Kuhn–Munkres) assignment with explicit unmatched costs | implemented |
| Matching cascade for 3D association (recently observed tracks first; on in the engine) | implemented |
| Appearance (ReID) cost in 3D association from per-detection embeddings (off by default) | implemented |
| Birth suppression next to confirmed tracks, with persistent neighbours still born (`min_birth_distance_m`, `birth_persistence`, off by default) | implemented |
| Robot-centric evaluation zone (range and bearing around the robot) | implemented |
| IoU gating (2D) and squared-Mahalanobis gating (3D) | implemented |
| Class-aware gating (`None` class is compatible with everything) | implemented |
| Track lifecycle: Tentative → Confirmed → Lost → Deleted, with recovery | implemented |
| Multi-sensor fusion into shared 3D tracks (sequential, independent noise) | implemented |
| Rigid 3D position/covariance transform helper (`RigidTransform3D`) | implemented |
| Time-based lifecycle independent of update count, timeouts in nanoseconds | implemented |
| Batch de-duplication (sensor + sequence) and past-timestamp rejection | implemented |
| JSONL observation/track/ground-truth I/O with schema version | implemented |
| Synthetic 2D and 3D demos with fixed seeds | implemented |
| Local HTML visualization (2D, and top-down XY for 3D) | implemented |
| Criterion benchmarks at 10/100/1000 objects | implemented |

Future work is listed in [`ROADMAP.md`](ROADMAP.md).

## Layout

```
factor-tracker/
  crates/factor-tracker-core/    tracking core (types, filter, assignment, lifecycle, trackers)
  crates/factor-tracker-engine/  camera-assisted LiDAR SDK (calibration, projection, async, eval, KITTI)
  crates/factor-tracker/     CLI, JSONL schemas, scenarios, reports
  crates/factor-tracker-wasm/    2D tracker exported to WebAssembly for the browser demo
  web/                       browser demo: phone camera -> person detector -> tracker (GitHub Pages)
  scripts/kitti/             KITTI tracking conversion and multi-sequence evaluation (Python + NumPy)
  scripts/coda/              CODa (robot dataset) frame fetch, CPU LiDAR detector, conversion to engine JSONL
  scripts/grandtour/         GrandTour (ANYmal + Livox Mid-360 + cameras) conversion to engine JSONL
  scripts/reid/              appearance embeddings for LiDAR detections from camera crops
  scripts/eval/              LiDAR/camera agreement for recordings without 3D ground truth
  scripts/runpod/            GPU detector runs (OpenPCDet, Ultralytics) on a RunPod pod
  docs/                      architecture, data format, algorithms, camera-lidar, datasets, evaluation
  assets/                    demo assets (hero image)
  .github/workflows/         CI and the GitHub Pages deployment of web/
```

## Camera-assisted LiDAR tracking (SDK)

`factor-tracker-engine` (the crate) turns sensor-native detections, calibration and
timestamps into common-frame 3D tracks plus association diagnostics, for one
camera and one LiDAR. You do not assemble the pipeline yourself:

```
sensor detections
    -> bounded arrival-order reorder buffer (event time preserved)
    -> per-observation coordinate normalization (calibration + ego pose at event time)
    -> LiDAR position tracking (factor-tracker-core)
    -> geometric camera association (projected 3D boxes vs 2D detections)
    -> camera class evidence as a soft prior (camera-assisted mode)
    -> unified track output, diagnostics, comparison and HTML report
```

It is **Camera-assisted LiDAR tracking**: the LiDAR tracks objects; the camera
associates and augments. 3D box shape is projection metadata, not an estimated
state. The camera never creates a 3D position from a 2D box, never updates the
position covariance, never deletes an out-of-view track, and never keeps an old
position valid indefinitely.

Run the end-to-end synthetic demo (fixture, three modes, evaluation, report):

```sh
cargo run -p factor-tracker -- cam-demo --out-dir cam-demo-out
```

This writes setup, inputs, truth, `comparison.json` and a local `report.html`.
The camera panel shows 2D detections and projected tracks; the second panel is
a top-down XY view of LiDAR observations and tracks. Select a frame to see why a
detection was associated or why an observation was rejected.

The three modes are `lidar_only`, `camera_associated` and `camera_assisted`;
they share the same LiDAR input, transforms, filter settings and evaluation
interval. The camera does not update position, so position metrics are not
expected to improve.

Real KITTI data uses the same `replay`/`evaluate`/`report` path after
`prepare`; see [`docs/datasets.md`](docs/datasets.md) and
[`docs/camera-lidar.md`](docs/camera-lidar.md). `scripts/kitti/` converts the
KITTI tracking benchmark and evaluates every sequence; `scripts/coda/` does
the same for the CODa robot dataset (Clearpath Husky, Ouster OS1-128); `scripts/runpod/` runs
GPU detectors (OpenPCDet, Ultralytics) on a RunPod pod to produce the
detections. Real-data numbers are in [`docs/evaluation.md`](docs/evaluation.md);
the demo numbers come from the synthetic fixture.

## Requirements

- Rust stable. The declared minimum supported Rust version (MSRV) is **1.89**,
  which is what the current `nalgebra` release requires. Development and CI are
  verified on a recent stable toolchain.

## Quick start

Generate both synthetic scenarios, track them and write visualizations:

```sh
cargo run -p factor-tracker -- demo --out-dir demo-out
```

Then open `demo-out/demo_2d.html` and `demo-out/demo_3d_fusion.html` in a
browser. The 3D page is a top-down (XY) projection, not a 3D rendering.

The same steps can be run individually:

```sh
# 1. Generate the 2D scenario (observations + ground truth, JSONL)
cargo run -p factor-tracker -- gen2d --out-dir demo-out --seed 42

# 2. Track it and render
cargo run -p factor-tracker -- track2d \
    --input demo-out/observations_2d.jsonl \
    --output demo-out/tracks_2d.jsonl \
    --truth demo-out/ground_truth_2d.jsonl \
    --html demo-out/demo_2d.html

# 3D multi-sensor fusion
cargo run -p factor-tracker -- gen3d --out-dir demo-out --seed 42
cargo run -p factor-tracker -- track3d \
    --input demo-out/observations_3d.jsonl \
    --output demo-out/tracks_3d.jsonl \
    --truth demo-out/ground_truth_3d.jsonl \
    --html demo-out/demo_3d_fusion.html
```

All commands use `--seed` with a fixed default of `42`; scenarios are fully
reproducible for a given seed.

## Rust API example

```rust
use nalgebra::Matrix4;
use factor_tracker_core::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tracker = BBox2DTracker::new(BBox2DTrackerConfig::default())?;

    let batch = ObservationBatch2D {
        sensor_id: SensorId::new(1),
        sequence: 0,
        timestamp: Timestamp::from_nanos(0),
        frame_id: FrameId::new("camera"),
        observations: vec![BBox2DObservation {
            bbox: BBox2D::from_corners(100.0, 50.0, 160.0, 130.0),
            score: 0.95,
            class_label: None,
            covariance: Matrix4::<f64>::identity(),
        }],
    };

    let tracks: Vec<Track2DSnapshot> = tracker.process_batch(&batch)?;
    for track in tracks {
        println!("{} {:?} {:?}", track.track_id, track.lifecycle, track.bbox);
    }
    Ok(())
}
```

Run the compiled examples:

```sh
cargo run -p factor-tracker-core --example quickstart
cargo run -p factor-tracker-core --example two_d_tracking
cargo run -p factor-tracker-core --example three_d_fusion
```

For 3D fusion, convert each sensor's measurement into the common tracking frame
first:

```rust
use nalgebra::{Matrix3, Vector3};
use factor_tracker_core::RigidTransform3D;

fn to_tracking_frame() -> Result<(), Box<dyn std::error::Error>> {
    let sensor_local_to_world =
        RigidTransform3D::new(Matrix3::identity(), Vector3::new(5.0, 0.0, 0.0))?;
    let world_position =
        sensor_local_to_world.transform_position(&Vector3::new(0.0, 0.0, 1.0));
    let world_covariance =
        sensor_local_to_world.transform_covariance(&(Matrix3::<f64>::identity() * 0.25))?;
    let _ = (world_position, world_covariance);
    Ok(())
}
```

## Input / output

Input and output are JSONL: one JSON object per line. Each record has a
`schema_version` (currently `1`) and a `type`. For 2D:

```json
{"type":"bbox2d_batch","schema_version":1,"sensor_id":1,"sequence":0,"timestamp_ns":0,"frame_id":"camera","observations":[{"bbox":[100.0,50.0,160.0,130.0],"score":0.95,"class":1,"covariance":[4,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4]}]}
```

Output lines look like this (abbreviated: the mandatory `covariance` array is
omitted for brevity):

```json
{"type":"bbox2d_tracks","schema_version":1,"timestamp_ns":0,"tracks":[{"track_id":1,"lifecycle":"confirmed","state_timestamp_ns":0,"last_observation_timestamp_ns":0,"last_update_sensor":1,"contributing_sensors":[1],"class":1,"updated":true,"prediction_only":false,"observation_times_count":2,"bbox":[100.0,50.0,160.0,130.0],"velocity":[0.0,0.0],"state":[130.0,90.0,60.0,80.0,0.0,0.0,0.0,0.0]}]}
```

The full schema, units and ordering are documented in
[`docs/data-format.md`](docs/data-format.md).

## Algorithm overview

```
Detections / Measurements
    → Validation and Coordinate Normalization
    → Prediction (constant velocity, dt from timestamps)
    → Data Association (gating + Hungarian)
    → State Update (linear Kalman filter)
    → Track Lifecycle Management (time based)
    → Tracked Objects
```

- **Prediction** uses the real timestamp difference; no fixed frame rate is
  assumed. Process noise is the continuous-time white-acceleration model scaled
  by `dt`.
- **Association** builds a cost matrix (IoU or squared Mahalanobis), applies a
  hard gate plus a class-compatibility gate, and solves a minimum-cost linear
  assignment. "Unmatched" is part of the optimization via dummy rows/columns.
- **Update** is a standard linear Kalman update with a Cholesky solve for the
  gain and a Joseph-form covariance update.
- **Lifecycle** transitions depend only on elapsed time and on distinct
  observation timestamps, never on the number of sensors or on how often
  `process_batch` is called.

Details, including exact equations and units, are in
[`docs/algorithms.md`](docs/algorithms.md).

## Assumptions and contracts

- **Time**: all sensors share one synchronized timeline in integer nanoseconds.
  Observation timestamps must be non-decreasing; a past batch is rejected, not
  treated as "now". Re-sent batches (same sensor and sequence) are rejected.
- **Frames**: each tracker is bound to one frame. A batch from another frame is
  rejected. Renaming a frame is never treated as a coordinate transform. Use
  `RigidTransform3D` explicitly (the transform's own uncertainty is not
  modelled).
- **Noise**: observations are assumed to have independent noise. Correlated
  observations and track-to-track fusion are out of scope.
- **Probability**: the detector `score` is a detection confidence. No track
  existence probability is estimated or output, and `score` is not used as one.
- **Identity**: stable IDs are expected only where the association problem is
  unambiguous. Crossings, prolonged full occlusion and identical nearby objects
  can cause ID switches; zero ID switches are not guaranteed.

## Known limitations

- Linear constant-velocity models only; no maneuver model.
- 2D and 3D observations are not fused with each other (no 2D-box-to-3D
  lifting).
- Box size clamping after prediction is a documented modeling choice; the
  covariance does not model the clamp.
- The linear assignment and Kalman filter are `O(n^3)` in the number of tracks
  / state dimension per update; no gating tree or track management optimization
  is implemented.
- `cargo bench` measures only the tracking core, not detection or video I/O, so
  its numbers are **not** an end-to-end FPS figure.

## Development and verification

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps
cargo bench --workspace --no-run
```

The same checks run in CI (`.github/workflows/ci.yml`).

## Related work and references

The algorithms used here are standard and were implemented from their public
descriptions; no third-party source code was copied.

- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems",
  1960.
- H. W. Kuhn, "The Hungarian method for the assignment problem", 1955;
  J. Munkres, "Algorithms for the Assignment and Transportation Problems", 1957.
  The assignment solver follows the shortest-augmenting-path formulation.
- A. Bewley et al., "Simple Online and Realtime Tracking" (SORT), 2016 — a
  reference point for IoU association and Kalman tracking.
- N. Wojke et al., "Simple Online and Realtime Tracking with a Deep Association
  Metric" (DeepSORT), 2017 — appearance-assisted association (not implemented).
- Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
  Detection Box", 2022 — track lifecycle handling (not implemented).
- The continuous-time white-acceleration process-noise model is the standard
  "discrete white noise acceleration" model, e.g. Y. Bar-Shalom et al.,
  "Estimation with Applications to Tracking and Navigation", 2001.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in this work shall be dual licensed as
above, without any additional terms or conditions.
