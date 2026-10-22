# Usage

Command line demos, the Rust API and the development checks.

## Requirements

- Rust stable. The declared minimum supported Rust version (MSRV) is **1.89**,
  which is what the current `nalgebra` release requires. Development and CI are
  verified on a recent stable toolchain.

## Synthetic demos

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
`prepare`; see [`docs/datasets.md`](datasets.md) and
[`docs/camera-lidar.md`](camera-lidar.md). `scripts/kitti/` converts the
KITTI tracking benchmark and evaluates every sequence; `scripts/coda/` does
the same for the CODa robot dataset (Clearpath Husky, Ouster OS1-128); `scripts/runpod/` runs
GPU detectors (OpenPCDet, Ultralytics) on a RunPod pod to produce the
detections. Real-data numbers are in [`evaluation.md`](evaluation.md);
the demo numbers come from the synthetic fixture.

## Rust API

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
[`docs/data-format.md`](data-format.md).

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
[`docs/algorithms.md`](algorithms.md).

## Development and verification

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps
cargo bench --workspace --no-run
```

The same checks run in CI (`.github/workflows/ci.yml`).
