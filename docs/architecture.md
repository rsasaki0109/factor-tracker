# Architecture

## Overview

`factor-tracker` is a small Cargo workspace with two crates:

- `factor-tracker-core` — the tracking library. It has no file I/O, networking or
  media dependencies. Its only runtime dependencies are `nalgebra` and
  `thiserror`.
- `factor-tracker` — a thin binary around the library that performs file I/O,
  synthetic scenario generation and HTML visualization.

The processing pipeline is:

```
ObservationBatch (one sensor, one timestamp)
    → validation and coordinate normalization
    → prediction (dt from timestamps)
    → data association (gating + Hungarian)
    → state update (linear Kalman filter)
    → track lifecycle management
    → TrackSnapshot list
```

Ground truth never enters the library. It exists only in the CLI scenario
generator and the renderer.

## Module map (`factor-tracker-core`)

| Module | Responsibility |
| --- | --- |
| `types` | IDs, `Timestamp`, `BBox2D`, observations, batches, `TrackerConfig` |
| `error` | `InputError`, `NumericError`, `TrackerError` |
| `math::linalg` | finite/symmetry/positive-definite checks, Cholesky solves, Mahalanobis |
| `math::models` | constant-velocity `F`/`Q` matrices and measurement matrices |
| `filter` | statically sized linear `KalmanFilter<N, M>` |
| `assignment` | Hungarian solver with explicit unmatched costs |
| `association` | IoU / Mahalanobis costs and class gating |
| `lifecycle` | time-based `TrackLifecycle` (crate-private) |
| `tracker::tracker2d` | `BBox2DTracker` and `Track2DSnapshot` |
| `tracker::tracker3d` | `Cartesian3DTracker` and `Track3DSnapshot` |
| `transform` | `RigidTransform3D` for positions and covariances |

## Why two tracker types

The 2D and 3D trackers share the filter, assignment solver, lifecycle and error
model, but their state spaces, units and association costs are different. They
are kept as separate concrete types rather than one generic "object tracker"
because an abstraction over both would either erase units or force the geometry
into an ambiguous representation. Shared code is factored into modules; the
geometry-specific parts stay explicit.

## Error model

Errors are split into two disjoint categories:

- `InputError` — the caller violated a documented contract: non-finite values,
  malformed covariances, non-positive box sizes, reversed or overflowing
  timestamps, duplicated batches, frame mismatches, invalid configuration.
- `NumericError` — the input was valid but a decomposition failed (typically a
  non-positive-definite innovation covariance) or a non-finite intermediate was
  produced.

A failed decomposition is never silently replaced by a zero or identity matrix.
The core reports the failure.

## Transactional batch processing

`process_batch` and `predict_to`:

1. validate the entire input without mutating state, then
2. take a cheap clone of the tracker state, run the update, and
3. restore the clone if any error occurs.

This guarantees that an invalid batch never leaves the tracker partially
updated. Timestamp ordering, batch de-duplication, frame checks and observation
validation all happen before any prediction or update.

## Time

- Time is integer nanoseconds in `i64` on a shared, pre-synchronized epoch.
- Input timestamps must be non-decreasing. Same-time batches from different
  sensors are accepted. A past timestamp is an explicit error; it is never
  clamped to "now".
- The tracker keeps the highest batch sequence per sensor and rejects repeats,
  so a retransmission is not fused twice.
- Lifecycle transitions are computed from elapsed time and from the set of
  distinct observation timestamps. They do not depend on call counts or sensor
  counts.

## Multi-sensor fusion

Each `ObservationBatch3D` is processed independently. Several sensors can update
the same track in sequence; after the first update the track's covariance
shrinks, which is correct for independent measurements. Observations are assumed
to have independent noise; correlations between sensors are not modelled.

When several batches share a timestamp, they are processed in input order. The
tracker predicts to that timestamp once (subsequent same-time predictions are
no-ops) and fuses each batch.

## Frames and transforms

A `BBox2DTracker` is bound to one camera frame and a `Cartesian3DTracker` to one
common 3D frame. Observations from another frame are rejected. Bringing a
measurement into the tracking frame is the caller's responsibility;
`RigidTransform3D` validates a rotation/translation and transforms positions and
covariances, but it does **not** model uncertainty in the transform and does not
estimate ego-motion.

## Numerics

- The Kalman gain is obtained by solving `S K^T = (P H^T)^T` with Cholesky; the
  inverse of `S` is never formed.
- Covariance updates use the Joseph form and the result is symmetrized every
  step.
- Allowed values are validated before use; NaNs and infinities are rejected at
  the boundary.

## Scope

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

Future work is listed in [`ROADMAP.md`](../ROADMAP.md).

## Repository layout

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

## References

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
