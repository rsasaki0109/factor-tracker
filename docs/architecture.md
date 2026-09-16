# Architecture

## Overview

`perception-engine` is a small Cargo workspace with two crates:

- `perception-core` — the tracking library. It has no file I/O, networking or
  media dependencies. Its only runtime dependencies are `nalgebra` and
  `thiserror`.
- `perception-cli` — a thin binary around the library that performs file I/O,
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

## Module map (`perception-core`)

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
