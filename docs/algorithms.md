# Algorithms

This document records the exact models, units and decisions used by the
trackers. It also reports the measured micro-benchmark setup. Nothing here is
compared to published trackers and no accuracy metric is claimed.

## Pipeline

```
ObservationBatch
    → validation
    → prediction (dt from timestamps)
    → data association (gating + Hungarian)
    → Kalman update (matched tracks)
    → new tracks (unmatched observations)
    → lifecycle transitions
    → TrackSnapshot
```

## Coordinate systems and units

- 2D: image pixels. State `[cx, cy, w, h, vcx, vcy, vw, vh]`; `cx, cy` are the
  box center, `w, h` its width/height; velocities in pixels/s. Measurement
  `[cx, cy, w, h]`, covariance a `4x4` in pixels².
- 3D: meters. State `[x, y, z, vx, vy, vz]`; velocity in m/s. Measurement
  `[x, y, z]`, covariance a `3x3` in m². Positions must already be in the
  tracker's frame; use `RigidTransform3D` to convert, which applies
  `p' = R p + t` and `C' = R C R^T` (transform uncertainty is not propagated).

## Validation

Before any state change the tracker rejects:

- non-finite values (NaN / infinity) in observations or covariances;
- covariances that are not finite, symmetric or positive definite;
- 2D boxes with non-positive width or height;
- detection scores outside `[0, 1]`;
- frame mismatches;
- timestamps earlier than the tracker's current time;
- batches whose `(sensor, sequence)` was already accepted;
- timestamp differences that overflow `i64` nanoseconds.

Input errors and numerical errors are distinct types. A failed Cholesky
decomposition is reported, never replaced by a zero or identity matrix.

## Prediction

The transition uses the actual `dt` derived from timestamps (no assumed frame
rate). Both trackers use the continuous-time white-acceleration model. For a
state ordered `[position, velocity]` along one axis, with acceleration spectral
density `q`:

```
F = [ 1  dt ]          Q = q * [ dt^3/3  dt^2/2 ]
    [ 0   1 ]                   [ dt^2/2  dt     ]
```

Axes are independent, so `F` is identity with `dt` on the position→velocity
block and `Q` is block diagonal.

- 2D: four axes (`cx, cy` with `q_pos`; `w, h` with `q_size`), all in
  pixels²/s³.
- 3D: three axes with one `q`, in m²/s³.

`q` is the acceleration white-noise spectral density, configurable per tracker.
Prediction propagates `x' = F x` and `P' = F P Fᵀ + Q`, then symmetrizes `P`.

### Large time gaps

Lifecycle transitions are evaluated *before* prediction. Any track that has not
been observed within `lost_to_delete_ns` is deleted first, so for a track that
survives, `dt` is bounded by the configured time-outs; the demo's sensor pause
is handled this way. A timestamp difference that cannot be represented in `i64`
nanoseconds is reported as `InputError::TimestampOverflow` rather than wrapping
or saturating silently.

### Box size under prediction

Because the 2D model is unbounded, a long prediction can drive `w` or `h` to
zero or below. After prediction, any size at or below `min_bbox_size` is
clamped to that floor and the corresponding size velocity is set to zero. The
covariance is not modified. This is a documented modelling choice; invalid
*input* boxes are rejected instead.

## Update

Standard linear Kalman update:

```
y = z - H x
S = H P Hᵀ + R
K = P Hᵀ S⁻¹
x' = x + K y
P' = (I - K H) P (I - K H)ᵀ + K R Kᵀ     (Joseph form)
```

`S⁻¹` is never formed. The gain is computed by solving `S Kᵀ = (P Hᵀ)ᵀ` with a
Cholesky decomposition, and `P` is symmetrized after every step.

Independence of two same-time measurements can be verified: updating with a
stacked measurement `z = [z₁; z₂]`, `H = [H₁; H₂]`, `R = diag(R₁, R₂)` gives the
same result as the two sequential updates (tested to `1e-9`).

## Data association

### Costs

- 2D: `cost = 1 - IoU(predicted_bbox, observed_bbox)`, dimensionless in
  `[0, 1]`. A pair is gated out when `IoU < gate_min_iou`.
- 3D: `cost = d²`, the **squared** Mahalanobis distance of the position from the
  track prediction using `S = H P Hᵀ + R`, for the **3-dimensional** position
  measurement. A pair is gated out when `d² > gate_squared_mahalanobis`
  (default `16.266`, the 0.999 quantile of a chi-square with 3 degrees of
  freedom).

Different units and scores are never summed. The detection `score` is not part
of the association cost; it only gates new-track creation in 2D.

### Class gating

If both the track and the observation have a known class and they differ, the
pair is forbidden. An unknown class (`None`) is compatible with any class.

### Linear assignment

The cost matrix is solved with a shortest-augmenting-path Hungarian
(Kuhn–Munkres) implementation. "Unmatched" is represented *inside* the problem:
one dummy column is added per row and one dummy row per column, with explicit
unmatched costs, and forbidden pairs are set to a large finite sentinel so they
can never be chosen and never block a valid pair. Forbidden pairs are passed as
non-finite and are sanitized internally (a non-finite value is treated as
"forbidden", not as "cheap").

Because there are no allowed edges between disconnected parts of the gating
graph, the problem is decomposed into connected components first and each
component is solved independently. This is exact (the global optimum is the
union of component optima) and makes many separated objects near-linear rather
than cubic. Rectangular problems, empty rows/columns and all-forbidden cases
are handled explicitly.

## Track lifecycle

```
Tentative --(confirm_min distinct observation times)--> Confirmed
Tentative --(no observation for tentative_timeout)-----> Deleted
Confirmed --(no observation for confirmed_to_lost)-----> Lost
Lost      --(observed)--------------------------------> Confirmed
Lost      --(no observation for lost_to_delete)--------> Deleted
```

- Confirmation counts **distinct observation timestamps**: several sensors
  reporting at the same timestamp count once.
- All time-outs are measured from the track's last observation, so lifecycle
  does not depend on how often `process_batch` is called or on the number of
  sensors.
- An empty batch from one sensor does not make another sensor's track `Lost`;
  only elapsed time does.
- Track ids are monotonically increasing and never reused, including after
  deletion.

## Multi-sensor fusion

Each batch is fused independently and sequentially. After one sensor updates a
track, its covariance shrinks; the next sensor's measurement is then fused into
that smaller covariance, which is correct for independent measurements. The
implementation assumes independent observation noise. Correlated observations,
track-to-track fusion and out-of-sequence measurements are out of scope; a
same-time arrival is simply processed in input order.

## Identity and ID switches

Association is based only on geometry (and class). Objects that are spatially
ambiguous — during crossings, prolonged full occlusion, or when two identical
objects are very close — can be assigned to the wrong track, causing an ID
switch. The library does not guarantee zero ID switches. The 2D demo contains a
temporary full occlusion, and the docs do not claim that identity survives it;
the tests only assert stable IDs in scenarios where the objects are well
separated.

## Benchmarks

### What is measured

Only `perception-core` tracking is timed. Observation batches are built before
the timed section; no detection, video decoding or other sensor processing is
included. The reported numbers are therefore **not** an end-to-end FPS figure
and are not comparable to detector throughput.

Each benchmark processes **20 frames** for **N** objects moving with small
constant velocity in a well-separated grid, with a small amount of position
noise. Seeds are fixed (`SplitMix64`) so the workload is reproducible.

### Environment

| Item | Value |
| --- | --- |
| CPU | 11th Gen Intel Core i5-1145G7 @ 2.60 GHz, 8 logical CPUs |
| Memory | 30 GiB |
| OS | Linux (kernel 7.0.0-30-generic), x86_64 |
| Rust | rustc 1.97.0, cargo 1.97.0 |
| Profile | `bench` (inherits `release`: `lto = "thin"`, `codegen-units = 1`) |
| Criterion | 0.8, `--warm-up-time 1 --measurement-time 2 --sample-size 10` |

### Results (median, time for 20 frames)

| Benchmark | 10 objects | 100 objects | 1000 objects |
| --- | --- | --- | --- |
| `track_2d` | 1.18 ms | 14.8 ms | 423 ms |
| `track_3d_fusion` | 1.11 ms | 51.5 ms | 4.50 s |

Derived per-frame figures: 2D ≈ 0.06 / 0.74 / 21 ms; 3D ≈ 0.06 / 2.6 / 225 ms.

These are single-machine micro-benchmark medians from a short run. They are not
a guarantee, not a product specification, and no comparison to any other
tracker is implied. Reproduce with:

```sh
cargo bench -p perception-core -- --warm-up-time 1 --measurement-time 2 --sample-size 10
```

The 3D benchmark is dominated by the `O(N²)` gating/cost construction (a
squared-Mahalanobis evaluation per track/observation pair) rather than by the
linear assignment, which is decomposed by connected components.
