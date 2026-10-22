<p align="center">
  <img src="assets/compare_ab3dmot.gif" alt="Side-by-side bird's-eye views of the same 16 s of CODa sequence 20: an AB3DMOT-style tracker (left) makes 25 identity switches, factor-tracker (right) makes 7" width="100%">
</p>

<h1 align="center">Factor Graph Object Tracker</h1>

<p align="center"><code>factor-tracker</code></p>

<p align="center">
  <a href="https://github.com/rsasaki0109/factor-tracker/actions/workflows/ci.yml"><img src="https://github.com/rsasaki0109/factor-tracker/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="License: MIT OR Apache-2.0"></a>
  <img src="https://img.shields.io/badge/MSRV-1.89-orange.svg" alt="MSRV 1.89">
  <img src="https://img.shields.io/badge/dependencies-nalgebra%20%7C%20thiserror-informational.svg" alt="Dependencies: nalgebra, thiserror">
</p>

<p align="center"><em>Same detector, two trackers, each at its best setting on dev (CODa seq 20, 16 s). Red rings: identity switches against CODa's labels.</em></p>

Detector-agnostic multi-object tracking and sensor fusion in Rust: Kalman
filtering and assignment per scan, an optional fixed-lag factor-graph
re-association (`--fgo-lag`), and LiDAR tracks that camera detections associate
with and label. It takes detections from any detector and outputs objects with
stable IDs and uncertainty. The core is CPU-only (no GPU, ROS, Python or OpenCV
dependency); a ROS 2 node and a [browser demo](https://rsasaki0109.github.io/factor-tracker/) are included.

## Results

Pedestrian tracking on 8 CODa windows, **same detector for every tracker**
(CODa PV-RCNN first stage on CPU):

| tracker | detections (NMS IoU, score) | dev HOTA | dev IDF1 | dev ID switches | held-out HOTA | held-out IDF1 | held-out ID switches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kalman + Mahalanobis gate | 0.1, >= 0.4 | 0.600 | 0.722 | 1151 | 0.745 | 0.888 | 22 |
| AB3DMOT-style | 0.1, >= 0.4 | 0.618 | 0.746 | 987 | 0.764 | 0.882 | 28 |
| factor-tracker | 0.1, >= 0.4 | 0.618 | 0.760 | 934 | 0.768 | 0.900 | **18** |
| **factor-tracker** + birth suppression (0.8 m) | 0.3, >= 0.5 | **0.628** | **0.772** | **773** | **0.777** | **0.902** | **18** |

Each tracker uses the detection setting that is best for it on dev (5
windows); held-out (3 windows) was scored once. A looser NMS finds more people
in crowds but also duplicate boxes; factor-tracker absorbs the duplicates by
not starting tracks next to confirmed ones, AB3DMOT-style cannot (HOTA 0.600
dev with the same detections). The detector was trained on most of these
scans, so absolute numbers are optimistic for every tracker. Details:
[`docs/evaluation.md`](docs/evaluation.md).

## How it works

Per LiDAR scan (details: [`docs/algorithms.md`](docs/algorithms.md),
[`docs/camera-lidar.md`](docs/camera-lidar.md)):

```mermaid
flowchart TD
  L["LiDAR detections<br/>(any 3D detector)"] --> N["Reorder by event time,<br/>transform to the tracking frame<br/>(calibration + ego pose)"]
  N --> P["Predict every track<br/>(constant-velocity Kalman, dt from timestamps)"]
  P --> C["Cost per track × detection:<br/>distance to the prediction, 1 m gate<br/>(or squared Mahalanobis gate)<br/>+ optional class-prior and appearance terms"]
  C --> H["Hungarian assignment"]
  H -->|matched| U["Kalman update"]
  H -->|unmatched detection| B["Birth<br/>(score threshold, optional suppression<br/>next to confirmed tracks)"]
  H -->|unmatched track| M["Coast on the prediction"]
  U --> LC
  B --> LC
  M --> LC["Lifecycle, time based:<br/>tentative → confirmed → lost → deleted"]
  LC --> O["Tracks: stable ID, position,<br/>velocity, covariance"]
  LC -.->|"optional (fgo-lag)"| F["Fixed-lag factor-graph re-association<br/>over the last N scans: swap / merge /<br/>split / drop when the cost drops"]
  F -.-> O
  CAM["Camera 2D detections"] --> A["Project track boxes into the image,<br/>match them to 2D detections"]
  LC --> A
  A -.->|"class evidence (soft prior)"| C
```

## Demos

<p align="center">
  <img src="assets/hero.gif" alt="factor-tracker on a Clearpath Husky robot driving through a campus crowd (CODa sequence 20): camera frames with projected 3D track boxes (left), Ouster OS1-128 scan with pedestrian tracks in a bird's-eye view (right)" width="100%">
</p>
<p align="center"><em>Camera + LiDAR on a wheeled robot (CODa seq 20, Ouster OS1-128).</em></p>

<p align="center">
  <img src="assets/hero_mid360.gif" alt="factor-tracker on GrandTour ETH-2: ANYmal quadruped with a Livox Mid-360 and a front camera among people in the ETH main hall" width="100%">
</p>
<p align="center"><em>Livox Mid-360 + camera on a quadruped (GrandTour ETH-2); see <a href="docs/mid360.md"><code>docs/mid360.md</code></a>.</em></p>

## Quick start

```sh
cargo run -p factor-tracker -- demo --out-dir demo-out        # synthetic 2D/3D tracking + HTML views
cargo run -p factor-tracker -- cam-demo --out-dir cam-out     # camera + LiDAR pipeline + report
```

Rust 1.89 or newer. More: [`docs/usage.md`](docs/usage.md) (CLI, Rust API,
SDK), [`ros2/`](ros2/factor_tracker_ros) (ROS 2 node), [live browser demo](https://rsasaki0109.github.io/factor-tracker/)
(phone camera, source in [`web/`](web)).

## Documentation

- [`docs/usage.md`](docs/usage.md): CLI, Rust API, input/output, algorithm overview
- [`docs/architecture.md`](docs/architecture.md): design, scope, assumptions, limitations, references
- [`docs/algorithms.md`](docs/algorithms.md), [`docs/data-format.md`](docs/data-format.md), [`docs/camera-lidar.md`](docs/camera-lidar.md)
- [`docs/datasets.md`](docs/datasets.md), [`docs/evaluation.md`](docs/evaluation.md), [`docs/detector_training.md`](docs/detector_training.md)
- [`ROADMAP.md`](ROADMAP.md)

## License

Code: MIT OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)). The animations contain dataset images:
CODa ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
Zhang et al.) and [GrandTour](https://grand-tour.leggedrobotics.com) (MIT).
