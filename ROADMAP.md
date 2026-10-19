# Roadmap

This roadmap lists directions that are **not implemented** in the initial
version. Nothing here should be read as a promise or a schedule. Items are
grouped roughly by theme and are ordered by rough difficulty/priority within
each group.

## Observation models

- 3D oriented bounding boxes: orientation and extent estimation.
- Radar measurement models for range / bearing / Doppler.
- Sensor-specific observation models beyond Cartesian position and image
  bounding boxes.

## Filtering and fusion

- Extended and unscented Kalman filters for non-linear observation models.
- Interacting multiple model (IMM) or other maneuver models.
- Out-of-sequence measurement buffering and re-processing.
- Explicit modelling of correlated observations between sensors.
- Track-to-track fusion of independent trackers.
- Auto-calibration of sensor extrinsics.

## Association and track management

- Appearance-assisted association: the cost term and per-track embeddings are
  implemented, but on CODa (one forward camera) only 13 % of LiDAR detections
  get an embedding and ID switches do not change (`docs/evaluation.md`).
  Combined with persistent birth suppression it cuts switches within 8 m of
  the robot by 21 % (33 % ahead) over five CODa windows with real
  detections, at no cost with oracle detections. Skipping occluded crops and
  a geometry-only LiDAR shape descriptor did not help (`docs/evaluation.md`).
  Mid-360 reflectivity separates people about as well as point-cloud shape
  and adds little to it (`docs/mid360.md`). A first Mid-360 identity
  benchmark exists (`benchmarks/grandtour_eth2`, 20 s, AI-annotated). Next: a
  ReID-trained model, a learned point-cloud descriptor, more (and
  human-checked) Mid-360 labels, and more windows with a crowded forward zone.
- Joint probabilistic data association (JPDA) and multiple hypothesis tracking
  (MHT). A fixed-lag re-association by factor-graph cost exists
  (`--fgo-lag`, -24 % switches within 8 m on CODa at 1 s lag); next: score
  it on a held-out set (JRDB), use the observations inside a close encounter
  better, and combine it with an IMM motion model.
- Learned or adaptive gating thresholds.

## Implemented since the initial version

- Matching cascade for 3D association (`recent_track_window_ns`).
- Real-dataset harness for KITTI tracking and CODa (`scripts/kitti`,
  `scripts/coda`), including CPU inference of the CODa PV-RCNN first stage.

## Implemented in the camera-assisted LiDAR SDK

- Camera / LiDAR adapter (KITTI text formats) that converts sensor-native data
  into observation batches.
- Camera 2D box to LiDAR 3D track association through a calibrated rectified
  pinhole projection, with per-association diagnostics.
- Camera intrinsics and sensor-to-sensor extrinsics.
- Coordinate normalization with externally supplied ego poses (interpolated,
  never extrapolated).
- Bounded, event-time-preserving asynchronous reordering with late / duplicate
  / overflow handling and deterministic same-time order.
- Camera class evidence as a soft association prior.
- Three-mode comparison (lidar-only / camera-associated / camera-assisted) and
  a local HTML report.

## Not yet implemented

- Ego-motion **estimation** (poses are an external input; none is estimated or
  evaluated here).
- Radar adapters and radar observation models.
- Camera distortion models other than a rectified pinhole.
- Non-linear (EKF/UKF) camera updates.
- ROS 2 adapter.
- Python bindings.
- Wasm bindings.
- More datasets (KITTI tracking and CODa are supported through scripts).

## Performance

- GPU acceleration.
- Sparse / large-scale association.
- Track management structures for very large numbers of tracks.

## Tooling and demos

- A larger GUI or interactive viewer.
- A dynamic plugin system for detectors and trackers.
- Real-dataset evaluation harness (for example MOTChallenge JSON) with
  clearly-named metrics. Any reported metric must be implemented and cited
  correctly; no metric will be labelled HOTA/IDF1 unless it actually is.

## Explicitly out of scope for now

- Object detection model training or inference.
- Monocular depth estimation.
- Large front-end application frameworks.
