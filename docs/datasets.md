# Datasets and the KITTI adapter

## Why KITTI

Only one dataset is supported, on purpose. KITTI was chosen because:

- camera and LiDAR are available for the same sequences,
- calibration (`calib.txt`) and timestamps (`times.txt`) are documented,
- the odometry benchmark distributes ego poses (`poses.txt`), which the adapter
  uses as an **external input** for coordinate normalization,
- the label/detection formats are plain text and have a tracking-id variant,
- ground truth (3D boxes and 2D boxes) is available for evaluation.

Other datasets were not added.

## Licence and availability

The KITTI data, labels and development kit are **not** redistributed in this
repository. They are downloaded from `https://www.cvlibs.net/datasets/kitti/`
after accepting the dataset terms. Check the site for the current terms before
downloading or redistributing anything. The adapter code in this repository is
original and covered by this project's MIT OR Apache-2.0 licence; the data and
any third-party weights keep their own licences.

**No real KITTI data was downloaded or executed in this environment.** Network
access to the KITTI download requires registration, so the adapter, conversion,
tests and procedures are implemented and unit-tested against synthetic files in
the KITTI text format, but the numbers reported by the demos come from the
synthetic fixture, not from real KITTI. Do not read the synthetic results as
real-data validation.

## Expected layout

`prepare` expects per-frame text files:

```
sequence/
  calib.txt                     # P_rect_00 (or P2), R_rect_00, Tr_velo_to_cam
  times.txt                     # one timestamp in seconds per line
  poses.txt                     # optional, one row-major 3x4 camera pose per line
  detections_lidar/000000.txt   # 3D detections, KITTI label format (+ optional score)
  detections_camera/000000.txt  # 2D detections, bbox + optional score
  labels_lidar/000000.txt       # optional, 3D labels with tracking ids (truth)
```

3D labels and 3D detections are interpreted in the camera0 rectified frame and
re-expressed in the velodyne frame:

```text
T_velo_from_cam     = inverse(R_rect_00 * Tr_velo_to_cam)
camera_to_reference = inverse(T_velo_from_cam)   # cam0 <- velo
lidar_to_reference  = identity                    # velodyne
```

The velodyne frame is the reference/tracking frame. Ego poses are camera0-to-
world and are composed with `camera_to_reference` to map velodyne to world.

## Obtaining detection results

There is no requirement to train a detector. Two reproducible options:

1. **Use the labels as oracle detections.** Point `detections_lidar` /
   `detections_camera` at the label files (the parsers accept the label format).
   This measures the tracking and association pipeline, not a detector.
2. **Run an external detector** and convert its output to the KITTI text
   format. Any detector can be used; the engine only needs 3D boxes/positions
   and 2D boxes. A Python preparation script is acceptable but is not a Rust
   core dependency.

## Commands

```sh
cargo run -p perception-cli -- prepare \
    --calib sequence/calib.txt \
    --times sequence/times.txt \
    --poses sequence/poses.txt \
    --lidar-detections sequence/detections_lidar \
    --camera-detections sequence/detections_camera \
    --lidar-labels sequence/labels_lidar \
    --out-setup cam-demo-out/setup.json \
    --out-inputs cam-demo-out/inputs.jsonl \
    --out-truth cam-demo-out/truth.jsonl

cargo run -p perception-cli -- replay --setup cam-demo-out/setup.json \
    --inputs cam-demo-out/inputs.jsonl --mode camera_assisted \
    --output cam-demo-out/outputs.jsonl --diagnostics cam-demo-out/diagnostics.json

cargo run -p perception-cli -- evaluate --output cam-demo-out/outputs.jsonl \
    --truth cam-demo-out/truth.jsonl --metrics cam-demo-out/metrics.json

cargo run -p perception-cli -- report --setup cam-demo-out/setup.json \
    --output cam-demo-out/outputs.jsonl --truth cam-demo-out/truth.jsonl \
    --metrics cam-demo-out/metrics.json --html cam-demo-out/report.html
```

## Synthetic fixture

When real data is unavailable, `cam-fixture` (or `cam-demo`) writes a
deterministic synthetic camera+LiDAR sequence in the engine JSONL schema plus a
setup file. It exercises calibration, ego-pose normalization, projection,
camera association, class evidence and the three modes. It is clearly labelled
as synthetic in every output.

## Ground-truth separation

Truth records are only read by `evaluate` and the report. They are never passed
to `EngineBatch` inputs and never enter the tracker. The synthetic 3D ground
truth and the camera detection-to-truth mapping are written to a separate
`truth.jsonl`.
