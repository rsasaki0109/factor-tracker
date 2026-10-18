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

The tracking benchmark calibration, OXTS and label archives (about 10 MB) have
been run through the conversion and the engine with the labels as oracle
detections (see `docs/evaluation.md`). That checks frames, projection and
poses on real files; it says nothing about detector performance. Point clouds
and images are only needed for real detectors (`scripts/runpod/`). The demo
numbers still come from the synthetic fixture.

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

## Converting the KITTI tracking benchmark

`scripts/kitti/convert_tracking.py` turns the official tracking archives into
the layout above, one directory per sequence:

```sh
python3 scripts/kitti/convert_tracking.py --kitti-root <kitti> --out out/sequences --oracle
# or, with detector output in <det>/{lidar,camera}/<seq>/<frame>.txt:
python3 scripts/kitti/convert_tracking.py --kitti-root <kitti> --out out/sequences --detections-root <det>
scripts/kitti/evaluate_sequences.sh out/sequences out/results
```

- `calib.txt` is copied unchanged; the adapter reads the tracking format
  (`R_rect` / `Tr_velo_cam` without a colon, `3x3` `R_rect`).
- The camera is `image_02`, so `P2` is used. Its translation column (camera 2
  is about 6 cm from camera 0) is applied as part of `camera_to_reference`.
- `times.txt` uses the nominal 10 Hz frame period: the tracking benchmark has
  no per-frame timestamps.
- `poses.txt` is computed from OXTS with the devkit formula
  (`convertOxtsToPose`), mapped to rectified camera 0 with
  `R_rect * Tr_velo_cam * Tr_imu_velo` and anchored at the first frame. With
  poses, tracking (and the ground truth) is in this `world` frame.
- `labels_lidar/` holds the tracking labels split per frame; `DontCare`
  entries are dropped.
- `image_size.txt` is read from the first `image_02` PNG when present.

To look at a few frames without downloading the full image (~16 GB) or
Velodyne (~37 GB) archives, `scripts/kitti/fetch_frames.py` extracts a frame
range of one sequence from the official ZIPs with HTTP range requests.

`--oracle` and `--detections-root` can be combined: sensors present under
`--detections-root` use the detector output, the others the labels.

## Obtaining detection results

There is no requirement to train a detector. Two reproducible options:

1. **Use the labels as oracle detections.** Point `detections_lidar` /
   `detections_camera` at the label files (the parsers accept the label format).
   This measures the tracking and association pipeline, not a detector.
2. **Run an external detector** and convert its output to the KITTI text
   format. Any detector can be used; the engine only needs 3D boxes/positions
   and 2D boxes. `scripts/runpod/` runs OpenPCDet (LiDAR) and an Ultralytics
   COCO model (camera) on a GPU pod; see `scripts/runpod/README.md`. These
   scripts are not a Rust core dependency.

3D detection lines may carry the score as a 16th field. 2D detection lines may
be the short form `type trunc occ alpha x1 y1 x2 y2 score`, a 15-field label
line (score 1.0) or a full 16-field detection line (score last).

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

## CODa (robot)

[CODa](https://github.com/ut-amrl/coda-devkit) (UT Campus Object Dataset) is
recorded from a Clearpath Husky robot driving at walking speed through the UT
Austin campus: an Ouster OS1-128 LiDAR hardware-triggering two global-shutter
cameras at 10 Hz, 3D boxes with instance ids for 50+ classes (mostly
pedestrians in busy sequences) and globally consistent SLAM poses. It is
licensed CC BY-NC-SA 4.0 with additional dataset terms (see the devkit's
`docs/DATA_REPORT.md`); nothing from it is redistributed here except the
README animation, which carries that licence.

The per-sequence ZIPs on the TACC server are 40-110 GB, so
`scripts/coda/fetch_coda.py` extracts a frame range with parallel HTTP range
requests (rectified cam0 images, ego-compensated scans, 3D boxes, calibration,
timestamps, `dense_global` poses). `scripts/coda/convert_coda.py` writes engine
JSONL directly:

- reference frame `os1`; tracking frame `world` from `poses/dense_global`
  (one pose per LiDAR frame);
- camera: stereo-rectified cam0, intrinsics from the rectified projection
  matrix, `camera_to_reference = inverse(R_rect * T_os1_cam0)`; LiDAR and
  camera share the LiDAR timestamp (hardware trigger);
- LiDAR detections: `--lidar-detections` reads per-frame detector output
  (CODa-style JSON with scores, from `scripts/coda/detect_lidar_cpu.py`, which
  runs the first stage of the published CODa PV-RCNN checkpoint on CPU);
  without it the CODa 3D boxes are used as oracle detections. Only dynamic
  classes (pedestrian, vehicles, scooter/segway/skateboard) are kept, position
  at the box centre, shape as explicit corners;
- camera detections: per-frame KITTI 2D files, e.g. from
  `scripts/runpod/detect_camera_yolo.py --image-dir ... --out-dir ...`;
- ground truth: the same boxes, instance ids made unique across classes.

```sh
python3 scripts/coda/fetch_coda.py --coda-root CODa --sequence 20 --first 8640 --last 8800
python3 scripts/runpod/detect_camera_yolo.py --image-dir CODa/2d_rect/cam0/20 --out-dir det/coda20
curl -O https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/pretrained_models/128channel/coda128_allclass_bestoracle.pth
python3 scripts/coda/detect_lidar_cpu.py --ckpt coda128_allclass_bestoracle.pth --coda-root CODa \
    --sequence 20 --first 8640 --last 8800 --out det/coda20_lidar
python3 scripts/coda/convert_coda.py --coda-root CODa --sequence 20 --first 8640 --last 8800 \
    --camera-detections det/coda20 --lidar-detections det/coda20_lidar --out coda-seq/20
REPLAY_ARGS="--initial-velocity-variance 1 --accel-spectral-density 0.5" \
EVAL_ARGS="--match-distance-m 1" scripts/kitti/evaluate_sequences.sh coda-seq coda-results 20
```

## GrandTour (ANYmal, Livox Mid-360)

[GrandTour](https://grand-tour.leggedrobotics.com) (ETH Zurich RSL, MIT
licence, Zarr on Hugging Face `leggedrobotics/grand_tour_dataset`) is recorded
on an ANYmal quadruped with a Livox Mid-360, a Hesai XT32, five Alphasense
global-shutter cameras and more. It has no 3D object labels. Download a
mission's `data/` tarballs (`tf`, `anymal_state_odometry`, `<camera>`,
`livox_points_undistorted`) and `images/<camera>.tar`, untar them, then:

```sh
python3 scripts/grandtour/grandtour.py prepare --mission GT/2024-10-01-11-47-44 \
    --camera alphasense_front_center --first-image 3700 --last-image 3900 --out gt-seq/eth2
python3 scripts/runpod/detect_camera_yolo.py --image-dir gt-seq/eth2/images --out-dir det/eth2_cam
python3 scripts/coda/detect_lidar_cpu.py --ckpt coda32_allclass_bestoracle.pth \
    --scan-pattern 'gt-seq/eth2/scans/{frame:06d}.bin' --first 3700 --last 3900 \
    --z-offset -0.25 --out det/eth2_lidar
python3 scripts/grandtour/grandtour.py assemble --out gt-seq/eth2 \
    --lidar-detections det/eth2_lidar --camera-detections det/eth2_cam --lidar-score-threshold 0.5
REPLAY_ARGS="--initial-velocity-variance 1 --accel-spectral-density 0.5" \
    scripts/kitti/evaluate_sequences.sh gt-seq gt-res eth2
python3 scripts/eval/camera_consistency.py --seq-dir gt-seq/eth2
```

Conventions, each checked on the data before use:

- static TF entries map base coordinates into the sensor frame
  (`p_sensor = A_sensor p_base`, `box_base` composed in): projecting Mid-360
  points into the camera with `A_cam A_lidar^-1` lands the floor rings and
  pillars on the image, the opposite composition does not;
- `anymal_state_odometry` is world-from-base (`p_odom = pose p_base`, `xyzw`
  quaternions): two sweeps 1 s apart overlap with a median nearest-neighbour
  distance of 5 cm, versus 1.6 m with the inverse;
- the Alphasense cameras are equidistant fisheye and are undistorted to a
  pinhole with the same K; on ETH-1/2/3 the HDR cameras have imprecise
  timestamps (GMSL arrival time), so the PTP-synchronized Alphasense cameras
  are used.

See `docs/mid360.md` for what this sensor pair implies for perception.
