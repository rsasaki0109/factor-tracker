# KITTI tracking on a RunPod GPU pod

These scripts run real detectors over the KITTI tracking benchmark and feed
the results to `perception-cli`. The GPU is used only to produce detections;
the engine itself stays CPU-only.

```
KITTI tracking ──► detect_lidar_pcdet.py (OpenPCDet, GPU) ──┐
                └► detect_camera_yolo.py (Ultralytics, GPU) ┴► convert_tracking.py ──► evaluate_sequences.sh
                                                                 (scripts/kitti)         (prepare / replay / evaluate / report)
```

> Status: the conversion and evaluation half (`scripts/kitti`) has been run on
> the real KITTI tracking calibration, OXTS and labels (oracle detections, see
> `docs/evaluation.md`). The two detector scripts have **not** been run yet
> (no GPU in the environment that wrote them); expect to fix small API
> details on the first pod run.

## 1. Pod

- Template: a RunPod PyTorch template (CUDA 11.8 or 12.x). Any 16 GB+ GPU is
  enough; PointPillars and YOLO inference are light.
- Volume: ~70 GB at `/workspace` (velodyne 37 GB + images 16 GB + outputs).

```sh
cd /workspace
git clone https://github.com/rsasaki0109/perception-engine.git
bash perception-engine/scripts/runpod/setup.sh
```

## 2. Data

Accept the KITTI terms at <https://www.cvlibs.net/datasets/kitti/> first. The
data is not redistributed by this repository.

```sh
KITTI_ROOT=/workspace/kitti bash perception-engine/scripts/runpod/download_kitti_tracking.sh
```

## 3. Checkpoint

Download a KITTI checkpoint from the
[OpenPCDet model zoo](https://github.com/open-mmlab/OpenPCDet#kitti-3d-object-detection-baselines)
(for example PointPillars, `pointpillar_7728.pth`) and put it under
`/workspace/ckpt/`. The config must match the checkpoint
(`cfgs/kitti_models/pointpillar.yaml` for PointPillars). Note that the zoo
models are trained on the KITTI *object* benchmark, whose training images
overlap with the tracking sequences, so the numbers are optimistic for a
detector and must not be read as a held-out result.

## 4. Run

```sh
cd /workspace/perception-engine
PCDET_CKPT=/workspace/ckpt/pointpillar_7728.pth bash scripts/runpod/run_all.sh          # all 21 sequences
PCDET_CKPT=/workspace/ckpt/pointpillar_7728.pth bash scripts/runpod/run_all.sh 0000 0001 # a subset
```

Variables: `KITTI_ROOT` (default `/workspace/kitti`), `OUT` (default
`/workspace/pe-out`), `PCDET_CFG`, `YOLO_MODEL` (default `yolo26x.pt`).

Outputs under `$OUT`:

| Path | Content |
| --- | --- |
| `detections/{lidar,camera}/<seq>/<frame>.txt` | raw detector output, KITTI text format |
| `detections.tar.gz` | the same, archived for download |
| `sequences/<seq>/` | layout read by `perception-cli prepare` |
| `results/<seq>/` | per-mode outputs, metrics, diagnostics, `report.html` |
| `results/summary.md`, `summary.json` | metrics aggregated over sequences |

The evaluation step runs only when `cargo` is installed in the pod
(`curl https://sh.rustup.rs -sSf | sh -s -- -y`). Otherwise download
`detections.tar.gz` (`runpodctl send`, or the pod's file browser) and run
locally:

```sh
python3 scripts/kitti/convert_tracking.py --kitti-root <kitti> --out out/sequences \
    --detections-root out/detections
scripts/kitti/evaluate_sequences.sh out/sequences out/results
```

The detector output is derived from KITTI data; keep it under the KITTI
terms and do not commit it to this repository.

# CODa: full two-stage PV-RCNN detections

`scripts/coda/detect_lidar_cpu.py` runs only the first stage of the CODa
PV-RCNN checkpoint (the second stage needs CUDA operators). On a GPU pod,
`coda_pvrcnn_orig.sh` runs the whole model (the coda-models fork of OpenPCDet
that the checkpoint was trained with) over the CODa windows in `WINDOWS`
(`seq:first`, 160 scans each; default the two development windows below) and
packs the detections:

```sh
cd /workspace
git clone -b claude/dreamy-noether-2h4v4t https://github.com/rsasaki0109/perception-engine.git
bash perception-engine/scripts/runpod/coda_pvrcnn_orig.sh
# -> /workspace/coda_pvrcnn_dets.tar.gz (a few MB)
```

- Pod: image `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`, an Ampere GPU
  (the script builds for sm_86: A4000-A6000, RTX 3090, A40), ~20 GB volume.
  `pod_start.sh` runs it unattended (`RUN_SCRIPT=coda_pvrcnn_orig.sh`) and
  serves only the log and the tarball, read-only, on port 8000.
- Most of the time goes to building the CUDA operators and fetching the scans
  from the CODa server (about 170 MB per window); inference is about 70 s per
  window on an A40.

## Result (October 2026): the second stage does not help tracking

First run, upstream OpenPCDet (`coda_pvrcnn.sh`): all 365 checkpoint tensors
load, but the RoI head scored real pedestrians low (median 0.105) and the
two-stage output kept only 0.34 recall at score 0.3. That was the port, not
the checkpoint: in the original environment (`coda_pvrcnn_orig.sh`: the
coda-models fork at c9a12d74, PyTorch 1.10 / CUDA 11.3, spconv-cu113) the
same checkpoint scores matching pedestrian RoIs at a median 0.75 (0.12 for
RoIs that match nobody). Use `coda_pvrcnn_orig.sh`; `coda_pvrcnn.sh` is kept
only as the record of that difference.

Development windows coda20_8640 and coda13_9700 (320 scans), final
pedestrian detections within 0.5 m of a CODa label:

| | score | recall | precision | mean BEV error |
| --- | --- | --- | --- | --- |
| first stage, CPU port (`detect_lidar_cpu.py`) | 0.4 | 0.773 | 0.784 | 0.130 m |
| full two-stage, original environment | 0.4 | 0.752 | 0.779 | 0.123 m |
| first stage, CPU port | 0.5 | 0.762 | 0.846 | 0.128 m |
| full two-stage, original environment | 0.5 | 0.718 | 0.806 | 0.120 m |

Tracking on the same two windows, same conversion and flags
(`docs/evaluation.md`, HOTA / IDF1):

| detections | ours + FGO | AB3DMOT-style |
| --- | --- | --- |
| first stage, CPU, score >= 0.4 | 0.552 / 0.658 | 0.566 / 0.665 |
| two-stage, score >= 0.3 | 0.530 / 0.645 | 0.551 / 0.663 |
| two-stage, score >= 0.4 | 0.530 / 0.647 | 0.549 / 0.656 |
| two-stage, score >= 0.5 | 0.526 / 0.646 | 0.544 / 0.651 |

The refinement moves boxes about 1 cm closer but loses recall, so tracking is
slightly worse at every threshold. Rescoring the first-stage RoIs with the
second-stage score raises RoI-level average precision only a little (0.780
alone, 0.799 with roi * sqrt(rcnn)). The CPU first stage stays the detector of
record; a better detector would have to be trained, not taken from this
checkpoint.
