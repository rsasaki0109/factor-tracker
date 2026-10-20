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
`coda_pvrcnn.sh` runs the whole model (upstream OpenPCDet with the
coda-models config) over the nine CODa evaluation windows (five used for
tuning, four held out) and packs the detections:

```sh
cd /workspace
git clone -b claude/dreamy-noether-2h4v4t https://github.com/rsasaki0109/perception-engine.git
bash perception-engine/scripts/runpod/coda_pvrcnn.sh
# -> /workspace/coda_pvrcnn_dets.tar.gz (a few MB)
```

- Pod: RunPod PyTorch template (CUDA 11.8 or 12.x), any 16 GB+ GPU, ~20 GB
  volume. Most of the time goes to fetching the scans from the CODa server
  (about 1.5 GB for nine windows); inference is a few minutes.
- Untested on a pod so far: the first run may need small fixes (OpenPCDet
  version, spconv wheel). The log of a failed run is enough to fix it.

## Result (October 2026): the second stage does not help

Run on RunPod (RTX 4090 / A4000, ~50 s per 160-scan window). Pedestrian
detections within 0.5 m of a CODa label, five development windows:

| | recall at score 0.3 | precision |
| --- | --- | --- |
| first stage, CPU port (`detect_lidar_cpu.py`, with NMS) | 0.82 | 0.75 |
| first stage on GPU (CenterHead RoIs, before final NMS) | 0.81 | 0.44 |
| full two-stage output | 0.34 | 0.68 |

All 365 checkpoint tensors load, but the RoI head scores real pedestrians low
(median 0.105, against 0.03 for RoIs that match nobody), so the final score
threshold removes most of them. Rescoring the first-stage RoIs with the
second-stage score lowers average precision (0.780 alone, 0.624-0.730
combined). Raising the RoI limit (`--set`) changes nothing: the RoIs come from
the CenterHead (up to 500). Whether this is the checkpoint itself or a
difference between upstream OpenPCDet and the coda-models fork (the
checkpoint was trained with pcdet 0.5) is not settled; the CPU first stage
stays the detector of record.
