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
`/workspace/pe-out`), `PCDET_CFG`, `YOLO_MODEL` (default `yolo11x.pt`).

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
