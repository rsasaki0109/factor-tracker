# Training a CODa pedestrian detector (handoff)

Status as of October 2026, for whoever picks this up next (on a laptop GPU).
Everything below is on branch `claude/dreamy-noether-2h4v4t`.

## Why

On the CODa evaluation windows the detector, not the tracker, now limits the
scores (`docs/evaluation.md`, "Standard metrics against baseline trackers"):

| on the same CODa dev windows | HOTA |
| --- | --- |
| engine today (CPU first stage, distance association) | 0.615 |
| perfect association on the same detections | 0.659 |
| perfect detections (truth boxes as detections) | 0.836 |

Things already tried that did not move it: the published checkpoint's second
stage (run in its original environment it tracks slightly worse than the
first stage; `scripts/runpod/README.md`), ByteTrack-style low-score
detections, FGO re-association on top of distance association, camera ReID and
a geometry-only LiDAR shape descriptor (`ROADMAP.md`).

## Success criterion

Write the new detector's boxes for the eight evaluation windows and run

```sh
cargo build --release -p factor-tracker
PY=python3 scripts/eval/coda_tracker_eval.sh <detections-root> <coda-root> 0.4 <work-dir>
```

It scores pedestrians only (detections and truth) and prints, per split, the
engine with distance association, the engine with the older Mahalanobis
association, an AB3DMOT-style baseline, and pedestrian detection recall /
precision. With the current detector (CPU first stage of
`coda128_allclass_bestoracle.pth`, score >= 0.4) it prints:

| split | engine (distance) HOTA / IDF1 | AB3DMOT-style HOTA / IDF1 | recall / precision (score >= 0.4) |
| --- | --- | --- | --- |
| dev (5 windows) | 0.618 / 0.760 | 0.618 / 0.764 | 0.801 / 0.804 |
| held-out (3 windows) | 0.768 / 0.900 | 0.764 / 0.887 | 0.890 / 0.916 |

(Before October 21 the script also kept the other CODa dynamic classes, in
detections and truth; that cost the published detector, which also reports
scooters and cyclists, about 0.06 held-out HOTA against a pedestrian-only
detector. `docs/evaluation.md` compares trackers on those mixed-class inputs,
which is fair between trackers but not between detectors.)

A new detector is better if it beats the engine row on both splits. Try a
few score thresholds (third argument) and use the one that is best on dev.

`<detections-root>` holds one directory per window, `coda<seq>_<first>/`, with
`3d_bbox_os1_<seq>_<frame>.json` per scan in the CODa label format plus a
`score` field: `{"3dbbox": [{"classId": "Pedestrian", "score": 0.9, "cX": ..,
"cY": .., "cZ": .., "l": .., "w": .., "h": .., "r": 0, "p": 0, "y": ..}, ...]}`
in the LiDAR (os1) frame, as `scripts/coda/detect_lidar_cpu.py` and
`scripts/runpod/detect_coda_pvrcnn.py` write it. Only `classId == "Pedestrian"`
is scored.

## Training on RunPod (October 2026)

`scripts/runpod/coda_train_ped.sh` does everything on one pod: installs the
coda-models environment (`setup_coda_models.sh`), fetches every third
labelled scan outside the evaluation windows
(`scripts/coda/fetch_coda_training.py`: 8513 scans, 11.8 GB, about 30 min),
trains `scripts/runpod/cfgs/coda_ped_centerpoint.yaml` with
`train_coda_ped.py`, and detects on the eight evaluation windows with the new
model and with the published PV-RCNN. Start it with `pod_start.sh` and
`RUN_SCRIPT=coda_train_ped.sh MODE=smoke|full RESULT=coda_train_ped.tar.gz`
on image `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel` (Python 3.7) with a
60 GB volume. `MODE=smoke` (300 scans, 200 iterations, one window) checks
every step in about 10 minutes.

Measured on an A40: 0.33 s per iteration at batch 4 (2.4 GB of GPU memory),
11.5 min per epoch of 8513 scans, so 20 epochs take about 3.8 h; the whole run
is about 4.5 h, about $2.7 at $0.59/h.

### Results: no detector beats the published first stage

Two runs on the same 8513 scans (every third labelled scan, Pedestrian
only, evaluation windows +-100 scans left out):

- from scratch: 20 epochs, final loss 1.8, about 4.8 h of pod time;
- fine-tuned from the published first stage (`INIT=published`,
  `init_from_published.py`; the heatmap keeps the Pedestrian channel):
  6 epochs at LR 0.001, final loss 1.4, about 2.5 h. The published weights
  have seen most evaluation scans, so this one is not a clean test.

Pedestrian tracking (`coda_tracker_eval.sh`, engine with distance
association; score threshold chosen on dev, ties broken by IDF1):

| detections | dev HOTA / IDF1 (threshold) | held-out HOTA / IDF1 (same threshold) |
| --- | --- | --- |
| published, CPU first stage (baseline) | 0.618 / 0.761 (0.5) | 0.769 / 0.900 |
| published, two-stage on GPU (point shift fixed) | 0.618 / 0.761 (0.5) | 0.744 / 0.877 |
| trained from scratch | 0.551 / 0.670 (0.4) | 0.717 / 0.835 |
| fine-tuned from the published weights | 0.615 / 0.754 (0.5) | 0.776 / 0.907 |

The fine-tuned model ties with the baseline (-0.003 dev, +0.007 held-out,
within what separates thresholds); it is pedestrian-only and covers +-51.2 m
instead of +-75.2 m, so it is faster for the same accuracy.
`scripts/coda/detect_lidar_cpu.py` runs it (pass its checkpoint as `--ckpt`;
the model is recognised from the number of classes): about 1.5 s per scan
on one core against 2.5 s for the published model, with boxes within 5 mm
and scores within 0.007 of OpenPCDet on the GPU and the same tracking scores
on all eight windows. The checkpoint (CC BY-NC-SA 4.0, as CODa) is on
Hugging Face, with a model card:

```sh
wget https://huggingface.co/rsasaki0109/coda-ped-centerpoint/resolve/main/coda_ped_centerpoint_ft.pth
# sha256 8fe8ac00fcde6c28165834b0a4a919f53683ed8911e91ae2e1bd2e912072463b
python3 scripts/coda/detect_lidar_cpu.py --ckpt coda_ped_centerpoint_ft.pth \
    --coda-root CODa --sequence 3 --first 1800 --last 1959 --out det/coda3_1800
```

To rebuild it, run `coda_train_ped.sh` with `INIT=published`, `EPOCHS=6` and
`LR=0.001` (about $2, 2.5 h on an A40). Total pod cost of both runs
with their smoke tests: about $4.7.

Pedestrian detection on the scans of CODa's own *testing* split inside the
windows (scans no model was trained on), score >= 0.4:

| detections | dev recall / precision | held-out recall / precision |
| --- | --- | --- |
| published, CPU first stage | 0.804 / 0.816 | 0.892 / 0.915 |
| trained from scratch | 0.721 / 0.783 | 0.863 / 0.852 |

The published model scores the same on CODa's training-split and
testing-split scans of these windows, so its having seen most of them
inflates its numbers little. The model trained from scratch is genuinely
weaker, mostly in the crowded dev windows (recall 0.72 against 0.80). The
published model was fine-tuned from a Waymo-trained model on all of CODa
(every labelled scan, all classes, 30 epochs). What is left to try: every
labelled scan instead of every third (about 3x the data and cost) and more
epochs, from scratch or fine-tuned.

## The evaluation windows (keep them out of training)

| split | sequence | scans |
| --- | --- | --- |
| dev | 13 | 9700-9860 |
| dev | 20 | 12700-12860, 4820-4980, 8640-8800, 9330-9490 |
| held-out | 3 | 1800-1959 |
| held-out | 4 | 6870-7029 |
| held-out | 9 | 8515-8674 |

CODa splits frames at random within each sequence, so 70-75 % of the scans in
every window are in CODa's own `training` split (e.g. sequence 20,
8640-8800: 115 training, 20 validation, 25 testing). The published checkpoint
was trained on that split, so **the baseline above has seen most of the
evaluation scans**: its numbers are optimistic.

For a clean comparison, drop every labelled frame within about 10 s (100
scans) of an evaluation window from the training set, i.e. exclude
`[first - 100, last + 100]` of each window above; the rest of those
sequences can stay. A detector trained this way that still matches the
baseline is genuinely better on new data. If time is short, fine-tuning the
published checkpoint is faster but keeps the leak; report which one was done.

## Recommended model

Keep the architecture of the current detector's first stage (OpenPCDet
`MeanVFE -> VoxelBackBone8x -> HeightCompression -> BaseBEVBackbone ->
CenterHead`, voxel 0.1 x 0.1 x 0.15 m, range +-75.2 m, x/y/z only; the first
half of `tools/cfgs/da-coda-coda_models/waymocenterhead/
pvrcnn_allclass128full_finetune_headfull.yaml` in
https://github.com/ut-amrl/coda-models, without the
`VoxelSetAbstraction`, `PointHeadSimple` and `PVRCNNHead` parts). Reasons:

- `scripts/coda/detect_lidar_cpu.py` runs exactly this on a CPU without CUDA,
  spconv or OpenPCDet, so the trained weights can ship in the CPU product path.
  It hard-codes the 56-class list and the config constants at the top of the
  file; a model with fewer classes (e.g. Pedestrian only, or a few dynamic
  classes) needs `CLASS_NAMES`, the CenterHead class grouping and `TO_CODA`
  updated there.
- The second stage did not help tracking with the published weights.

Pedestrian-only (or Pedestrian plus a few dynamic classes) cuts training time
and memory; the tracker uses only pedestrians.

## Data

- CODa: https://github.com/ut-amrl/coda-devkit (CC BY-NC-SA 4.0 plus dataset
  terms; read `docs/DATA_REPORT.md` there). Do not commit CODa or anything
  derived from it to this repository.
- 23 sequences; about 32 k scans carry 3D labels (sequence 20: 6000 of 14 k
  scans, 13: 4176, 9: 1165, 1: 1000, 0: 850, 11: 800, 4: 580, 3: 500, 5: 200,
  22: none). The label lists and CODa's own splits are in
  `metadata/<seq>.json` (`ObjectTracking.training / validation / testing`).
- One ego-compensated 128-beam scan (`3d_comp/os1/<seq>/*.bin`, float32
  x y z intensity) is about 2 MB, so all labelled scans are about 65 GB.
- `scripts/coda/fetch_coda.py --lidar-only` pulls a frame range of one
  sequence (scans, labels, calibrations, poses, timestamps) out of the
  40-110 GB per-sequence ZIPs with HTTP range requests. It fetches every scan
  in the range; for training, fetching only scans that have a `3d_bbox` file
  would save space (not implemented yet).
- coda-models converts CODa to its training format with
  `python tools/create_data.py coda` and
  `python -m pcdet.datasets.coda.coda_dataset create_coda_infos <dataset cfg>`
  (`docs/GETTING_STARTED.md` there). Filter the evaluation windows out of the
  training infos after that step.

## Environment pitfalls (from the RunPod runs)

- The published checkpoint behaves correctly only in the coda-models fork
  (pcdet 0.5) with PyTorch 1.10 / CUDA 11.3 / spconv-cu113: upstream OpenPCDet
  loads all its tensors but the second stage scores people wrongly.
  `scripts/runpod/coda_pvrcnn_orig.sh` is a working install recipe
  (pinned commit c9a12d74, `numpy<1.22`, `TORCH_CUDA_ARCH_LIST=8.6`).
- CUDA 11.3 builds up to sm_86 (RTX 30xx / A-series). An RTX 40xx laptop GPU
  (sm_89) needs CUDA 11.8 or later, so either try the coda-models build with
  `TORCH_CUDA_ARCH_LIST="8.6+PTX"` on a newer PyTorch, or train in upstream
  OpenPCDet (newer PyTorch) with coda-models' `pcdet/datasets/coda` dataset
  class copied in. Training from scratch does not have the checkpoint
  compatibility problem; only loading the published weights does.
- The published config trains with batch 4 per 24 GB GPU for 30 epochs; on a
  laptop GPU (8-12 GB) expect batch 1-2.

## Producing the evaluation detections

- With the same architecture: `scripts/coda/detect_lidar_cpu.py --ckpt
  <new.pth> --coda-root <CODa> --sequence <seq> --first <first> --last <last>
  --out <detections-root>/coda<seq>_<first>` (CPU, about 20 s per scan; split
  each window over several processes as `scripts/coda/run_window.sh` does).
- On a GPU with OpenPCDet installed: `scripts/runpod/detect_coda_pvrcnn.py`
  runs any OpenPCDet config and checkpoint and writes the same files (its
  RoI-diagnostic dump assumes a two-stage model; drop it for a one-stage
  model).
- Fetch the windows first: `scripts/coda/fetch_coda.py --coda-root <CODa>
  --sequence <seq> --first <first> --last <last> --lidar-only`.
