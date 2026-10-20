#!/usr/bin/env bash
# Full two-stage CODa PV-RCNN detections for the CODa evaluation windows, on a
# RunPod GPU pod. Produces $WORK/coda_pvrcnn_dets.tar.gz (a few MB): copy it
# back and convert with scripts/coda/convert_coda.py --lidar-detections.
#
# usage (inside the pod, PyTorch template):
#   cd /workspace && git clone -b <branch> https://github.com/rsasaki0109/perception-engine.git
#   bash perception-engine/scripts/runpod/coda_pvrcnn.sh
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/workspace}
CODA=$WORK/CODa
CKPT=$WORK/coda128_allclass_bestoracle.pth
CFG=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_headfull.yaml
# Development windows (used for tuning) and held-out windows (scored once).
WINDOWS=${WINDOWS:-"13:9700 20:12700 20:4820 20:8640 20:9330 4:6870 9:8515 3:1800 11:5220"}

WORK=$WORK bash "$repo/scripts/runpod/setup.sh"
pip install -q shapely pandas
[[ -d $WORK/coda-models ]] || git clone -q --depth 1 https://github.com/ut-amrl/coda-models.git "$WORK/coda-models"
[[ -f $CKPT ]] || curl -sSL -o "$CKPT" \
  https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/pretrained_models/128channel/coda128_allclass_bestoracle.pth

# One window at a time (fetch, then detect), so a problem shows after the first.
for w in $WINDOWS; do
  seq=${w%%:*}; first=${w#*:}; last=$((first + 159))
  for try in 1 2 3 4 5; do
    python "$repo/scripts/coda/fetch_coda.py" --coda-root "$CODA" --sequence "$seq" --first "$first" --last "$last" \
      --lidar-only && break
    echo "fetch failed, retrying in 30 s"; sleep 30
  done
  (cd "$WORK/coda-models/tools" && python "$repo/scripts/runpod/detect_coda_pvrcnn.py" --cfg "$CFG" --ckpt "$CKPT" \
    --coda-root "$CODA" --out "$WORK/coda_pvrcnn" --window "$seq" "$first" "$last")
done
tar czf "$WORK/coda_pvrcnn_dets.tar.gz" -C "$WORK" coda_pvrcnn
echo "done: $WORK/coda_pvrcnn_dets.tar.gz ($(du -h "$WORK/coda_pvrcnn_dets.tar.gz" | cut -f1))"
