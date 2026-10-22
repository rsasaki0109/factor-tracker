#!/usr/bin/env bash
# The CODa PV-RCNN checkpoint in its original environment: coda-models as
# published (pcdet 0.5 fork) on PyTorch 1.10 + CUDA 11.3, for comparison with
# coda_pvrcnn.sh (upstream OpenPCDet). Same detector script and outputs.
#
# Pod image and GPU: see setup_coda_models.sh.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/workspace}
CODA=$WORK/CODa
CKPT=$WORK/coda128_allclass_bestoracle.pth
CFG=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_headfull.yaml
WINDOWS=${WINDOWS:-"20:8640 13:9700"}

WORK=$WORK bash "$repo/scripts/runpod/setup_coda_models.sh"

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
echo "done: $WORK/coda_pvrcnn_dets.tar.gz"
