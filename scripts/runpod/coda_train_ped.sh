#!/usr/bin/env bash
# Trains the CODa pedestrian detector (cfgs/coda_ped_centerpoint.yaml) on a
# RunPod pod, then detects on the eight tracking evaluation windows with it
# and with the published two-stage PV-RCNN, and packs checkpoint, log and
# detections into $WORK/coda_train_ped.tar.gz for
# scripts/eval/coda_tracker_eval.sh.
#
# MODE=smoke  300 scans, 200 iterations, one window: checks every step and
#             measures the time per iteration.
# MODE=full   every STRIDE-th labelled scan (default 3), EPOCHS (20), BATCH (4).
# INIT=published  fine-tune from the published PV-RCNN's first stage instead of
#             training from scratch (set LR, e.g. 0.001, and fewer EPOCHS).
# DETECT_PVRCNN=0 skips detection with the published two-stage model.
#
# Pod image and GPU: see setup_coda_models.sh; about 25 GB of scans in full mode.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/workspace}
CODA=$WORK/CODa
MODE=${MODE:-smoke}
STRIDE=${STRIDE:-3}; EPOCHS=${EPOCHS:-20}; BATCH=${BATCH:-4}
INIT=${INIT:-}; LR=${LR:-}; DETECT_PVRCNN=${DETECT_PVRCNN:-1}
PED_CFG=$repo/scripts/runpod/cfgs/coda_ped_centerpoint.yaml
PVRCNN_CFG=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_headfull.yaml
WINDOWS="13:9700:9860 20:12700:12860 20:4820:4980 20:8640:8800 20:9330:9490 3:1800:1959 4:6870:7029 9:8515:8674"

WORK=$WORK bash "$repo/scripts/runpod/setup_coda_models.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

limit=(); train=(--epochs "$EPOCHS" --batch "$BATCH")
if [[ $INIT == published ]]; then train+=(--init-ckpt "$WORK/coda128_allclass_bestoracle.pth"); fi
if [[ -n $LR ]]; then train+=(--set OPTIMIZATION.LR "$LR"); fi
if [[ $MODE == smoke ]]; then
  limit=(--limit 300); train+=(--max-iters 200); WINDOWS="20:8640:8800"
fi
echo "== fetch training scans $(date +%T)"
python "$repo/scripts/coda/fetch_coda_training.py" --coda-root "$CODA" --stride "$STRIDE" "${limit[@]}"
echo "== train $(date +%T)"
cd "$WORK/coda-models/tools"
python "$repo/scripts/runpod/train_coda_ped.py" --cfg "$PED_CFG" --coda-root "$CODA" --out "$WORK/train" "${train[@]}"
ckpt=$(ls -v "$WORK"/train/checkpoint_epoch_*.pth | tail -1)
echo "== detect with $ckpt $(date +%T)"
for w in $WINDOWS; do
  IFS=: read -r seq first last <<< "$w"
  for try in 1 2 3 4 5; do
    python "$repo/scripts/coda/fetch_coda.py" --coda-root "$CODA" --sequence "$seq" --first "$first" --last "$last" \
      --lidar-only && break
    echo "fetch failed, retrying in 30 s"; sleep 30
  done
  python "$repo/scripts/runpod/detect_coda_pvrcnn.py" --cfg "$PED_CFG" --ckpt "$ckpt" --coda-root "$CODA" \
    --out "$WORK/dets_ped" --window "$seq" "$first" "$last"
  if [[ $DETECT_PVRCNN != 1 ]]; then continue; fi
  python "$repo/scripts/runpod/detect_coda_pvrcnn.py" --cfg "$PVRCNN_CFG" --ckpt "$WORK/coda128_allclass_bestoracle.pth" \
    --coda-root "$CODA" --out "$WORK/dets_pvrcnn" --window "$seq" "$first" "$last"
done
mkdir -p "$WORK/result" "$WORK/dets_pvrcnn"
find "$WORK/dets_pvrcnn" -name "rois_*.json" -delete
cp "$WORK/train/train.log" "$ckpt" "$WORK/result/"
mv "$WORK/dets_ped" "$WORK/dets_pvrcnn" "$WORK/result/"
tar czf "$WORK/coda_train_ped.tar.gz" -C "$WORK" result
echo "done: $WORK/coda_train_ped.tar.gz ($(du -h "$WORK/coda_train_ped.tar.gz" | cut -f1)) $(date +%T)"
