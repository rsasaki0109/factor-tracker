#!/usr/bin/env bash
# End-to-end KITTI tracking run on a RunPod GPU pod:
#   detectors (GPU) -> conversion -> engine evaluation (CPU, if cargo exists).
#
# usage: PCDET_CKPT=/workspace/ckpt/pointpillar_7728.pth bash scripts/runpod/run_all.sh [sequence ...]
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/workspace}
KITTI_ROOT=${KITTI_ROOT:-$WORK/kitti}
OUT=${OUT:-$WORK/pe-out}
PCDET_CFG=${PCDET_CFG:-cfgs/kitti_models/pointpillar.yaml}
YOLO_MODEL=${YOLO_MODEL:-yolo26x.pt}
: "${PCDET_CKPT:?set PCDET_CKPT to an OpenPCDet KITTI checkpoint}"

seq_args=()
[[ $# -gt 0 ]] && seq_args=(--sequences "$@")

(cd "$WORK/OpenPCDet/tools" && python "$repo/scripts/runpod/detect_lidar_pcdet.py" \
  --kitti-root "$KITTI_ROOT" --out "$OUT/detections" --cfg "$PCDET_CFG" --ckpt "$PCDET_CKPT" "${seq_args[@]}")
python "$repo/scripts/runpod/detect_camera_yolo.py" \
  --kitti-root "$KITTI_ROOT" --out "$OUT/detections" --model "$YOLO_MODEL" "${seq_args[@]}"

python "$repo/scripts/kitti/convert_tracking.py" \
  --kitti-root "$KITTI_ROOT" --out "$OUT/sequences" --detections-root "$OUT/detections" "${seq_args[@]}"
tar czf "$OUT/detections.tar.gz" -C "$OUT" detections
echo "detections archived to $OUT/detections.tar.gz"

if command -v cargo > /dev/null; then
  "$repo/scripts/kitti/evaluate_sequences.sh" "$OUT/sequences" "$OUT/results" "$@"
else
  echo "cargo not found: copy $OUT/sequences and run scripts/kitti/evaluate_sequences.sh locally"
fi
