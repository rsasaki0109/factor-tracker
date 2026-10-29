#!/usr/bin/env bash
# KITTI 3D MOT benchmark in AB3DMOT's protocol (pedestrians, the PointRCNN
# detections AB3DMOT ships, its evaluator at 3D IoU 0.25): converts the
# validation sequences, replays factor-tracker and writes one result
# directory for AB3DMOT's evaluator. docs/evaluation.md ("KITTI 3D MOT").
#
# usage: scripts/kitti/ab3dmot_benchmark.sh <kitti-root> <ab3dmot-root> <work> [name]
#   <kitti-root>    KITTI tracking data as convert_tracking.py reads it
#   <ab3dmot-root>  a checkout of https://github.com/xinshuoweng/AB3DMOT
#                   (its data/KITTI/detection and scripts/KITTI/evaluate.py)
#   <work>          converted sequences and engine outputs
#   [name]          result name (default factor_tracker): results go to
#                   <ab3dmot-root>/results/KITTI/<name>/data_0
#
# Then, in <ab3dmot-root>, with AB3DMOT's Python dependencies:
#   python3 scripts/KITTI/evaluate.py <name> 1 3D 0.25
# Its seqmap (scripts/KITTI/evaluate_tracking.seqmap.val) selects the
# sequences scored; the tune/test halves of docs/evaluation.md are subsets.
#
# REPLAY_ARGS overrides the engine settings below (the ones reported).
set -euo pipefail
REPLAY_ARGS=${REPLAY_ARGS:---initial-velocity-variance 1 --accel-spectral-density 0.5 \
--max-association-distance-m 1.0 --recent-track-window-ns 0 --min-birth-distance-m 0 \
--new-track-score-threshold 0 --report-min-confidence 0.73}

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <kitti-root> <ab3dmot-root> <work> [name]" >&2
  exit 2
fi
kitti=$1
ab3dmot=$2
work=$3
name=${4:-factor_tracker}
sequences=(0001 0006 0008 0010 0012 0013 0014 0015 0016 0018 0019)

repo=$(cd "$(dirname "$0")/../.." && pwd)
cargo build --release -q -p factor-tracker --manifest-path "$repo/Cargo.toml"
cli="$repo/target/release/factor-tracker"

python3 "$repo/scripts/kitti/ab3dmot_val.py" detections "$ab3dmot" "$work/det" --category Pedestrian
python3 "$repo/scripts/kitti/convert_tracking.py" --kitti-root "$kitti" --out "$work/seq" \
  --sequences "${sequences[@]}" --detections-root "$work/det"
out="$ab3dmot/results/KITTI/$name/data_0"
mkdir -p "$out"
for seq in "${sequences[@]}"; do
  d="$work/seq/$seq"
  read -r w h < "$d/image_size.txt"
  # detection position standard deviation 0.1 m
  "$cli" prepare --calib "$d/calib.txt" --times "$d/times.txt" --poses "$d/poses.txt" \
    --lidar-detections "$d/detections_lidar" --camera-detections "$d/detections_camera" \
    --lidar-labels "$d/labels_lidar" --image-width "$w" --image-height "$h" --default-covariance 0.01 \
    --out-setup "$d/setup.json" --out-inputs "$d/inputs.jsonl" --out-truth "$d/truth.jsonl" > /dev/null
  # shellcheck disable=SC2086
  "$cli" replay --setup "$d/setup.json" --inputs "$d/inputs.jsonl" --mode lidar_only \
    --output "$d/engine.jsonl" --diagnostics /dev/null $REPLAY_ARGS > /dev/null
  # AB3DMOT's output protocol: a track from its first detection, and for up
  # to three missed scans
  python3 "$repo/scripts/kitti/ab3dmot_val.py" results "$d" "$d/engine.jsonl" "$out/$seq.txt" \
    --include-tentative --coast-scans 3
done
