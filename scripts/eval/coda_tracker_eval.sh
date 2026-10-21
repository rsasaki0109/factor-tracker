#!/usr/bin/env bash
# Scores a CODa LiDAR detector by the tracking it gives: converts its
# detections for the eight CODa evaluation windows (five development windows,
# three held-out), runs the engine and an AB3DMOT-style baseline on them and
# prints mean HOTA / DetA / AssA / MOTA / IDF1 per split, plus pedestrian
# detection recall and precision.
#
# usage: scripts/eval/coda_tracker_eval.sh <detections-root> <coda-root> [score-threshold] [work-dir]
#   detections-root  one directory per window, coda<seq>_<first>/, holding
#                    3d_bbox_os1_<seq>_<frame>.json (CODa boxes plus "score",
#                    as written by scripts/coda/detect_lidar_cpu.py)
#   coda-root        CODa with calibrations, timestamps, poses and 3d_bbox for
#                    the windows (scripts/coda/fetch_coda.py --lidar-only)
# env: PY  python with numpy and scipy (default python3)
#
# The development windows have 161 scans and the held-out ones 160, as in
# docs/evaluation.md; the numbers there are reproduced with the CPU first stage.
set -euo pipefail
dets=$1; coda=$2; th=${3:-0.4}; work=${4:-$(mktemp -d)}
mkdir -p "$work"
repo=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-python3}; cli=$repo/target/release/factor-tracker
cargo build --release -q -p factor-tracker --manifest-path "$repo/Cargo.toml"
DEV="13:9700:9860 20:12700:12860 20:4820:4980 20:8640:8800 20:9330:9490"
HELDOUT="3:1800:1959 4:6870:7029 9:8515:8674"
prior="--initial-velocity-variance 1 --accel-spectral-density 0.5 --report-coast-ms 0"
declare -A CONFIGS=(
  [engine_distance]="$prior --max-association-distance-m 1.0 --recent-track-window-ns 0"
  [engine_mahalanobis]="$prior --gate-squared-mahalanobis 5.99"
)
mean() { "$PY" -c "
import json, sys
ms = [json.loads(l) for l in open(sys.argv[2])]
print(f'{sys.argv[1]:20s}', ' '.join(f'{k} {sum(m[k] for m in ms) / len(ms):.3f}' for k in ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1']),
      'IDSW', int(sum(m['IDSW'] for m in ms)))" "$1" "$2"; }
for split in dev heldout; do
  [[ $split == dev ]] && windows=$DEV || windows=$HELDOUT
  echo "== $split (score >= $th)"
  rm -f "$work"/$split.*.txt
  for w in $windows; do
    IFS=: read -r seq first last <<< "$w"; name=coda${seq}_${first}; d=$work/$name/$seq
    [[ -d $dets/$name ]] || { echo "missing $dets/$name" >&2; exit 1; }
    python3 "$repo/scripts/coda/convert_coda.py" --coda-root "$coda" --sequence "$seq" --first "$first" --last "$last" \
      --lidar-detections "$dets/$name" --lidar-score-threshold "$th" --out "$d" > "$work/$name.convert.log"
    for config in "${!CONFIGS[@]}"; do
      $cli replay --setup "$d/setup.json" --inputs "$d/inputs.jsonl" --mode lidar_only --output "$d/$config.jsonl" \
        --diagnostics /dev/null ${CONFIGS[$config]} > /dev/null
      "$PY" "$repo/scripts/eval/mot_metrics.py" "$d/$config.jsonl" "$d/truth.jsonl" --json >> "$work/$split.$config.txt"
    done
    "$PY" "$repo/scripts/eval/baselines.py" ab3dmot "$d/engine_mahalanobis.jsonl" "$d/ab3dmot.jsonl" --gate 0.8 --max-age 8 --min-hits 2
    "$PY" "$repo/scripts/eval/mot_metrics.py" "$d/ab3dmot.jsonl" "$d/truth.jsonl" --json >> "$work/$split.ab3dmot.txt"
  done
  for config in engine_distance engine_mahalanobis ab3dmot; do mean "$config" "$work/$split.$config.txt"; done
  "$PY" "$repo/scripts/eval/coda_detections.py" "$dets" "$coda" $windows
done
echo "work dir: $work"
