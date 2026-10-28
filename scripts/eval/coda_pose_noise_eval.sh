#!/usr/bin/env bash
# Robustness to ego-pose noise: takes the window dirs of a
# scripts/eval/coda_tracker_eval.sh run, adds pose noise to each
# (scripts/eval/pose_noise.py: yaw and translation jitter per scan) and runs
# the birth-suppressed engine with and without --pose-correction and the
# AB3DMOT-style baseline on the noisy windows. Prints mean HOTA / DetA /
# AssA / IDF1 / ID switches per split.
#
# usage: scripts/eval/coda_pose_noise_eval.sh <coda_tracker_eval work dir> [yaw-deg] [translation-m] [out-dir]
# env: PY  python with numpy and scipy (default python3)
set -euo pipefail
src=$1; yaw=${2:-1.0}; shift_m=${3:-0.1}; work=${4:-$(mktemp -d)}
repo=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-python3}; cli=$repo/target/release/factor-tracker
cargo build --release -q -p factor-tracker --manifest-path "$repo/Cargo.toml"
prior="--initial-velocity-variance 1 --accel-spectral-density 0.5 --report-coast-ms 0"
birth="$prior --max-association-distance-m 1.0 --recent-track-window-ns 0 --min-birth-distance-m 0.8"
mean() { "$PY" -c "
import json, sys
ms = [json.loads(l) for l in open(sys.argv[2])]
print(f'{sys.argv[1]:20s}', ' '.join(f'{k} {sum(m[k] for m in ms) / len(ms):.3f}' for k in ['HOTA', 'DetA', 'AssA', 'IDF1']),
      'IDSW', int(sum(m['IDSW'] for m in ms)))" "$1" "$2"; }
for split in dev heldout; do
  [[ $split == dev ]] && windows="coda13_9700/13 coda20_12700/20 coda20_4820/20 coda20_8640/20 coda20_9330/20" \
                      || windows="coda3_1800/3 coda4_6870/4 coda9_8515/9"
  echo "== $split (pose noise: yaw $yaw deg, translation $shift_m m)"
  rm -f "$work"/$split.*.txt
  for w in $windows; do
    d=$work/${w%/*}
    "$PY" "$repo/scripts/eval/pose_noise.py" "$src/$w" "$d" --yaw-deg "$yaw" --translation-m "$shift_m"
    run() { $cli replay --setup "$d/setup.json" --inputs "$d/inputs.jsonl" --mode lidar_only --output "$d/$1.jsonl" \
              --diagnostics /dev/null "${@:2}" > /dev/null; }
    run engine_birth $birth
    run engine_pose $birth --pose-correction
    run engine_mahalanobis $prior --gate-squared-mahalanobis 5.99
    "$PY" "$repo/scripts/eval/baselines.py" ab3dmot "$d/engine_mahalanobis.jsonl" "$d/ab3dmot.jsonl" --gate 0.8 --max-age 8 --min-hits 2
    for config in engine_birth engine_pose ab3dmot; do
      "$PY" "$repo/scripts/eval/mot_metrics.py" "$d/$config.jsonl" "$d/truth.jsonl" --json >> "$work/$split.$config.txt"
    done
  done
  for config in engine_birth engine_pose ab3dmot; do mean "$config" "$work/$split.$config.txt"; done
done
echo "work dir: $work"
