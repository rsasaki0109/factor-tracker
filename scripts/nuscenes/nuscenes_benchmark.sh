#!/usr/bin/env bash
# nuScenes tracking benchmark, pedestrians: the MEGVII detections of the
# validation split, factor-tracker on every scene, one result file in the
# nuScenes tracking format. docs/evaluation.md ("nuScenes").
#
# usage: scripts/nuscenes/nuscenes_benchmark.sh <dataroot> <megvii_val.json> <work> [out.json]
#   <dataroot>         holds v1.0-trainval/ (the metadata tables, v1.0-trainval_meta.tgz)
#   <megvii_val.json>  from https://www.nuscenes.org/data/detection-megvii.zip
#   <work>             one directory per scene (inputs and engine outputs)
# needs python3 with nuscenes-devkit (for the split's scene list).
#
# Then score with the official devkit (TrackingEval, tracking_nips_2019) and
# read the pedestrian row; the devkit needs <dataroot>/maps/*.png to exist
# (empty files will do, the tracking evaluation does not read them).
#
# REPLAY_ARGS overrides the engine settings below (chosen on the tune half,
# scored once on the test half).
set -euo pipefail
REPLAY_ARGS=${REPLAY_ARGS:---initial-velocity-variance 1 --accel-spectral-density 0.5 \
--max-association-distance-m 1.5 --recent-track-window-ns 0 --new-track-score-threshold 0 \
--velocity-std 0.5}

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <dataroot> <megvii_val.json> <work> [out.json]" >&2
  exit 2
fi
dataroot=$1
detections=$2
work=$3
out=${4:-$work/results_val_factor_tracker.json}

repo=$(cd "$(dirname "$0")/../.." && pwd)
cargo build --release -q -p factor-tracker --manifest-path "$repo/Cargo.toml"
cli="$repo/target/release/factor-tracker"

python3 "$repo/scripts/nuscenes/nuscenes_tracking.py" inputs "$dataroot" "$detections" "$work"
for d in "$work"/*/; do
  # shellcheck disable=SC2086
  "$cli" replay --setup "$d/setup.json" --inputs "$d/inputs.jsonl" --mode lidar_only \
    --output "$d/engine.jsonl" --diagnostics /dev/null $REPLAY_ARGS > /dev/null
done
# each box scored by the detection that updated its track
python3 "$repo/scripts/nuscenes/nuscenes_tracking.py" results "$work" engine.jsonl "$out" --score detection
