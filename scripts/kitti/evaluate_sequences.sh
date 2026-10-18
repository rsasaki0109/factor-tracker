#!/usr/bin/env bash
# Runs prepare -> replay (three modes) -> evaluate -> report for every sequence
# directory written by convert_tracking.py (or already holding setup.json,
# inputs.jsonl and truth.jsonl), then aggregates the metrics.
#
# usage: scripts/kitti/evaluate_sequences.sh <converted-root> <results-dir> [sequence ...]
#
# REPLAY_ARGS is passed to every `replay` call. It defaults to an initial
# velocity standard deviation of 10 m/s, because KITTI tracks live in the world
# frame where vehicles move at 10-30 m/s (see docs/evaluation.md). Set
# REPLAY_ARGS="" for the library defaults.
set -euo pipefail
REPLAY_ARGS=${REPLAY_ARGS---initial-velocity-variance 100}
# EVAL_ARGS is passed to every `evaluate` call, e.g. "--match-distance-m 1".
EVAL_ARGS=${EVAL_ARGS:-}

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <converted-root> <results-dir> [sequence ...]" >&2
  exit 2
fi
converted=$1
results=$2
shift 2

repo=$(cd "$(dirname "$0")/../.." && pwd)
cargo build --release -q -p perception-cli --manifest-path "$repo/Cargo.toml"
cli="$repo/target/release/perception-cli"
modes=(lidar_only camera_associated camera_assisted)

if [[ $# -gt 0 ]]; then
  sequences=("$@")
else
  sequences=()
  for dir in "$converted"/*/; do sequences+=("$(basename "$dir")"); done
fi

for seq in "${sequences[@]}"; do
  in="$converted/$seq"
  out="$results/$seq"
  mkdir -p "$out"
  if [[ -f "$in/setup.json" ]]; then
    # Already in engine JSONL (e.g. scripts/coda/convert_coda.py).
    [[ "$in" -ef "$out" ]] || cp "$in/setup.json" "$in/inputs.jsonl" "$in/truth.jsonl" "$out/"
  else
    read -r width height < "$in/image_size.txt"
    labels=()
    [[ -d "$in/labels_lidar" ]] && labels=(--lidar-labels "$in/labels_lidar")
    "$cli" prepare --calib "$in/calib.txt" --times "$in/times.txt" --poses "$in/poses.txt" \
      --lidar-detections "$in/detections_lidar" --camera-detections "$in/detections_camera" \
      "${labels[@]}" --image-width "$width" --image-height "$height" \
      --out-setup "$out/setup.json" --out-inputs "$out/inputs.jsonl" --out-truth "$out/truth.jsonl" \
      > "$out/prepare.log"
  fi
  for mode in "${modes[@]}"; do
    "$cli" replay --setup "$out/setup.json" --inputs "$out/inputs.jsonl" --mode "$mode" \
      --output "$out/$mode.outputs.jsonl" --diagnostics "$out/$mode.diagnostics.json" \
      $REPLAY_ARGS > "$out/$mode.replay.log"
    "$cli" evaluate --output "$out/$mode.outputs.jsonl" --truth "$out/truth.jsonl" \
      --metrics "$out/$mode.metrics.json" $EVAL_ARGS > /dev/null
  done
  "$cli" report --setup "$out/setup.json" --output "$out/camera_assisted.outputs.jsonl" \
    --truth "$out/truth.jsonl" --metrics "$out/camera_assisted.metrics.json" \
    --diagnostics "$out/camera_assisted.diagnostics.json" \
    --html "$out/report.html" --title "KITTI tracking $seq (camera_assisted)" > /dev/null
  echo "$seq done"
done

python3 "$repo/scripts/kitti/summarize.py" "$results" "${modes[@]}"
