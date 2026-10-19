#!/usr/bin/env bash
# Scores the refiner against the base tracker on the development sets:
# five CODa windows (real detections) and GrandTour ETH-2.
# usage: dev_eval.sh <data-dir> <work-dir> [refine.py options]
#   <data-dir>/coda/<window>/{setup,inputs,truth}.json(l), <data-dir>/eth2/...
set -euo pipefail
data=$1; work=$2; shift 2
repo=$(cd "$(dirname "$0")/../.." && pwd); cli=$repo/target/release/perception-cli
prior="--initial-velocity-variance 1 --accel-spectral-density 0.5"
mkdir -p "$work"
score() {  # output truth setup zone-flags -> "switches coverage"
  $cli evaluate --output "$1" --truth "$2" --metrics "$work/m.json" --match-distance-m 1 --setup "$3" $4 > /dev/null
  python3 -c "import json; m=json.load(open('$work/m.json'))['modes'][0]; print(m['id_switches'], round(m['track_coverage'],3))"
}
printf "%-16s %-8s %16s %16s\n" set zone base refined
for d in "$data"/coda/* "$data"/eth2; do
  name=$(basename "$d"); o=$work/$name; mkdir -p "$o"
  [[ -f ${BASE_CACHE:-/nonexistent}/$name/base.jsonl ]] && { mkdir -p "$o"; cp "${BASE_CACHE}/$name/base.jsonl" "$o/"; }
  [[ -f $o/base.jsonl ]] || $cli replay --setup "$d/setup.json" --inputs "$d/inputs.jsonl" --mode lidar_only \
    --output "$o/base.jsonl" --diagnostics /dev/null $prior > /dev/null
  ${PY:-python3} "$repo/scripts/fgo/${REFINER:-refine.py}" "$o/base.jsonl" "$o/refined.jsonl" "$@" > "$o/refine.log"
  zones=("all:" "around8:--zone-range-m 8"); [[ $name == eth2 ]] && zones=("all:")
  for z in "${zones[@]}"; do
    printf "%-16s %-8s %16s %16s\n" "$name" "${z%%:*}" "$(score $o/base.jsonl $d/truth.jsonl $d/setup.json "${z#*:}")" \
      "$(score $o/refined.jsonl $d/truth.jsonl $d/setup.json "${z#*:}")"
  done
done
