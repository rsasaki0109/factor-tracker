#!/usr/bin/env bash
# Replays the ETH-2 Mid-360 detections with a few tracker settings and scores
# them against the identity labels (scans 3700-3800, pedestrians within 8 m).
#
# usage: benchmarks/grandtour_eth2/evaluate.sh [work-dir]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../.." && pwd)
work=${1:-$(mktemp -d)}; mkdir -p "$work"
cargo build --release -q -p perception-cli --manifest-path "$repo/Cargo.toml"
cli=$repo/target/release/perception-cli
for f in setup.json inputs.jsonl truth.jsonl; do gzip -dc "$here/$f.gz" > "$work/$f"; done
prior="--initial-velocity-variance 1 --accel-spectral-density 0.5"
printf "%-36s %8s %9s\n" config switches coverage
while read -r name flags; do
  [[ -z "$name" ]] && continue
  $cli replay --setup "$work/setup.json" --inputs "$work/inputs.jsonl" --mode lidar_only \
    --output "$work/$name.jsonl" --diagnostics /dev/null $prior $flags > /dev/null
  $cli evaluate --output "$work/$name.jsonl" --truth "$work/truth.jsonl" --metrics "$work/$name.metrics.json" \
    --match-distance-m 1 > /dev/null
  python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['modes'][0]; print(f\"{sys.argv[2]:36s} {m['id_switches']:8d} {m['track_coverage']:9.3f}\")" \
    "$work/$name.metrics.json" "$name"
done <<'CONFIGS'
base
birth_1.0m --min-birth-distance-m 1.0
birth_1.5m_persistence_3 --min-birth-distance-m 1.5 --birth-persistence 3
distance_1.0m --max-association-distance-m 1.0 --recent-track-window-ns 0
CONFIGS
