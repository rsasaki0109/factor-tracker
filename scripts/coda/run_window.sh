#!/usr/bin/env bash
# One CODa window end to end: fetch -> YOLO -> CPU LiDAR detector -> engine
# JSONL (oracle and real LiDAR) -> ReID embeddings -> tracker configurations
# scored over the whole range and in robot-centric zones.
#
# usage: scripts/coda/run_window.sh <sequence> <first> <last> <workdir>
# env:   PY      python with torch, timm, ultralytics, shapely (default python3)
#        CKPT    CODa checkpoint (default <workdir>/coda128_allclass_bestoracle.pth)
#        YOLO    Ultralytics weights (default yolo11x.pt)
#        JOBS    parallel LiDAR detector processes (default 4)
set -euo pipefail
seq=$1; first=$2; last=$3; work=$4
repo=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}; CKPT=${CKPT:-$work/coda128_allclass_bestoracle.pth}; YOLO=${YOLO:-yolo11x.pt}; JOBS=${JOBS:-4}
coda=$work/CODa; name=coda${seq}_${first}; det=$work/det/$name; seqs=$work/seq
cli=$repo/target/release/perception-cli
cargo build --release -q -p perception-cli --manifest-path "$repo/Cargo.toml"

"$PY" "$repo/scripts/coda/fetch_coda.py" --coda-root "$coda" --sequence "$seq" --first "$first" --last "$last"
"$PY" "$repo/scripts/runpod/detect_camera_yolo.py" --image-dir "$coda/2d_rect/cam0/$seq" --out-dir "$det/camera" \
  --model "$YOLO" > "$work/$name.yolo.log" 2>&1
# The image directory may hold other windows of the same sequence; keep this one.
span=$(( last - first + 1 )); chunk=$(( (span + JOBS - 1) / JOBS ))
for ((a = first; a <= last; a += chunk)); do
  b=$(( a + chunk - 1 < last ? a + chunk - 1 : last ))
  "$PY" "$repo/scripts/coda/detect_lidar_cpu.py" --ckpt "$CKPT" --coda-root "$coda" --sequence "$seq" \
    --first "$a" --last "$b" --out "$det/lidar" --threads 1 > "$work/$name.lidar_$a.log" 2>&1 &
done
wait

python3 "$repo/scripts/coda/convert_coda.py" --coda-root "$coda" --sequence "$seq" --first "$first" --last "$last" \
  --camera-detections "$det/camera" --out "$seqs/$name-oracle/$seq"
python3 "$repo/scripts/coda/convert_coda.py" --coda-root "$coda" --sequence "$seq" --first "$first" --last "$last" \
  --camera-detections "$det/camera" --lidar-detections "$det/lidar" --lidar-score-threshold 0.4 \
  --out "$seqs/$name-real/$seq"
for kind in oracle real; do
  "$PY" "$repo/scripts/reid/embed_detections.py" --seq-dir "$seqs/$name-$kind/$seq" \
    --image-pattern "$coda/2d_rect/cam0/$seq/2d_rect_cam0_${seq}_{frame}.png" --out "$seqs/$name-$kind-reid/$seq" \
    > "$work/$name.$kind.reid.log" 2>&1
done

out=$work/results/$name.tsv
mkdir -p "$(dirname "$out")"
echo -e "detections\tconfig\tzone\tid_switches\tcoverage\tgt_object_frames" > "$out"
prior="--initial-velocity-variance 1 --accel-spectral-density 0.5"
for kind in oracle real; do
  d=$seqs/$name-$kind-reid/$seq
  while read -r config flags; do
    o=$work/runs/$name-$kind-$config; mkdir -p "$o"
    $cli replay --setup "$d/setup.json" --inputs "$d/inputs.jsonl" --mode lidar_only --output "$o/out.jsonl" \
      --diagnostics "$o/diag.json" $prior $flags > /dev/null
    for zone in "front8:--zone-range-m 8 --zone-half-angle-deg 40" "near5:--zone-range-m 5 --zone-half-angle-deg 40" \
                "around8:--zone-range-m 8" "all:"; do
      $cli evaluate --output "$o/out.jsonl" --truth "$d/truth.jsonl" --metrics "$o/${zone%%:*}.json" \
        --match-distance-m 1 --setup "$d/setup.json" ${zone#*:} > /dev/null
      python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['modes'][0]; print('\t'.join(map(str,[sys.argv[2],sys.argv[3],sys.argv[4],m['id_switches'],round(m['track_coverage'],3),m['gt_object_times']])))" \
        "$o/${zone%%:*}.json" "$kind" "$config" "${zone%%:*}" >> "$out"
    done
  done <<'CONFIGS'
base
reid10 --appearance-weight 10
birth10 --min-birth-distance-m 1.0
birth10_reid10 --min-birth-distance-m 1.0 --appearance-weight 10
birth10_reid5 --min-birth-distance-m 1.0 --appearance-weight 5
birth12_reid10 --min-birth-distance-m 1.2 --appearance-weight 10
birth08_reid10 --min-birth-distance-m 0.8 --appearance-weight 10
CONFIGS
done
echo "results -> $out"
