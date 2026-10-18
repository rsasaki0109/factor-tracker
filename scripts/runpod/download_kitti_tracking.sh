#!/usr/bin/env bash
# Downloads and unpacks the KITTI tracking benchmark (training + testing).
#
# Accept the KITTI dataset terms at https://www.cvlibs.net/datasets/kitti/
# before running this. The data is not redistributed by this repository.
# Sizes: velodyne ~37 GB, image_2 ~16 GB, calib/oxts/labels ~10 MB.
#
# usage: KITTI_ROOT=/workspace/kitti bash scripts/runpod/download_kitti_tracking.sh [parts...]
#        parts default to: calib oxts label_2 velodyne image_2
set -euo pipefail

KITTI_ROOT=${KITTI_ROOT:-/workspace/kitti}
BASE=https://s3.eu-central-1.amazonaws.com/avg-kitti
parts=("$@")
[[ ${#parts[@]} -eq 0 ]] && parts=(calib oxts label_2 velodyne image_2)

mkdir -p "$KITTI_ROOT"
cd "$KITTI_ROOT"
for part in "${parts[@]}"; do
  zip="data_tracking_$part.zip"
  if [[ ! -f "$zip.done" ]]; then
    curl -fL --retry 5 -C - -o "$zip" "$BASE/$zip"
    unzip -qo "$zip"
    rm -f "$zip"
    touch "$zip.done"
  fi
  echo "$part ready"
done
