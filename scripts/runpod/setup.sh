#!/usr/bin/env bash
# One-time setup inside a RunPod GPU pod (PyTorch template, CUDA 11.8 or 12.x).
# Installs OpenPCDet (LiDAR 3D detection) and Ultralytics (camera 2D detection).
#
# usage: bash scripts/runpod/setup.sh            # installs into $WORK (default /workspace)
set -euo pipefail

WORK=${WORK:-/workspace}
PCDET_REF=${PCDET_REF:-v0.6.0}

if command -v apt-get > /dev/null; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends git unzip libgl1 libglib2.0-0 > /dev/null
fi

cuda=$(python -c "import torch; print(torch.version.cuda or '')")
case "$cuda" in
  11.8*) spconv=spconv-cu118 ;;
  12.*)  spconv=spconv-cu120 ;;
  *) echo "unsupported or missing CUDA in torch: '$cuda'" >&2; exit 1 ;;
esac
echo "torch CUDA $cuda -> $spconv"

pip install -q "numpy<2" "$spconv" ultralytics

if [[ ! -d "$WORK/OpenPCDet" ]]; then
  git clone --depth 1 --branch "$PCDET_REF" https://github.com/open-mmlab/OpenPCDet.git "$WORK/OpenPCDet"
fi
cd "$WORK/OpenPCDet"
pip install -q -r requirements.txt
python setup.py develop > /dev/null
python -c "import pcdet, spconv, ultralytics; print('pcdet', pcdet.__version__, '| ultralytics', ultralytics.__version__)"
