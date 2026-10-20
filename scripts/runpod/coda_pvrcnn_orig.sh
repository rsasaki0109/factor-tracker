#!/usr/bin/env bash
# The CODa PV-RCNN checkpoint in its original environment: coda-models as
# published (pcdet 0.5 fork) on PyTorch 1.10 + CUDA 11.3, for comparison with
# coda_pvrcnn.sh (upstream OpenPCDet). Same detector script and outputs.
#
# Pod image: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel; an Ampere GPU
# (sm_86: RTX A4000/A5000/3090/A6000, A40). No apt (the image's NVIDIA apt
# keys are stale): downloads use Python.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/workspace}
CODA=$WORK/CODa
CKPT=$WORK/coda128_allclass_bestoracle.pth
CFG=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_headfull.yaml
WINDOWS=${WINDOWS:-"20:8640 13:9700"}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}

python - "$WORK" "$CKPT" <<'PY'
import io, sys, tarfile, urllib.request
from pathlib import Path
work, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
if not (work / "coda-models").exists():
    data = urllib.request.urlopen("https://codeload.github.com/ut-amrl/coda-models/tar.gz/c9a12d740e55c8794da32d71e8cf30af8cefcff8").read()
    tarfile.open(fileobj=io.BytesIO(data)).extractall(work)
    next(work.glob("coda-models-*")).rename(work / "coda-models")
if not ckpt.exists():
    urllib.request.urlretrieve("https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/pretrained_models/128channel/"
                               "coda128_allclass_bestoracle.pth", ckpt)
PY
pip install -q "numpy<1.22" numba easydict pyyaml scikit-image tqdm tensorboardX spconv-cu113 shapely scipy pyquaternion
(cd "$WORK/coda-models" && python setup.py develop > "$WORK/build.log" 2>&1) || { tail -40 "$WORK/build.log"; exit 1; }
python -c "import pcdet, spconv, torch; print('pcdet', pcdet.__file__, 'torch', torch.__version__, torch.version.cuda)"

for w in $WINDOWS; do
  seq=${w%%:*}; first=${w#*:}; last=$((first + 159))
  for try in 1 2 3 4 5; do
    python "$repo/scripts/coda/fetch_coda.py" --coda-root "$CODA" --sequence "$seq" --first "$first" --last "$last" \
      --lidar-only && break
    echo "fetch failed, retrying in 30 s"; sleep 30
  done
  (cd "$WORK/coda-models/tools" && python "$repo/scripts/runpod/detect_coda_pvrcnn.py" --cfg "$CFG" --ckpt "$CKPT" \
    --coda-root "$CODA" --out "$WORK/coda_pvrcnn" --window "$seq" "$first" "$last")
done
tar czf "$WORK/coda_pvrcnn_dets.tar.gz" -C "$WORK" coda_pvrcnn
echo "done: $WORK/coda_pvrcnn_dets.tar.gz"
