#!/usr/bin/env bash
# Installs the coda-models fork of OpenPCDet (pcdet 0.5, pinned commit
# c9a12d74) on PyTorch 1.10 + CUDA 11.3 and downloads the published CODa
# PV-RCNN checkpoint to $WORK/coda128_allclass_bestoracle.pth.
#
# Pod image: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel; an Ampere GPU
# (sm_86: RTX A4000/A5000/3090/A6000, A40). No apt (the image's NVIDIA apt
# keys are stale): downloads use Python.
set -euo pipefail
WORK=${WORK:-/workspace}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
python - "$WORK" "$WORK/coda128_allclass_bestoracle.pth" <<'PY'
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
