#!/usr/bin/env python3
r"""Full two-stage CODa PV-RCNN on a GPU (upstream OpenPCDet).

The CPU detector (scripts/coda/detect_lidar_cpu.py) runs only the first stage
of the published CODa checkpoint. This runs the whole model: VoxelBackBone8x
-> BEV backbone -> CenterHead proposals -> VoxelSetAbstraction keypoints ->
PV-RCNN RoI refinement and rescoring, with the model config from
https://github.com/ut-amrl/coda-models and upstream OpenPCDet's (newer PyTorch)
implementation of the same modules.

Output is the CPU detector's format, one JSON per frame
(`<out>/<window>/3d_bbox_os1_<seq>_<frame>.json`, CODa boxes plus `score`),
so scripts/coda/convert_coda.py --lidar-detections reads it unchanged.

Run from coda-models/tools so the config's _BASE_CONFIG_ paths resolve (the
installed upstream `pcdet` package is the one imported):

    cd /workspace/coda-models/tools
    python /workspace/perception-engine/scripts/runpod/detect_coda_pvrcnn.py \
        --cfg cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_headfull.yaml \
        --ckpt /workspace/coda128_allclass_bestoracle.pth --coda-root /workspace/CODa \
        --out /workspace/coda_pvrcnn --window 20 8640 8799 --window 4 6870 7029
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coda"))

from detect_lidar_cpu import CLASS_NAMES, TO_CODA  # noqa: E402

# pcdet.datasets imports every dataset module, and the Argoverse 2 one needs
# the av2 package at import time; it is not used here, so stub it.
try:
    import av2.utils.io  # noqa: F401
except ImportError:
    import types

    for name in ("av2", "av2.utils", "av2.utils.io"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["av2.utils.io"].read_feather = None

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


class Scans(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, scans, logger):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=None,
                         logger=logger)
        self.scans = scans

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        points = np.fromfile(self.scans[index], dtype=np.float32).reshape(-1, 4)
        points = points[np.isfinite(points).all(axis=1)]
        return self.prepare_data(data_dict={"points": points, "frame_id": index})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--coda-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", nargs=3, action="append", metavar=("SEQ", "FIRST", "LAST"), required=True)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    if list(cfg.CLASS_NAMES) != CLASS_NAMES:
        raise SystemExit("config class list differs from detect_lidar_cpu.CLASS_NAMES")
    logger = common_utils.create_logger()
    model = None
    for seq, first, last in args.window:
        first, last = int(first), int(last)
        frames = [f for f in range(first, last + 1)
                  if (args.coda_root / "3d_comp" / "os1" / seq / f"3d_comp_os1_{seq}_{f}.bin").exists()]
        scans = [args.coda_root / "3d_comp" / "os1" / seq / f"3d_comp_os1_{seq}_{f}.bin" for f in frames]
        dataset = Scans(cfg.DATA_CONFIG, cfg.CLASS_NAMES, scans, logger)
        if model is None:
            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
            model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=False)
            model.cuda().eval()
        out = args.out / f"coda{seq}_{first}"
        out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with torch.no_grad():
            for i, frame in enumerate(frames):
                batch = dataset.collate_batch([dataset[i]])
                load_data_to_gpu(batch)
                pred, _ = model(batch)
                boxes = pred[0]["pred_boxes"].cpu().numpy()
                scores = pred[0]["pred_scores"].cpu().numpy()
                labels = pred[0]["pred_labels"].cpu().numpy()
                records = []
                for box, score, label in zip(boxes, scores, labels):
                    name = CLASS_NAMES[int(label) - 1]
                    records.append({
                        "classId": TO_CODA.get(name, name), "modelClass": name, "score": round(float(score), 4),
                        "cX": float(box[0]), "cY": float(box[1]), "cZ": float(box[2]),
                        "l": float(box[3]), "w": float(box[4]), "h": float(box[5]), "r": 0.0, "p": 0.0,
                        "y": float(box[6]),
                    })
                (out / f"3d_bbox_os1_{seq}_{frame}.json").write_text(json.dumps({"3dbbox": records}))
        print(f"coda{seq}_{first}: {len(frames)} scans in {time.time() - started:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
