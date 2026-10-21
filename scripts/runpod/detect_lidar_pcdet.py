#!/usr/bin/env python3
r"""Runs an OpenPCDet KITTI model over KITTI tracking velodyne scans.

Writes one 16-field KITTI detection file per frame (camera 0 rectified frame,
score last) to <out>/lidar/<seq>/<frame>.txt, the layout that
scripts/kitti/convert_tracking.py --detections-root expects.

Run from OpenPCDet/tools so the model config's _BASE_CONFIG_ resolves:

    cd /workspace/OpenPCDet/tools
    python /workspace/factor-tracker/scripts/runpod/detect_lidar_pcdet.py \
        --kitti-root /workspace/kitti --out /workspace/pe-out/detections \
        --cfg cfgs/kitti_models/pointpillar.yaml --ckpt /workspace/ckpt/pointpillar_7728.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kitti"))

import kitti_tools  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


class SequenceScans(DatasetTemplate):
    """The scans of one sequence, cropped to the camera 2 field of view (the
    KITTI models are trained on FOV points only)."""

    def __init__(self, dataset_cfg, class_names, scans, calib, image_size, logger):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=None, logger=logger
        )
        self.scans = scans
        self.calib = calib
        self.image_size = image_size

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        points = np.fromfile(self.scans[index], dtype=np.float32).reshape(-1, 4)
        points = points[kitti_tools.fov_mask(points, self.calib, self.image_size)]
        return self.prepare_data(data_dict={"points": points, "frame_id": index})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cfg", required=True, help="OpenPCDet KITTI model config")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()
    split = args.kitti_root / args.split
    sequences = args.sequences or sorted(p.name for p in (split / "velodyne").iterdir() if p.is_dir())

    model = None
    for seq in sequences:
        calib = kitti_tools.TrackingCalib.read(split / "calib" / f"{seq}.txt")
        first_image = split / "image_02" / seq / "000000.png"
        image_size = kitti_tools.png_size(first_image) if first_image.exists() else (1242, 375)
        scans = sorted((split / "velodyne" / seq).glob("*.bin"))
        dataset = SequenceScans(cfg.DATA_CONFIG, cfg.CLASS_NAMES, scans, calib, image_size, logger)
        if model is None:
            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
            model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=True)
            model.cuda().eval()

        out_dir = args.out / "lidar" / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with torch.no_grad():
            for index, scan in enumerate(scans):
                batch = dataset.collate_batch([dataset[index]])
                load_data_to_gpu(batch)
                predictions, _ = model.forward(batch)
                boxes = predictions[0]["pred_boxes"].cpu().numpy()
                scores = predictions[0]["pred_scores"].cpu().numpy()
                labels = predictions[0]["pred_labels"].cpu().numpy()
                keep = scores >= args.score_threshold
                names = [cfg.CLASS_NAMES[label - 1] for label in labels[keep]]
                lines = kitti_tools.lidar_boxes_to_kitti_lines(boxes[keep], scores[keep], names, calib, image_size)
                (out_dir / f"{scan.stem}.txt").write_text("".join(line + "\n" for line in lines))
        elapsed = time.time() - started
        print(f"{seq}: {len(scans)} scans in {elapsed:.1f} s ({len(scans) / max(elapsed, 1e-9):.1f} Hz)")


if __name__ == "__main__":
    main()
