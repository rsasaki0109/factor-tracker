#!/usr/bin/env python3
r"""Trains the CODa pedestrian detector (cfgs/coda_ped_centerpoint.yaml) on a GPU.

Reads the scans listed in `<coda-root>/training_frames.txt` (written by
scripts/coda/fetch_coda_training.py): `3d_comp/os1/<seq>/*.bin` and the
Pedestrian boxes of `3d_bbox/os1/<seq>/*.json`, both in the OS1 frame.
Uses the coda-models fork of OpenPCDet (pcdet 0.5) for the model, the
augmentation and the optimizer; run from coda-models/tools:

    cd /workspace/coda-models/tools
    python /workspace/pe/scripts/runpod/train_coda_ped.py \
        --cfg /workspace/pe/scripts/runpod/cfgs/coda_ped_centerpoint.yaml \
        --coda-root /workspace/CODa --out /workspace/train

`--init-ckpt` fine-tunes from the published checkpoint instead of training
from scratch. Writes `<out>/checkpoint_epoch_<n>.pth` after every epoch and logs the loss
and the time per iteration. `--max-iters` stops early (a smoke test that
measures speed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils

sys.path.insert(0, str(Path.cwd()))  # coda-models/tools, for train_utils
from train_utils.optimization import build_optimizer, build_scheduler  # noqa: E402

from init_from_published import load_published  # noqa: E402


class CodaFrames(DatasetTemplate):
    """Labelled CODa scans; boxes as [x, y, z, l, w, h, yaw] in the OS1 frame."""

    def __init__(self, dataset_cfg, class_names, frames, root, training, logger):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training,
                         root_path=root, logger=logger)
        self.frames = frames
        self.shift = np.array(dataset_cfg.get("SHIFT_COOR", [0.0, 0.0, 0.0]), dtype=np.float32)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        seq, frame = self.frames[index]
        points = np.fromfile(self.root_path / "3d_comp" / "os1" / seq / f"3d_comp_os1_{seq}_{frame}.bin",
                             dtype=np.float32).reshape(-1, 4)
        points = points[np.isfinite(points).all(axis=1)]
        points[:, :3] += self.shift
        labels = json.loads((self.root_path / "3d_bbox" / "os1" / seq / f"3d_bbox_os1_{seq}_{frame}.json").read_text())
        boxes = [[b["cX"], b["cY"], b["cZ"], b["l"], b["w"], b["h"], b["y"]] for b in labels["3dbbox"]]
        names = [b["classId"] for b in labels["3dbbox"]]
        gt_boxes = np.array(boxes, dtype=np.float32).reshape(-1, 7)
        gt_boxes[:, :3] += self.shift
        return self.prepare_data(data_dict={
            "points": points, "gt_boxes": gt_boxes, "gt_names": np.array(names), "frame_id": f"{seq}_{frame}",
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--coda-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="override OPTIMIZATION.NUM_EPOCHS")
    parser.add_argument("--batch", type=int, help="override OPTIMIZATION.BATCH_SIZE_PER_GPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-iters", type=int, help="stop after this many iterations (smoke test)")
    parser.add_argument("--init-ckpt", type=Path,
                        help="start from the published CODa PV-RCNN checkpoint (init_from_published.py)")
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    if args.set:
        cfg_from_list([x for kv in args.set for x in kv], cfg)
    epochs = args.epochs or cfg.OPTIMIZATION.NUM_EPOCHS
    batch = args.batch or cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    args.out.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(args.out / "train.log")
    frames = [tuple(line.split()) for line in (args.coda_root / "training_frames.txt").read_text().splitlines()]
    frames = [(s, int(f)) for s, f in frames]
    dataset = CodaFrames(cfg.DATA_CONFIG, cfg.CLASS_NAMES, frames, args.coda_root, True, logger)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=args.workers,
                                         collate_fn=dataset.collate_batch, drop_last=True, pin_memory=True,
                                         persistent_workers=args.workers > 0)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    if args.init_ckpt:
        load_published(model, args.init_ckpt, list(cfg.CLASS_NAMES), logger)
    model = model.cuda()
    model.train()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    scheduler, _ = build_scheduler(optimizer, total_iters_each_epoch=len(loader), total_epochs=epochs,
                                   last_epoch=-1, optim_cfg=cfg.OPTIMIZATION)
    model_func = model_fn_decorator()
    logger.info(f"{len(dataset)} scans, batch {batch}, {len(loader)} iterations per epoch, {epochs} epochs")

    it = 0
    started = time.time()
    for epoch in range(epochs):
        window_start, window_loss = time.time(), []
        for batch_dict in loader:
            scheduler.step(it)
            optimizer.zero_grad()
            loss, tb_dict, _ = model_func(model, batch_dict)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            optimizer.step()
            it += 1
            window_loss.append(loss.item())
            if it % 50 == 0:
                per_iter = (time.time() - window_start) / len(window_loss)
                remaining = per_iter * (len(loader) * epochs - it) / 3600
                logger.info(f"epoch {epoch + 1} iter {it} loss {np.mean(window_loss):.3f} "
                            f"lr {optimizer.lr:.5f} {per_iter:.2f} s/iter, ~{remaining:.1f} h left, "
                            f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
                window_start, window_loss = time.time(), []
            if args.max_iters and it >= args.max_iters:
                break
        torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "it": it},
                   args.out / f"checkpoint_epoch_{epoch + 1}.pth")
        logger.info(f"epoch {epoch + 1} done after {(time.time() - started) / 3600:.2f} h")
        if args.max_iters and it >= args.max_iters:
            break
    logger.info("training finished")


if __name__ == "__main__":
    main()
