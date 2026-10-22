"""Initializes the pedestrian detector from the published CODa PV-RCNN.

The published checkpoint's first stage (VFE, VoxelBackBone8x, BEV backbone,
CenterHead) has the same layers as cfgs/coda_ped_centerpoint.yaml except the
heatmap output, which has one channel per class (56). Every tensor with a
matching name and shape is copied; the heatmap's last layer keeps only the
channels of the model's classes. The second-stage tensors are ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coda"))

from detect_lidar_cpu import CLASS_NAMES as PUBLISHED_CLASSES  # noqa: E402


def published_state(model_state: dict, published: dict, class_names: list) -> tuple:
    """Returns (state dict for the model, names copied, names left at their init)."""
    channels = [PUBLISHED_CLASSES.index(name) for name in class_names]
    state, copied, missing = {}, [], []
    for name, target in model_state.items():
        source = published.get(name)
        if source is not None and source.shape == target.shape:
            state[name] = source
            copied.append(name)
        elif (source is not None and source.dim() == target.dim() and source.shape[0] == len(PUBLISHED_CLASSES)
              and source.shape[1:] == target.shape[1:] and target.shape[0] == len(channels)):
            state[name] = source[channels].clone()
            copied.append(name)
        else:
            state[name] = target
            missing.append(name)
    return state, copied, missing


def load_published(model: torch.nn.Module, checkpoint: Path, class_names: list, logger) -> None:
    published = torch.load(str(checkpoint), map_location="cpu")["model_state"]
    state, copied, missing = published_state(model.state_dict(), published, class_names)
    model.load_state_dict(state)
    logger.info(f"initialized {len(copied)} of {len(state)} tensors from {checkpoint.name}; not found: {missing}")
    learned = [n for n in missing if not n.endswith("num_batches_tracked")]
    if learned:
        raise SystemExit(f"published checkpoint does not cover {learned}")
