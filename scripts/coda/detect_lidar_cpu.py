#!/usr/bin/env python3
r"""CPU inference of the first stage of the CODa PV-RCNN model.

Runs the published CODa checkpoint `coda128_allclass_bestoracle.pth`
(OpenPCDet PV-RCNN, CenterHead, 56 classes; config
`tools/cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_*.yaml`
in https://github.com/ut-amrl/coda-models) without CUDA, spconv or OpenPCDet.
It also runs the pedestrian-only CenterPoint trained by
`scripts/runpod/train_coda_ped.py` (`scripts/runpod/cfgs/coda_ped_centerpoint.yaml`:
same network, one class, +-51.2 m instead of +-75.2 m, about twice as fast);
the model is recognised from the checkpoint's number of classes. Its weights:
https://huggingface.co/rsasaki0109/coda-ped-centerpoint (CC BY-NC-SA 4.0).

    points -> mean voxel features -> VoxelBackBone8x (sparse 3D conv)
           -> height compression -> BaseBEVBackbone -> CenterHead -> boxes

The sparse convolutions are implemented with gather/scatter over active
voxels and compute the same function as spconv's SubMConv3d / SparseConv3d.
The second stage (keypoint set abstraction and RoI refinement) needs custom
CUDA operators and is NOT run: boxes and scores are the CenterHead proposals,
and non-maximum suppression is a rotated bird's-eye-view IoU NMS done with
Shapely at the config's final threshold (0.1). Expect lower precision than the
published two-stage numbers. About 20-30 s per 128-beam scan on one CPU core
with the published model.

Output: one CODa-style JSON per frame (`3d_bbox_os1_<seq>_<frame>.json`) with
an added `score`, readable by convert_coda.py --lidar-detections.

usage: detect_lidar_cpu.py --ckpt coda128_allclass_bestoracle.pth --coda-root CODa \
           --sequence 20 --first 8640 --last 8800 --out det/coda20_lidar
Requires PyTorch (CPU), NumPy and Shapely.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon

# --- model configuration (pvrcnn_allclass128full) -------------------------------
CLASS_NAMES = [
    "Car", "Pedestrian", "Cyclist", "Motorcycle", "Scooter", "Tree", "TrafficSign", "Canopy", "TrafficLight",
    "BikeRack", "Bollard", "ConstructionBarrier", "ParkingKiosk", "Mailbox", "FireHydrant", "FreestandingPlant",
    "Pole", "InformationalSign", "Door", "Fence", "Railing", "Cone", "Chair", "Bench", "Table", "TrashCan",
    "NewspaperDispenser", "RoomLabel", "Stanchion", "SanitizerDispenser", "CondimentDispenser", "VendingMachine",
    "EmergencyAidKit", "FireExtinguisher", "Computer", "Television", "Other", "PickupTruck", "DeliveryTruck",
    "ServiceVehicle", "UtilityVehicle", "FireAlarm", "ATM", "Cart", "Couch", "TrafficArm", "WallSign", "FloorSign",
    "DoorSwitch", "EmergencyPhone", "Dumpster", "VacuumCleaner", "Segway", "Bus", "Skateboard", "WaterFountain",
]
# Model class -> CODa label name (as used by convert_coda.py) for dynamic classes.
TO_CODA = {
    "Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Scooter", "Motorcycle": "Motorcycle",
    "Scooter": "Scooter", "PickupTruck": "Pickup Truck", "DeliveryTruck": "Delivery Truck",
    "ServiceVehicle": "Service Vehicle", "UtilityVehicle": "Utility Vehicle", "Segway": "Segway",
    "Bus": "Bus", "Skateboard": "Skateboard",
}
# Pedestrian-only CenterPoint (scripts/runpod/cfgs/coda_ped_centerpoint.yaml).
PED_CLASS_NAMES = ["Pedestrian"]
POINT_RANGE = np.array([-75.2, -75.2, -2.0, 75.2, 75.2, 4.0])
PED_POINT_RANGE = np.array([-51.2, -51.2, -2.0, 51.2, 51.2, 4.0])
VOXEL_SIZE = np.array([0.1, 0.1, 0.15])
SHIFT_Z = 0.8  # DATA_CONFIG.SHIFT_COOR: points are lifted by 0.8 m before voxelization
MAX_POINTS_PER_VOXEL = 5
FEATURE_STRIDE = 8
BN_EPS = 1e-3


# --- sparse 3D convolution ------------------------------------------------------
class Sparse:
    """Active voxels: `coords` (N, 3) as (z, y, x), `feats` (N, C), grid `shape` (D, H, W)."""

    def __init__(self, coords: torch.Tensor, feats: torch.Tensor, shape: tuple[int, int, int]):
        self.coords, self.feats, self.shape = coords, feats, shape

    def keys(self, coords: torch.Tensor) -> torch.Tensor:
        d, h, w = self.shape
        return (coords[:, 0] * h + coords[:, 1]) * w + coords[:, 2]


def kernel_offsets(kernel):
    kz, ky, kx = kernel
    return [(a, b, c) for a in range(kz) for b in range(ky) for c in range(kx)]


def subm_conv(x: Sparse, weight: torch.Tensor) -> Sparse:
    """SubMConv3d, kernel 3, padding 1: outputs only at the input's active sites."""
    keys = x.keys(x.coords)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    out = torch.zeros(len(keys), weight.shape[0])
    bounds = torch.tensor(x.shape)
    for a, b, c in kernel_offsets(weight.shape[1:4]):
        offset = torch.tensor([a - 1, b - 1, c - 1])
        neighbour = x.coords + offset
        valid = ((neighbour >= 0) & (neighbour < bounds)).all(dim=1)
        pos = torch.searchsorted(sorted_keys, x.keys(neighbour.clamp(min=0)))
        pos = pos.clamp(max=len(sorted_keys) - 1)
        hit = valid & (sorted_keys[pos] == x.keys(neighbour.clamp(min=0)))
        rows = torch.nonzero(hit, as_tuple=True)[0]
        out.index_add_(0, rows, x.feats[order[pos[rows]]] @ weight[:, a, b, c, :].T)
    return Sparse(x.coords, out, x.shape)


def sparse_conv(x: Sparse, weight: torch.Tensor, stride, padding) -> Sparse:
    """SparseConv3d: an output site is active if any input in its receptive field is."""
    kernel = weight.shape[1:4]
    stride, padding = torch.tensor(stride), torch.tensor(padding)
    in_shape = torch.tensor(x.shape)
    out_shape = (in_shape + 2 * padding - torch.tensor(kernel)) // stride + 1
    out_keys, in_rows, taps = [], [], []
    d, h, w = (int(v) for v in out_shape)
    for tap, (a, b, c) in enumerate(kernel_offsets(kernel)):
        numerator = x.coords + padding - torch.tensor([a, b, c])
        o = torch.div(numerator, stride, rounding_mode="floor")
        valid = ((numerator % stride) == 0).all(dim=1) & ((o >= 0) & (o < out_shape)).all(dim=1)
        rows = torch.nonzero(valid, as_tuple=True)[0]
        oc = o[rows]
        out_keys.append((oc[:, 0] * h + oc[:, 1]) * w + oc[:, 2])
        in_rows.append(rows)
        taps.append(torch.full_like(rows, tap))
    out_keys, in_rows, taps = torch.cat(out_keys), torch.cat(in_rows), torch.cat(taps)
    unique, inverse = torch.unique(out_keys, return_inverse=True)
    out = torch.zeros(len(unique), weight.shape[0])
    flat = weight.reshape(weight.shape[0], -1, weight.shape[-1])  # (out, taps, in)
    for tap in range(flat.shape[1]):
        mask = taps == tap
        out.index_add_(0, inverse[mask], x.feats[in_rows[mask]] @ flat[:, tap, :].T)
    coords = torch.stack([unique // (h * w), (unique // w) % h, unique % w], dim=1)
    return Sparse(coords, out, (d, h, w))


def bn_relu(x: torch.Tensor, sd: dict, prefix: str) -> torch.Tensor:
    mean, var = sd[prefix + ".running_mean"], sd[prefix + ".running_var"]
    y = (x - mean) / torch.sqrt(var + BN_EPS) * sd[prefix + ".weight"] + sd[prefix + ".bias"]
    return torch.relu(y)


def bn_relu_2d(x: torch.Tensor, sd: dict, prefix: str) -> torch.Tensor:
    shape = (1, -1, 1, 1)
    mean, var = sd[prefix + ".running_mean"].view(shape), sd[prefix + ".running_var"].view(shape)
    y = (x - mean) / torch.sqrt(var + BN_EPS) * sd[prefix + ".weight"].view(shape) + sd[prefix + ".bias"].view(shape)
    return torch.relu(y)


# --- network --------------------------------------------------------------------
def voxelize(points: np.ndarray, point_range: np.ndarray) -> Sparse:
    xyz = points[:, :3].astype(np.float64).copy()
    xyz[:, 2] += SHIFT_Z
    inside = np.all((xyz >= point_range[:3]) & (xyz < point_range[3:]), axis=1)
    xyz = xyz[inside]
    grid = np.round((point_range[3:] - point_range[:3]) / VOXEL_SIZE).astype(int)  # (x, y, z)
    idx = np.floor((xyz - point_range[:3]) / VOXEL_SIZE).astype(np.int64)
    keys = (idx[:, 2] * grid[1] + idx[:, 1]) * grid[0] + idx[:, 0]
    # Keep the first MAX_POINTS_PER_VOXEL points of each voxel, then average them (MeanVFE).
    order = np.argsort(keys, kind="stable")
    keys, xyz = keys[order], xyz[order]
    unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
    rank = np.arange(len(keys)) - np.repeat(first, counts)
    keep = rank < MAX_POINTS_PER_VOXEL
    voxel_of = np.repeat(np.arange(len(unique)), counts)[keep]
    sums = np.zeros((len(unique), 3))
    np.add.at(sums, voxel_of, xyz[keep])
    feats = sums / np.minimum(counts, MAX_POINTS_PER_VOXEL)[:, None]
    coords = np.stack([unique // (grid[1] * grid[0]), (unique // grid[0]) % grid[1], unique % grid[0]], axis=1)
    # VoxelBackBone8x uses sparse_shape = grid_size[::-1] + [1, 0, 0].
    shape = (int(grid[2]) + 1, int(grid[1]), int(grid[0]))
    return Sparse(torch.from_numpy(coords), torch.from_numpy(feats).float(), shape)


def backbone_3d(x: Sparse, sd: dict) -> torch.Tensor:
    def subm(x, name):
        x = subm_conv(x, sd[name + ".0.weight"])
        return Sparse(x.coords, bn_relu(x.feats, sd, name + ".1"), x.shape)

    def down(x, name, stride, padding):
        x = sparse_conv(x, sd[name + ".0.weight"], stride, padding)
        return Sparse(x.coords, bn_relu(x.feats, sd, name + ".1"), x.shape)

    p = "backbone_3d."
    x = subm(x, p + "conv_input")
    x = subm(x, p + "conv1.0")
    x = down(x, p + "conv2.0", (2, 2, 2), (1, 1, 1))
    x = subm(x, p + "conv2.1")
    x = subm(x, p + "conv2.2")
    x = down(x, p + "conv3.0", (2, 2, 2), (1, 1, 1))
    x = subm(x, p + "conv3.1")
    x = subm(x, p + "conv3.2")
    x = down(x, p + "conv4.0", (2, 2, 2), (0, 1, 1))
    x = subm(x, p + "conv4.1")
    x = subm(x, p + "conv4.2")
    x = down(x, p + "conv_out", (2, 1, 1), (0, 0, 0))
    # HeightCompression: dense (C, D, H, W) -> (C * D, H, W).
    d, h, w = x.shape
    dense = torch.zeros(x.feats.shape[1], d, h, w)
    dense[:, x.coords[:, 0], x.coords[:, 1], x.coords[:, 2]] = x.feats.T
    return dense.reshape(1, -1, h, w)


def backbone_2d(x: torch.Tensor, sd: dict) -> torch.Tensor:
    ups = []
    for level, stride in enumerate((1, 2)):
        p = f"backbone_2d.blocks.{level}."
        x = F.conv2d(F.pad(x, (1, 1, 1, 1)), sd[p + "1.weight"], stride=stride)
        x = bn_relu_2d(x, sd, p + "2")
        for layer in range(5):
            conv = 4 + 3 * layer
            x = F.conv2d(x, sd[p + f"{conv}.weight"], padding=1)
            x = bn_relu_2d(x, sd, p + f"{conv + 1}")
        q = f"backbone_2d.deblocks.{level}."
        up = F.conv_transpose2d(x, sd[q + "0.weight"], stride=(1, 2)[level])
        ups.append(bn_relu_2d(up, sd, q + "1"))
    return torch.cat(ups, dim=1)


def center_head(x: torch.Tensor, sd: dict) -> dict[str, torch.Tensor]:
    p = "dense_head."
    x = F.conv2d(x, sd[p + "shared_conv.0.weight"], sd[p + "shared_conv.0.bias"], padding=1)
    x = bn_relu_2d(x, sd, p + "shared_conv.1")
    out = {}
    for name in ("center", "center_z", "dim", "rot", "hm"):
        q = f"{p}heads_list.0.{name}."
        y = F.conv2d(x, sd[q + "0.0.weight"], sd[q + "0.0.bias"], padding=1)
        y = bn_relu_2d(y, sd, q + "0.1")
        out[name] = F.conv2d(y, sd[q + "1.weight"], sd[q + "1.bias"], padding=1)[0]
    return out


def decode(head: dict[str, torch.Tensor], score_threshold: float, point_range: np.ndarray, max_objects: int = 500):
    heat = torch.sigmoid(head["hm"])
    classes, h, w = heat.shape
    scores, flat = torch.topk(heat.reshape(-1), min(max_objects, heat.numel()))
    keep = scores >= score_threshold
    scores, flat = scores[keep], flat[keep]
    cls = flat // (h * w)
    ys, xs = (flat % (h * w)) // w, flat % w
    cx = (xs.float() + head["center"][0, ys, xs]) * FEATURE_STRIDE * VOXEL_SIZE[0] + point_range[0]
    cy = (ys.float() + head["center"][1, ys, xs]) * FEATURE_STRIDE * VOXEL_SIZE[1] + point_range[1]
    cz = head["center_z"][0, ys, xs] - SHIFT_Z
    dims = torch.exp(head["dim"][:, ys, xs]).T
    yaw = torch.atan2(head["rot"][1, ys, xs], head["rot"][0, ys, xs])
    boxes = torch.stack([cx, cy, cz, dims[:, 0], dims[:, 1], dims[:, 2], yaw], dim=1).numpy()
    return boxes, scores.numpy(), cls.numpy()


def bev_polygon(box) -> Polygon:
    x, y, _, l, w, _, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    corners = [(l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)]
    return Polygon([(x + c * u - s * v, y + s * u + c * v) for u, v in corners])


def rotated_nms(boxes, scores, threshold: float):
    """Class-agnostic greedy NMS on rotated bird's-eye-view IoU (as OpenPCDet's CenterHead)."""
    order = np.argsort(-scores)
    polygons = [bev_polygon(b) for b in boxes]
    kept = []
    for i in order:
        if all(
            np.hypot(*(boxes[i][:2] - boxes[j][:2])) > (boxes[i][3] + boxes[j][3])
            or polygons[i].intersection(polygons[j]).area / max(polygons[i].union(polygons[j]).area, 1e-9) <= threshold
            for j in kept
        ):
            kept.append(i)
    return np.array(kept, dtype=int)


def model_config(sd: dict) -> tuple[list[str], np.ndarray]:
    """Class names and point range of the checkpoint, told apart by the heatmap's class count."""
    classes = sd["dense_head.heads_list.0.hm.1.weight"].shape[0]
    if classes == len(CLASS_NAMES):
        return CLASS_NAMES, POINT_RANGE
    if classes == len(PED_CLASS_NAMES):
        return PED_CLASS_NAMES, PED_POINT_RANGE
    raise ValueError(f"unknown checkpoint: {classes} heatmap classes")


@torch.no_grad()
def detect(points: np.ndarray, sd: dict, score_threshold: float, nms_threshold: float):
    _, point_range = model_config(sd)
    features = backbone_2d(backbone_3d(voxelize(points, point_range), sd), sd)
    boxes, scores, cls = decode(center_head(features, sd), score_threshold, point_range)
    keep = rotated_nms(boxes, scores, nms_threshold)
    return boxes[keep], scores[keep], cls[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--coda-root", type=Path)
    parser.add_argument("--sequence")
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument(
        "--scan-pattern",
        help="any float32 x y z intensity scans, e.g. 'scans/{frame:06d}.bin' (instead of --coda-root); "
        "outputs are named scan_<frame>.json and missing frames are skipped",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="added to point z before inference (and removed from box z) so the ground sits ~0.8 m below "
        "the origin as for the CODa Ouster; e.g. -0.25 for points in the ANYmal base frame",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.1, help="CenterHead SCORE_THRESH")
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.1,
        help="BEV IoU NMS threshold (the config's final NMS_THRESH; the first-stage one is 0.7). For pedestrian "
        "tracking 0.3 with tracker birth suppression scores better (docs/evaluation.md)",
    )
    parser.add_argument("--scan-dir", default="3d_comp", help="3d_comp (ego-compensated) or 3d_raw")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in checkpoint["model_state"].items() if v.is_floating_point()}
    class_names, _ = model_config(sd)
    args.out.mkdir(parents=True, exist_ok=True)
    seq = args.sequence
    if args.scan_pattern is None and (args.coda_root is None or seq is None):
        parser.error("--coda-root and --sequence are required without --scan-pattern")
    for frame in range(args.first, args.last + 1):
        if args.scan_pattern:
            scan = Path(args.scan_pattern.format(frame=frame))
            target = args.out / f"scan_{frame:06d}.json"
            if not scan.exists():
                continue
        else:
            scan = args.coda_root / args.scan_dir / "os1" / seq / f"{args.scan_dir}_os1_{seq}_{frame}.bin"
            target = args.out / f"3d_bbox_os1_{seq}_{frame}.json"
        if target.exists():
            continue
        points = np.fromfile(scan, dtype=np.float32).reshape(-1, 4)
        points = points[np.isfinite(points).all(axis=1)].copy()
        points[:, 2] += args.z_offset
        started = time.time()
        boxes, scores, cls = detect(points, sd, args.score_threshold, args.nms_threshold)
        records = []
        for box, score, c in zip(boxes, scores, cls):
            name = class_names[c]
            records.append({
                "classId": TO_CODA.get(name, name), "modelClass": name, "score": round(float(score), 4),
                "cX": float(box[0]), "cY": float(box[1]), "cZ": float(box[2]) - args.z_offset,
                "l": float(box[3]), "w": float(box[4]), "h": float(box[5]), "r": 0.0, "p": 0.0, "y": float(box[6]),
            })
        target.write_text(json.dumps({"3dbbox": records}))
        dynamic = sum(r["classId"] in TO_CODA.values() for r in records)
        print(f"{frame}: {len(records)} boxes ({dynamic} dynamic) in {time.time() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
