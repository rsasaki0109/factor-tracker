#!/usr/bin/env python3
r"""Adds appearance (ReID) embeddings to the LiDAR detections of an engine
JSONL recording.

For every LiDAR batch the 3D detection boxes are projected into the camera
image with the recording's own calibration (setup.json) and matched to the
camera detections of the nearest camera batch (within --max-camera-dt) by 2D
IoU (one-to-one, greedy). The
matched camera box is cropped from the image and embedded with a frozen
image backbone (default: DINOv2 ViT-S/14 from timm, no ReID fine-tuning).
Crops more than --max-occlusion covered by other camera boxes that are closer
to the camera (lower bottom edge) are not embedded: in crowds they mostly show
the person in front.
The embedding is written to the detection's `appearance` field; detections
without a confident match get none, and the tracker then falls back to
position alone for them.

usage: embed_detections.py --seq-dir coda-seq/20 \
           --image-pattern 'CODa/2d_rect/cam0/20/2d_rect_cam0_20_{frame}.png' --out coda-seq-reid/20

`{frame}` is the LiDAR batch `sequence` number (the dataset frame index for
the KITTI and CODa converters). setup.json and truth.jsonl are copied to
--out unchanged. Requires PyTorch, timm, NumPy and Pillow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def mat4(values) -> np.ndarray:
    return np.array(values, dtype=float).reshape(4, 4)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def occlusion(box, others) -> float:
    """Fraction of `box` covered by boxes in `others` whose bottom edge is lower
    in the image (closer to the camera on flat ground)."""
    x1, y1, x2, y2 = box
    if x2 - x1 < 1 or y2 - y1 < 1:
        return 0.0
    gx, gy = np.meshgrid(np.linspace(x1, x2, 24), np.linspace(y1, y2, 48))
    covered = np.zeros_like(gx, dtype=bool)
    for o in others:
        if o[3] > y2:
            covered |= (gx >= o[0]) & (gx <= o[2]) & (gy >= o[1]) & (gy <= o[3])
    return float(covered.mean())


class Projector:
    def __init__(self, setup: dict):
        lidar_to_ref = mat4(setup["lidar_to_reference"])
        self.lidar_to_cam = np.linalg.inv(mat4(setup["camera_to_reference"])) @ lidar_to_ref
        self.k = setup["intrinsics"]

    def box(self, detection: dict):
        """2D bounding box of the projected 3D box, or None if not fully in front."""
        center = np.array(detection["position"])
        shape = detection.get("shape") or {}
        corners = np.array(shape["corners"]).reshape(8, 3) + center if shape.get("corners") else center[None]
        cam = (self.lidar_to_cam[:3, :3] @ corners.T).T + self.lidar_to_cam[:3, 3]
        if np.any(cam[:, 2] < 0.5):
            return None
        u = self.k["fx"] * cam[:, 0] / cam[:, 2] + self.k["skew"] * cam[:, 1] / cam[:, 2] + self.k["cx"]
        v = self.k["fy"] * cam[:, 1] / cam[:, 2] + self.k["cy"]
        x1, y1 = max(u.min(), 0.0), max(v.min(), 0.0)
        x2, y2 = min(u.max(), self.k["image_width"] - 1.0), min(v.max(), self.k["image_height"] - 1.0)
        return (x1, y1, x2, y2) if x2 - x1 > 2 and y2 - y1 > 2 else None


class Embedder:
    def __init__(self, model: str, height: int, width: int, batch: int):
        import timm

        self.model = timm.create_model(model, pretrained=True, num_classes=0, dynamic_img_size=True).eval()
        config = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(config["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(config["std"]).view(1, 3, 1, 1)
        self.size = (width, height)
        self.batch = batch

    @torch.no_grad()
    def __call__(self, crops: list[Image.Image]) -> np.ndarray:
        out = []
        for start in range(0, len(crops), self.batch):
            chunk = crops[start : start + self.batch]
            x = torch.from_numpy(np.stack([np.asarray(c.resize(self.size, Image.BILINEAR)) for c in chunk]))
            x = (x.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
            out.append(self.model(x).numpy())
        return np.concatenate(out) if out else np.zeros((0, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-dir", type=Path, required=True, help="directory with setup.json, inputs.jsonl, truth.jsonl")
    parser.add_argument("--image-pattern", required=True, help="image path with {frame}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--crop-size", type=int, nargs=2, default=(140, 70), metavar=("H", "W"))
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument("--min-height", type=float, default=24.0, help="minimum crop height in pixels")
    parser.add_argument("--max-occlusion", type=float, default=1.0,
                        help="skip crops covered more than this by closer camera boxes (1 keeps all)")
    parser.add_argument("--max-camera-dt", type=float, default=0.06,
                        help="seconds; use the nearest camera batch within this of the LiDAR batch")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    setup = json.loads((args.seq_dir / "setup.json").read_text())
    projector = Projector(setup)
    embedder = Embedder(args.model, *args.crop_size, args.batch)
    records = [json.loads(line) for line in open(args.seq_dir / "inputs.jsonl")]
    cameras = sorted((r["event_time_ns"], r) for r in records if r["type"] == "camera_batch")
    camera_times = np.array([t for t, _ in cameras], dtype=np.int64)

    started, embedded, total, occluded = time.time(), 0, 0, 0
    for record in records:
        if record["type"] != "lidar_batch":
            continue
        camera = None
        if len(camera_times):
            k = int(np.argmin(np.abs(camera_times - record["event_time_ns"])))
            if abs(camera_times[k] - record["event_time_ns"]) <= args.max_camera_dt * 1e9:
                camera = cameras[k][1]
        detections = record["detections"]
        total += len(detections)
        if camera is None or not camera["detections"]:
            continue
        projected = [projector.box(d) for d in detections]
        pairs = sorted(
            (
                (iou(p, c["bbox"]), i, j)
                for i, p in enumerate(projected) if p is not None
                for j, c in enumerate(camera["detections"])
                if detections[i].get("class") is None or c.get("class") is None or detections[i]["class"] == c["class"]
            ),
            reverse=True,
        )
        used_lidar, used_camera, matches = set(), set(), []
        for overlap, i, j in pairs:
            if overlap < args.min_iou:
                break
            if i in used_lidar or j in used_camera:
                continue
            box = camera["detections"][j]["bbox"]
            if box[3] - box[1] < args.min_height:
                continue
            others = [c["bbox"] for k, c in enumerate(camera["detections"]) if k != j]
            if occlusion(box, others) > args.max_occlusion:
                occluded += 1
                continue
            used_lidar.add(i)
            used_camera.add(j)
            matches.append((i, box))
        if not matches:
            continue
        image = Image.open(args.image_pattern.format(frame=record["sequence"])).convert("RGB")
        crops = [image.crop(tuple(int(round(v)) for v in box)) for _, box in matches]
        features = embedder(crops)
        for (i, _), feature in zip(matches, features):
            detections[i]["appearance"] = [round(float(v), 5) for v in feature]
        embedded += len(matches)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "inputs.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in records)
    for name in ("setup.json", "truth.jsonl"):
        if (args.seq_dir / name).exists():
            shutil.copyfile(args.seq_dir / name, args.out / name)
    print(f"embedded {embedded} of {total} LiDAR detections ({occluded} occluded crops skipped) in {time.time() - started:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
