#!/usr/bin/env python3
r"""LiDAR/camera agreement for recordings without 3D ground truth.

For every LiDAR batch in an engine JSONL recording, the LiDAR detections are
projected into the camera image of the nearest camera batch (within
--max-dt) and matched one-to-one to the camera detections by 2D IoU. Reported
per class, for objects whose projected box is fully inside the image and
closer than --max-range:

  camera-confirmed recall     matched camera boxes / camera boxes
  camera-confirmed precision  matched LiDAR boxes / LiDAR boxes in view

These are agreement rates between two detectors, not accuracy: both can miss
the same person, and a camera box that no LiDAR box explains counts as a LiDAR
miss even if the camera detection is wrong. Camera boxes shorter than
--min-height pixels are ignored (far people the LiDAR barely samples).

usage: camera_consistency.py --seq-dir gt-seq/eth2 [--class 2] [--score 0.3 0.4 0.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-dir", type=Path, required=True)
    parser.add_argument("--class", dest="cls", type=int, default=2, help="engine class id (2 = pedestrian)")
    parser.add_argument("--score", type=float, nargs="+", default=[0.3, 0.4, 0.5], help="LiDAR score thresholds")
    parser.add_argument("--camera-score", type=float, default=0.4)
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument("--min-height", type=float, default=80.0)
    parser.add_argument("--max-range", type=float, default=15.0)
    parser.add_argument("--max-dt", type=float, default=0.06, help="seconds")
    args = parser.parse_args()

    setup = json.loads((args.seq_dir / "setup.json").read_text())
    lidar_to_cam = np.linalg.inv(np.array(setup["camera_to_reference"]).reshape(4, 4)) @ np.array(
        setup["lidar_to_reference"]).reshape(4, 4)
    k = setup["intrinsics"]
    records = [json.loads(line) for line in open(args.seq_dir / "inputs.jsonl")]
    cameras = sorted((r["event_time_ns"], r) for r in records if r["type"] == "camera_batch")
    camera_times = np.array([t for t, _ in cameras])

    def project(detection):
        corners = np.array(detection["shape"]["corners"]).reshape(8, 3) + np.array(detection["position"])
        cam = corners @ lidar_to_cam[:3, :3].T + lidar_to_cam[:3, 3]
        if np.any(cam[:, 2] < 0.5):
            return None
        u = k["fx"] * cam[:, 0] / cam[:, 2] + k["cx"]
        v = k["fy"] * cam[:, 1] / cam[:, 2] + k["cy"]
        box = (u.min(), v.min(), u.max(), v.max())
        inside = box[0] >= 0 and box[1] >= 0 and box[2] < k["image_width"] and box[3] < k["image_height"]
        return box if inside else None

    print(f"class {args.cls}, range < {args.max_range} m, camera boxes >= {args.min_height} px, IoU >= {args.min_iou}")
    for threshold in args.score:
        cam_total = cam_hit = lidar_total = lidar_hit = 0
        for r in records:
            if r["type"] != "lidar_batch" or not len(camera_times):
                continue
            j = int(np.argmin(np.abs(camera_times - r["event_time_ns"])))
            if abs(camera_times[j] - r["event_time_ns"]) > args.max_dt * 1e9:
                continue
            camera = [c["bbox"] for c in cameras[j][1]["detections"]
                      if c.get("class") == args.cls and c["score"] >= args.camera_score
                      and c["bbox"][3] - c["bbox"][1] >= args.min_height]
            lidar = []
            for d in r["detections"]:
                if d.get("class") != args.cls or d["score"] < threshold:
                    continue
                if np.hypot(*d["position"][:2]) > args.max_range:
                    continue
                box = project(d)
                if box is not None:
                    lidar.append(box)
            pairs = sorted(((iou(a, b), i, j2) for i, a in enumerate(lidar) for j2, b in enumerate(camera)), reverse=True)
            used_l, used_c = set(), set()
            for overlap, i, j2 in pairs:
                if overlap < args.min_iou:
                    break
                if i not in used_l and j2 not in used_c:
                    used_l.add(i)
                    used_c.add(j2)
            cam_total += len(camera)
            cam_hit += len(used_c)
            lidar_total += len(lidar)
            lidar_hit += len(used_l)
        print(f"  LiDAR score >= {threshold}: camera-confirmed recall {cam_hit / max(cam_total, 1):.2f} "
              f"({cam_hit}/{cam_total}), precision {lidar_hit / max(lidar_total, 1):.2f} ({lidar_hit}/{lidar_total})")


if __name__ == "__main__":
    main()
