#!/usr/bin/env python3
"""KITTI 3D MOT benchmark in AB3DMOT's protocol: the KITTI tracking validation
split (sequences 0001 0006 0008 0010 0012 0013 0014 0015 0016 0018 0019),
the PointRCNN detections AB3DMOT ships, and AB3DMOT's evaluator
(scripts/KITTI/evaluate.py: sAMOTA / MOTA / MOTP / IDS at a 3D IoU threshold).

  detections <ab3dmot-root> <out> --category Pedestrian
      AB3DMOT's data/KITTI/detection/pointrcnn_<category>_val/<seq>.txt (one
      CSV row per box: frame, type, 2D box, score, h w l x y z ry, alpha) ->
      <out>/lidar/<seq>/<frame>.txt in KITTI label format with a score, plus
      empty <out>/camera/<seq>/, the layout scripts/kitti/convert_tracking.py
      --detections-root reads. PointRCNN scores are logits; they become
      sigmoid(score), which keeps their order.

  results <sequence-dir> <engine-output.jsonl> <out.txt> --category Pedestrian
      An engine output (factor-tracker replay on the prepared sequence) ->
      one KITTI tracking result file: each confirmed track reported in a frame
      becomes `frame id type 0 0 alpha x1 y1 x2 y2 h w l x y z ry score` in
      rectified camera 0, at the track's position, with the size, heading,
      image box and score of the detection that updated it.

Every box written is checked: mapping its detection's position back through
the same transforms must give the detection's original location.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kitti_tools  # noqa: E402

VAL_SEQUENCES = ["0001", "0006", "0008", "0010", "0012", "0013", "0014", "0015", "0016", "0018", "0019"]
AB3DMOT_TYPES = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}


def detections(a: argparse.Namespace) -> None:
    source = a.ab3dmot_root / "data" / "KITTI" / "detection" / f"pointrcnn_{a.category}_val"
    for seq in VAL_SEQUENCES:
        frames: dict[int, list[str]] = {}
        for row in (source / f"{seq}.txt").read_text().splitlines():
            if not row.strip():
                continue
            v = [float(x) for x in row.split(",")]
            frame, kind, x1, y1, x2, y2, score, h, w, l, x, y, z, ry, alpha = v
            name = AB3DMOT_TYPES[int(kind)]
            prob = 1.0 / (1.0 + math.exp(-score))
            frames.setdefault(int(frame), []).append(
                f"{name} 0 0 {alpha:.6f} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} "
                f"{h:.6f} {w:.6f} {l:.6f} {x:.6f} {y:.6f} {z:.6f} {ry:.6f} {prob:.6f}")
        lidar = a.out / "lidar" / seq
        lidar.mkdir(parents=True, exist_ok=True)
        for frame, lines in frames.items():
            (lidar / f"{frame:06d}.txt").write_text("".join(line + "\n" for line in lines))
        (a.out / "camera" / seq).mkdir(parents=True, exist_ok=True)
        print(f"{seq}: {sum(len(v) for v in frames.values())} {a.category} detections in {len(frames)} frames")


def results(a: argparse.Namespace) -> None:
    calib = kitti_tools.TrackingCalib.read(a.sequence / "calib.txt")
    velo_to_cam = calib.velo_to_cam0
    setup = json.loads((a.setup or a.sequence / "setup.json").read_text())
    lidar_to_ref = np.array(setup["lidar_to_reference"]).reshape(4, 4)
    poses = {p["timestamp_ns"]: np.array(p["transform"]).reshape(4, 4) for p in setup["poses"]}
    times = [float(t) for t in (a.sequence / "times.txt").read_text().split()]
    frame_of = {int(round(t * 1e9)): k for k, t in enumerate(times)}
    lines = []
    checked = 0
    for line in open(a.output):
        r = json.loads(line)
        if r.get("type") != "lidar_frame":
            continue
        frame = frame_of[r["timestamp_ns"]]
        det_path = a.sequence / "detections_lidar" / f"{frame:06d}.txt"
        dets = [d.split() for d in det_path.read_text().splitlines() if d.strip()] if det_path.exists() else []
        if len(dets) != len(r["observations"]):
            raise SystemExit(f"frame {frame}: {len(dets)} detections but {len(r['observations'])} observations")
        # tracking frame -> rectified camera 0 at this scan
        world_to_cam = velo_to_cam @ np.linalg.inv(poses[r["timestamp_ns"]] @ lidar_to_ref)
        to_cam = lambda p: (world_to_cam @ np.append(np.asarray(p, float)[:3], 1.0))[:3]  # noqa: E731
        detection_of = {x["observation_index"]: x["track_id"] for x in r["associations"] if x["track_id"] is not None}
        by_track = {}
        for j, track in detection_of.items():
            by_track[track] = j
            back = to_cam(r["observations"][j])
            orig = np.array([float(v) for v in dets[j][11:14]])
            if np.linalg.norm(back - orig) > 1e-3:
                raise SystemExit(f"frame {frame}: detection {j} maps back to {back}, not {orig}")
            checked += 1
        for t in r["tracks"]:
            if t["lifecycle"] != "confirmed" or t["track_id"] not in by_track:
                continue
            d = dets[by_track[t["track_id"]]]
            if d[0] != a.category:
                continue
            x, y, z = to_cam(t["position"])
            # alpha (observation angle) follows the heading as seen from the camera
            ry = float(d[14])
            alpha = ry - math.atan2(x, z)
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
            lines.append(f"{frame} {t['track_id']} {a.category} 0 0 {alpha:.6f} {d[4]} {d[5]} {d[6]} {d[7]} "
                         f"{d[8]} {d[9]} {d[10]} {x:.6f} {y:.6f} {z:.6f} {ry:.6f} {float(d[15]):.6f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(line + "\n" for line in lines))
    print(f"{a.out}: {len(lines)} boxes ({checked} detections mapped back exactly)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("detections")
    d.add_argument("ab3dmot_root", type=Path)
    d.add_argument("out", type=Path)
    d.add_argument("--category", default="Pedestrian", choices=["Pedestrian", "Car", "Cyclist"])
    r = sub.add_parser("results")
    r.add_argument("sequence", type=Path, help="sequence directory written by convert_tracking.py")
    r.add_argument("output", type=Path, help="engine output of factor-tracker replay")
    r.add_argument("out", type=Path)
    r.add_argument("--setup", type=Path, help="setup.json of the replay (default: <sequence>/setup.json)")
    r.add_argument("--category", default="Pedestrian", choices=["Pedestrian", "Car", "Cyclist"])
    a = ap.parse_args()
    detections(a) if a.command == "detections" else results(a)


if __name__ == "__main__":
    main()
