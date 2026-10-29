#!/usr/bin/env python3
"""nuScenes tracking benchmark: detections in the nuScenes detection format
(e.g. the MEGVII results, https://www.nuscenes.org/data/detection-megvii.zip)
-> engine inputs, and engine outputs -> the nuScenes tracking result format
for the official evaluation (nuscenes-devkit, AMOTA).

  inputs <dataroot> <detections.json> <out> [--category pedestrian]
      One directory per validation scene: setup.json (tracking in the global
      frame, where the detections already are; no ego poses needed),
      inputs.jsonl (one lidar_batch per sample, at the sample's timestamp)
      and boxes.jsonl (each detection's size, rotation, velocity and score,
      in the same order, for `results`). Only the metadata tables of
      v1.0-trainval are read (scene.json, sample.json).

  results <scenes> <engine-output-name> <out.json> [--coast-scans N]
      Engine outputs <scenes>/<scene>/<engine-output-name> -> one result
      file: each reported track at each sample becomes a box at the track's
      position with the size and rotation of its last detection, the
      track's velocity and its confidence as `tracking_score`. Every
      sample of every scene is listed (empty when nothing is reported).

  splits <scenes> <out-dir>
      tune.txt and test.txt: the validation scenes in name order, alternately.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
LIDAR_SENSOR = 1
PEDESTRIAN = 2  # engine class id, as in the KITTI and CODa adapters
IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
SETUP = {
    "tracking_frame": "global",
    "lidar_frame": "global",
    "camera_frame": "unused_camera",
    "lidar_to_reference": IDENTITY,
    "camera_to_reference": IDENTITY,
    "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "skew": 0.0, "image_width": 1, "image_height": 1},
    "near_plane": 0.1,
}


def val_scenes() -> list[str]:
    from nuscenes.utils.splits import create_splits_scenes

    return create_splits_scenes()["val"]


def scene_samples(dataroot: Path) -> dict[str, list[dict]]:
    """Scene name -> its samples (token, timestamp in microseconds) in order."""
    tables = dataroot / "v1.0-trainval"
    samples = {s["token"]: s for s in json.loads((tables / "sample.json").read_text())}
    out = {}
    for scene in json.loads((tables / "scene.json").read_text()):
        token, ordered = scene["first_sample_token"], []
        while token:
            ordered.append(samples[token])
            token = samples[token]["next"]
        out[scene["name"]] = ordered
    return out


def inputs(a: argparse.Namespace) -> None:
    detections = json.loads(a.detections.read_text())["results"]
    by_scene = scene_samples(a.dataroot)
    variance = a.position_std**2
    for name in val_scenes():
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "setup.json").write_text(json.dumps(SETUP, indent=1))
        lines, boxes = [], []
        for k, sample in enumerate(by_scene[name]):
            t = sample["timestamp"] * 1000
            dets = [b for b in detections.get(sample["token"], [])
                    if b["detection_name"] == a.category and b["detection_score"] >= a.score_threshold]
            lines.append(json.dumps({
                "type": "lidar_batch", "schema_version": SCHEMA_VERSION, "sensor_id": LIDAR_SENSOR,
                "sequence": k, "event_time_ns": t, "arrival_time_ns": t, "frame_id": "global",
                "detections": [{"position": b["translation"],
                                "covariance": [variance, 0, 0, 0, variance, 0, 0, 0, variance],
                                "class": PEDESTRIAN, "score": b["detection_score"],
                                **({"velocity": b["velocity"][:2]} if all(math.isfinite(v) for v in b["velocity"][:2]) else {})}
                               for b in dets]}))
            boxes.append(json.dumps({"sample_token": sample["token"], "timestamp_ns": t,
                                     "boxes": [{k2: b[k2] for k2 in ("size", "rotation", "velocity", "detection_score")}
                                               for b in dets]}))
        (d / "inputs.jsonl").write_text("\n".join(lines) + "\n")
        (d / "boxes.jsonl").write_text("\n".join(boxes) + "\n")
    print(f"{a.out}: {len(val_scenes())} scenes")


def results(a: argparse.Namespace) -> None:
    out: dict[str, list[dict]] = {}
    lifecycles = {"confirmed", "tentative"}
    for d in sorted(p for p in a.scenes.iterdir() if (p / "boxes.jsonl").exists()):
        samples = [json.loads(line) for line in (d / "boxes.jsonl").read_text().splitlines()]
        sample_at = {s["timestamp_ns"]: s for s in samples}
        for s in samples:
            out[s["sample_token"]] = []
        last_box: dict[int, dict] = {}
        for line in open(d / a.engine_output):
            r = json.loads(line)
            if r.get("type") != "lidar_frame":
                continue
            s = sample_at[r["timestamp_ns"]]
            if len(s["boxes"]) != len(r["observations"]):
                raise SystemExit(f"{d.name}: {len(s['boxes'])} boxes but {len(r['observations'])} observations")
            updated = set()
            for x in r["associations"]:
                if x["track_id"] is not None:
                    last_box[x["track_id"]] = s["boxes"][x["observation_index"]]
                    updated.add(x["track_id"])
            for t in r["tracks"]:
                if t["lifecycle"] not in lifecycles or t["track_id"] not in last_box:
                    continue
                if t["track_id"] not in updated and t["age_since_spatial_update_ns"] > a.coast_scans * 500_000_000 + 1_000_000:
                    continue
                b = last_box[t["track_id"]]
                score = t.get("confidence") if a.score == "confidence" else b["detection_score"]
                out[s["sample_token"]].append({
                    "sample_token": s["sample_token"],
                    "translation": t["position"][:3],
                    "size": b["size"],
                    "rotation": b["rotation"],
                    "velocity": t["velocity"][:2],
                    "tracking_id": f"{d.name}_{t['track_id']}",
                    "tracking_name": a.category,
                    "tracking_score": score,
                })
    meta = {"use_camera": False, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False}
    a.out.write_text(json.dumps({"meta": meta, "results": out}))
    print(f"{a.out}: {sum(len(v) for v in out.values())} boxes in {len(out)} samples")


def split(a: argparse.Namespace) -> None:
    names = sorted(p.name for p in a.scenes.iterdir() if (p / "boxes.jsonl").exists())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tune.txt").write_text("\n".join(names[0::2]) + "\n")
    (a.out / "test.txt").write_text("\n".join(names[1::2]) + "\n")
    print(f"{len(names[0::2])} tune, {len(names[1::2])} test scenes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inputs")
    i.add_argument("dataroot", type=Path)
    i.add_argument("detections", type=Path)
    i.add_argument("out", type=Path)
    i.add_argument("--category", default="pedestrian")
    i.add_argument("--score-threshold", type=float, default=0.0)
    i.add_argument("--position-std", type=float, default=0.1, help="detection position standard deviation (m)")
    r = sub.add_parser("results")
    r.add_argument("scenes", type=Path)
    r.add_argument("engine_output")
    r.add_argument("out", type=Path)
    r.add_argument("--category", default="pedestrian")
    r.add_argument("--coast-scans", type=int, default=0, help="also write tracks that missed up to this many samples")
    r.add_argument("--score", choices=["confidence", "detection"], default="confidence")
    s = sub.add_parser("splits")
    s.add_argument("scenes", type=Path)
    s.add_argument("out", type=Path)
    a = ap.parse_args()
    {"inputs": inputs, "results": results, "splits": split}[a.command](a)


if __name__ == "__main__":
    main()
