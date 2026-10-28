#!/usr/bin/env python3
"""Adds ego-pose noise to a window: each scan's pose is rotated about the
sensor (yaw, normal with --yaw-deg standard deviation) and shifted (x and y,
--translation-m), independently per scan, as a jittery odometry would. The
labels live in the sensor frame, so the truth is moved with the same noisy
pose; the detections (sensor frame) are copied unchanged.

usage: pose_noise.py <window-dir> <out-dir> [--yaw-deg 1.0] [--translation-m 0.1] [--seed 1]
  window-dir  setup.json, inputs.jsonl, truth.jsonl (scripts/eval/coda_tracker_eval.sh)
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("window", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--yaw-deg", type=float, default=1.0)
    ap.add_argument("--translation-m", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    setup = json.load(open(a.window / "setup.json"))
    moved = {}  # timestamp -> noisy pose * inverse(given pose), applied to world-frame truth
    for pose in setup["poses"]:
        T = np.array(pose["transform"]).reshape(4, 4)
        yaw = np.radians(rng.normal(0.0, a.yaw_deg))
        N = np.eye(4)
        N[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        N[:2, 3] = rng.normal(0.0, a.translation_m, 2)
        noisy = T @ N
        moved[pose["timestamp_ns"]] = noisy @ np.linalg.inv(T)
        pose["transform"] = noisy.reshape(-1).tolist()
    (a.out / "setup.json").write_text(json.dumps(setup))
    lines = []
    for line in open(a.window / "truth.jsonl"):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            M = moved[r["timestamp_ns"]]
            for o in r["objects"]:
                o["position"][:3] = (M @ np.append(np.array(o["position"][:3], float), 1.0))[:3].tolist()
        lines.append(json.dumps(r))
    (a.out / "truth.jsonl").write_text("\n".join(lines) + "\n")
    shutil.copy(a.window / "inputs.jsonl", a.out / "inputs.jsonl")


if __name__ == "__main__":
    main()
