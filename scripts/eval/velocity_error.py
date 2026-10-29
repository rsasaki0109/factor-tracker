#!/usr/bin/env python3
"""Velocity error of a tracker output against labels, in m/s (horizontal).

The label velocity of a person at a scan is the slope of a straight line
fitted to their labelled positions within +-5 scans (at least 5 of them).
Confirmed tracks are matched to the labelled people per scan (Hungarian on
centre distance, within 1 m; the truth is mapped through a frame's
`pose_correction` as in mot_metrics.py), and each matched track's reported
velocity is compared with the person's. A dual output is scored as in
mot_metrics.py (smoothed frames, or --realtime).

usage: velocity_error.py <output.jsonl> <truth.jsonl> [--json] [--realtime]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mot_metrics import lidar_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output")
    ap.add_argument("truth")
    ap.add_argument("--max-distance", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="score the real-time frames of a dual output")
    a = ap.parse_args()
    truth = {}
    for line in open(a.truth):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            truth[r["timestamp_ns"]] = {o["truth_id"]: np.array(o["position"][:3], float) for o in r["objects"]}
    frames = lidar_frames(a.output, a.realtime)
    times = [f["timestamp_ns"] for f in frames]
    corrections = [np.array(f["pose_correction"]).reshape(4, 4) if f.get("pose_correction") else np.eye(4) for f in frames]

    def label_velocity(person, k):
        scans = [q for q in range(max(0, k - 5), min(len(times), k + 6)) if person in truth.get(times[q], {})]
        if len(scans) < 5:
            return None
        t = np.array([(times[q] - times[k]) * 1e-9 for q in scans])
        # positions in scan k's (corrected) frame
        C = corrections[k]
        p = np.array([(C[:3, :3] @ truth[times[q]][person] + C[:3, 3])[:2] for q in scans])
        return np.polyfit(t, p, 1)[0]

    errors = []
    for k, f in enumerate(frames):
        C = corrections[k]
        people = [(g, C[:3, :3] @ p + C[:3, 3]) for g, p in truth.get(f["timestamp_ns"], {}).items()]
        tracks = [t for t in f["tracks"] if t["lifecycle"] == "confirmed"]
        if not people or not tracks:
            continue
        d = np.linalg.norm(np.array([p for _, p in people])[:, None] - np.array([t["position"][:3] for t in tracks])[None], axis=2)
        cost = np.where(d <= a.max_distance, d, 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] >= 1e6:
                continue
            v = label_velocity(people[i][0], k)
            if v is not None:
                errors.append(float(np.linalg.norm(np.array(tracks[j]["velocity"][:2]) - v)))
    e = np.array(errors)
    result = {"matches": len(e), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "rmse": float(np.sqrt(np.mean(e**2)))}
    if a.json:
        print(json.dumps(result))
    else:
        print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in result.items()))


if __name__ == "__main__":
    main()
