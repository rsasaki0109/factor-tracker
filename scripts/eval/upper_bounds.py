#!/usr/bin/env python3
"""Perfect-association upper bound on fixed detections.

Every LiDAR detection gets the identity of the truth object it matches (per
frame, Hungarian on centre distance, within --max-distance), so association
is perfect and only the detections limit the score. Unmatched detections are
kept as one-scan tracks (false positives, as a tracker would report them) or,
with --drop-unmatched, removed. Writes lidar_frame records that
mot_metrics.py scores like any tracker output.

usage: upper_bounds.py <window-dir> <out.jsonl> [--drop-unmatched]
  window-dir  setup.json, inputs.jsonl, truth.jsonl (scripts/eval/coda_tracker_eval.sh)
"""

import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("window")
    ap.add_argument("out")
    ap.add_argument("--max-distance", type=float, default=1.0)
    ap.add_argument("--drop-unmatched", action="store_true")
    a = ap.parse_args()

    setup = json.load(open(f"{a.window}/setup.json"))
    lidar_to_ref = np.array(setup["lidar_to_reference"]).reshape(4, 4)
    poses = {p["timestamp_ns"]: np.array(p["transform"]).reshape(4, 4) @ lidar_to_ref for p in setup["poses"]}
    truth = {}
    for line in open(f"{a.window}/truth.jsonl"):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            truth[r["timestamp_ns"]] = [(o["truth_id"], np.array(o["position"][:3], float)) for o in r["objects"]]

    next_fp_id = 10**9
    with open(a.out, "w") as out:
        for line in open(f"{a.window}/inputs.jsonl"):
            r = json.loads(line)
            if r.get("type") != "lidar_batch":
                continue
            t = r["event_time_ns"]
            T = poses[t]
            dets = [(T @ np.append(np.array(d["position"][:3], float), 1.0))[:3] for d in r["detections"]]
            objs = truth.get(t, [])
            ids = [None] * len(dets)
            if dets and objs:
                dist = np.linalg.norm(np.array([p for _, p in objs])[:, None] - np.array(dets)[None], axis=2)
                for i, j in zip(*linear_sum_assignment(np.where(dist <= a.max_distance, dist, 1e6))):
                    if dist[i, j] <= a.max_distance:
                        ids[j] = objs[i][0]
            tracks = []
            for pos, track_id in zip(dets, ids):
                if track_id is None:
                    if a.drop_unmatched:
                        continue
                    track_id, next_fp_id = next_fp_id, next_fp_id + 1
                tracks.append({"track_id": track_id, "lifecycle": "confirmed", "position": [float(x) for x in pos]})
            out.write(json.dumps({"type": "lidar_frame", "timestamp_ns": t, "tracks": tracks}) + "\n")


if __name__ == "__main__":
    main()
