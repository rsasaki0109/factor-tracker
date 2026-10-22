#!/usr/bin/env python3
"""Can box size / heading tell neighbouring people apart where position fails?

usage: shape_cue.py <window-dir>...   (window dirs written by coda_tracker_eval.sh)

For each person A at scan t+1 with a neighbour B detected within 1.5 m of A's
detection, compare A's template (mean of its last 5 detections) with A's and B's
detections at t+1: pairwise accuracy = P(feature distance to A < to B)."""
import json, sys, math
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import linear_sum_assignment

def load(d):
    dets, truth = {}, {}
    setup = json.load(open(f"{d}/setup.json"))
    l2r = np.array(setup["lidar_to_reference"]).reshape(4, 4)
    poses = {p["timestamp_ns"]: np.array(p["transform"]).reshape(4, 4) @ l2r for p in setup["poses"]}
    for l in open(f"{d}/inputs.jsonl"):
        r = json.loads(l)
        if r.get("type") == "lidar_batch":
            T = poses[r["event_time_ns"]]
            heading = math.atan2(T[1, 0], T[0, 0])
            rows = []
            for o in r["detections"]:
                s, c = o["shape"], o["shape"]["corners"]
                # corners 0 and 3 differ along the box length (sensor frame); heading in the world frame
                yaw = math.atan2(c[1] - c[10], c[0] - c[9]) + heading
                pos = (T @ np.array([*o["position"][:3], 1.0]))[:2]
                rows.append((pos, s["height"], s["width"], s["length"], yaw))
            dets[r["event_time_ns"]] = rows
    for l in open(f"{d}/truth.jsonl"):
        r = json.loads(l)
        if r.get("type") == "truth3d":
            truth[r["timestamp_ns"]] = [(o["truth_id"], np.array(o["position"][:2])) for o in r["objects"]]
    return dets, truth

def angle_diff(a, b):  # boxes are symmetric under 180 deg
    x = abs((a - b) % math.pi)
    return min(x, math.pi - x)

res = defaultdict(list)
for d in sys.argv[1:]:
    dets, truth = load(d)
    hist = defaultdict(lambda: deque(maxlen=5))
    prev_pos = {}
    for t in sorted(dets):
        rows, objs = dets[t], truth.get(t, [])
        if not rows or not objs:
            continue
        D = np.linalg.norm(np.array([o[1] for o in objs])[:, None] - np.array([r[0] for r in rows])[None], axis=2)
        r_i, c_i = linear_sum_assignment(D)
        match = {objs[r][0]: rows[c] for r, c in zip(r_i, c_i) if D[r, c] <= 0.5}
        for a, det_a in match.items():
            if len(hist[a]) >= 3:
                tmpl = np.array([[h, w, l] for _, h, w, l, _ in hist[a]]).mean(0)
                yaws = [y for *_, y in hist[a]]
                for b, det_b in match.items():
                    if b == a or np.linalg.norm(det_a[0] - det_b[0]) > 1.5:
                        continue
                    da, db = np.array(det_a[1:4]), np.array(det_b[1:4])
                    res["height"].append(abs(da[0] - tmpl[0]) < abs(db[0] - tmpl[0]))
                    res["width"].append(abs(da[1] - tmpl[1]) < abs(db[1] - tmpl[1]))
                    res["length"].append(abs(da[2] - tmpl[2]) < abs(db[2] - tmpl[2]))
                    res["size (h,w,l)"].append(np.linalg.norm(da - tmpl) < np.linalg.norm(db - tmpl))
                    res["heading"].append(angle_diff(det_a[4], yaws[-1]) < angle_diff(det_b[4], yaws[-1]))
                    if a in prev_pos:
                        pos_ok = np.linalg.norm(det_a[0] - prev_pos[a]) < np.linalg.norm(det_b[0] - prev_pos[a])
                        res["position (last)"].append(pos_ok)
                        if not pos_ok:
                            res["size | position wrong"].append(res["size (h,w,l)"][-1])
                            res["height | position wrong"].append(res["height"][-1])
                            res["heading | position wrong"].append(res["heading"][-1])
            hist[a].append(det_a)
            prev_pos[a] = det_a[0]
for k, v in res.items():
    print(f"{k:28s} accuracy {np.mean(v):.3f}  (n={len(v)})")
