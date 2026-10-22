"""Pre-labels v2: smoothed frame-to-frame linking + tracklet stitching. usage: link2.py seq out.json"""
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
seq, out = sys.argv[1], sys.argv[2]
setup = json.load(open(f"{seq}/setup.json"))
pt = np.array([p["timestamp_ns"] for p in setup["poses"]]); pm = [np.array(p["transform"]).reshape(4, 4) for p in setup["poses"]]
dets = []
for l in open(f"{seq}/inputs.jsonl"):
    r = json.loads(l)
    if r["type"] != "lidar_batch": continue
    T = pm[int(np.argmin(np.abs(pt - r["event_time_ns"])))]
    for k, d in enumerate(r["detections"]):
        p = np.array(d["position"]); w = T[:3, :3] @ p + T[:3, 3]
        dets.append(dict(frame=r["sequence"], k=k, base=p.tolist(), world=w.tolist(), score=d["score"], id=None))
frames = sorted({d["frame"] for d in dets}); byf = {f: [d for d in dets if d["frame"] == f] for f in frames}
# pass 1: strict frame-to-frame links (gap <= 2), smoothed position estimate
tracks, nid = [], 0
for f in frames:
    cur = byf[f]; live = [t for t in tracks if f - t["f"][-1] <= 2]
    C = np.full((len(live), len(cur)), 1e6)
    for i, t in enumerate(live):
        est = np.mean(t["xy"][-3:], axis=0)
        v = (np.mean(t["xy"][-3:], 0) - np.mean(t["xy"][-6:-3], 0)) / 3 if len(t["xy"]) >= 6 else np.zeros(2)
        pred = est + np.clip(v, -0.2, 0.2) * (f - t["f"][-1] + 1)
        for j, d in enumerate(cur):
            c = np.linalg.norm(pred - d["world"][:2])
            if c < 0.45: C[i, j] = c
    rows, cols = linear_sum_assignment(C) if len(live) and len(cur) else ([], [])
    used = set()
    for i, j in zip(rows, cols):
        if C[i, j] < 1e5:
            live[i]["f"].append(f); live[i]["xy"].append(np.array(cur[j]["world"][:2])); live[i]["d"].append(cur[j]); used.add(j)
    for j, d in enumerate(cur):
        if j not in used:
            nid += 1; tracks.append(dict(id=nid, f=[f], xy=[np.array(d["world"][:2])], d=[d]))
# pass 2: stitch tracklet ends to starts (gap 1..15 frames), unique best, distance <= 0.3 + 0.06*gap
changed = True
while changed:
    changed = False
    ends = [t for t in tracks if len(t["f"]) >= 2]
    cands = []
    for a in ends:
        for b in tracks:
            if b is a: continue
            gap = b["f"][0] - a["f"][-1]
            if 1 <= gap <= 15:
                dist = np.linalg.norm(np.mean(a["xy"][-3:], 0) - np.mean(b["xy"][:3], 0))
                if dist <= 0.3 + 0.06 * gap: cands.append((dist, id(a), id(b), a, b))
    cands.sort(key=lambda c: c[0]); ua, ub = set(), set()
    for dist, ia, ib, a, b in cands:
        if ia in ua or ib in ub or ia in ub or ib in ua: continue
        a["f"] += b["f"]; a["xy"] += b["xy"]; a["d"] += b["d"]; b["f"] = []; ua.add(ia); ub.add(ib); changed = True
    tracks = [t for t in tracks if t["f"]]
tracks.sort(key=lambda t: (t["f"][0], t["xy"][0][0]))
for n, t in enumerate(tracks, 1):
    for d in t["d"]: d["id"] = n
json.dump(dets, open(out, "w"))
L = np.array([len(t["f"]) for t in tracks])
print(f"{len(tracks)} tracks; >=20 frames {np.sum(L >= 20)}, 5-19 {np.sum((L >= 5) & (L < 20))}, 2-4 {np.sum((L >= 2) & (L < 5))}, singletons {np.sum(L == 1)}")
