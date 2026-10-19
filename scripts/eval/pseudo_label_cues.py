#!/usr/bin/env python3
r"""Separability of identity cues on a recording without identity labels.

Pseudo-labels: detections are chained frame to frame (mutual nearest within
0.5 m in the world frame) while no other detection is within --isolation
metres; chains of at least five frames are taken as one person each. Pairs
1-10 frames apart in the same chain count as "same"; pairs from two chains
that coexist in time and are closer than --near metres count as "different".
For each cue the ROC AUC of the pair distance is printed (0.5 = no
information). The protocol picks easy positives (isolated people), so the
numbers are higher than with real identity labels; compare cues with each
other, and calibrate on a labelled recording (CODa) with the same protocol.

Cues: shape (point fraction and spread in 8 height bins of 0.25 m, top
height), refl (median log reflectivity per height bin + reflectivity
histogram; needs a scan format with intensity), refl+shape, cam (the
`appearance` embedding from scripts/reid/embed_detections.py).

usage: pseudo_label_cues.py --seq-dir gt-reid/eth2 --scan-pattern 'gt-seq/eth2/scans/{frame:06d}.bin' \
           [--isolation 1.2] [--near 2.5] [--cues shape,refl,refl+shape,cam]
"""

import argparse
import json

from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--seq-dir", type=Path, required=True)
parser.add_argument("--scan-pattern", required=True, help="float32 x y z intensity scans, with {frame}")
parser.add_argument("--isolation", type=float, default=1.2)
parser.add_argument("--near", type=float, default=2.5)
parser.add_argument("--cues", default="shape,refl,refl+shape,cam")
args = parser.parse_args()
d, scans, ISO, NEAR = args.seq_dir, args.scan_pattern, args.isolation, args.near
setup = json.loads((d / "setup.json").read_text())
pt = np.array([p["timestamp_ns"] for p in setup["poses"]])
pm = [np.array(p["transform"]).reshape(4, 4) for p in setup["poses"]]
def to_world(t, x):
    T = pm[int(np.argmin(np.abs(pt - t)))]; return T[:3, :3] @ x + T[:3, 3]

def descriptors(points, det):
    c = np.array(det["shape"]["corners"]).reshape(8, 3)
    yaw = np.arctan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0])
    L = np.linalg.norm(c[1, :2] - c[0, :2]); W = np.linalg.norm(c[2, :2] - c[1, :2]); H = c[4, 2] - c[0, 2]
    dd = points[:, :3] - det["position"]; cs, sn = np.cos(-yaw), np.sin(-yaw)
    lx = cs * dd[:, 0] - sn * dd[:, 1]; ly = sn * dd[:, 0] + cs * dd[:, 1]
    m = (np.abs(lx) < L / 2 + 0.1) & (np.abs(ly) < W / 2 + 0.1) & (np.abs(dd[:, 2]) < H / 2 + 0.1)
    if m.sum() < 20: return None, None
    z = dd[m, 2] + H / 2; xy = np.c_[lx[m], ly[m]]; refl = points[m, 3]
    shape, rbin = [], []
    for a in np.linspace(0, 1.75, 8):
        k = (z >= a) & (z < a + 0.25)
        shape += [k.mean(), np.linalg.norm(xy[k] - xy[k].mean(0), axis=1).mean() if k.sum() > 2 else 0.0]
        rbin.append(np.median(refl[k]) if k.sum() > 2 else np.nan)
    shape.append(np.percentile(z, 98))
    rbin = np.array(rbin); rbin = np.where(np.isnan(rbin), np.nanmedian(refl), rbin)
    hist = np.histogram(np.log1p(refl), bins=8, range=(0, np.log1p(255)))[0] / len(refl)
    return np.array(shape), np.r_[np.log1p(rbin), hist]

obs = []  # frame, world xy, shape, refl, cam
for line in open(d / "inputs.jsonl"):
    r = json.loads(line)
    if r["type"] != "lidar_batch": continue
    pts = np.fromfile(scans.format(frame=r["sequence"]), dtype=np.float32).reshape(-1, 4)
    for det in r["detections"]:
        s, f = descriptors(pts, det)
        w = to_world(r["event_time_ns"], np.array(det["position"]))
        cam = det.get("appearance"); cam = None if cam is None else np.array(cam) / np.linalg.norm(cam)
        obs.append(dict(frame=r["sequence"], xy=w[:2], shape=s, refl=f, cam=cam, rng=np.hypot(*det["position"][:2])))
frames = {}
for i, o in enumerate(obs): frames.setdefault(o["frame"], []).append(i)

# Tracklets: mutual nearest neighbours within 0.5 m between consecutive frames, both isolated (no other detection within 2 m).
def isolated(i):
    return all(j == i or np.linalg.norm(obs[i]["xy"] - obs[j]["xy"]) > ISO for j in frames[obs[i]["frame"]])
chain = {}; next_chain = 0
for f in sorted(frames):
    for i in frames[f]:
        if i in chain or not isolated(i): continue
        chain[i] = next_chain; cur = i; next_chain += 1
        while (obs[cur]["frame"] + 1) in frames:
            cand = frames[obs[cur]["frame"] + 1]
            dist = [np.linalg.norm(obs[cur]["xy"] - obs[j]["xy"]) for j in cand]
            j = cand[int(np.argmin(dist))]
            back = min(np.linalg.norm(obs[j]["xy"] - obs[k]["xy"]) for k in frames[obs[cur]["frame"]])
            if min(dist) > 0.5 or back < min(dist) - 1e-9 or not isolated(j) or j in chain: break
            chain[j] = chain[i]; cur = j
members = {}
for i, c in chain.items(): members.setdefault(c, []).append(i)
long = {c: sorted(v, key=lambda i: obs[i]["frame"]) for c, v in members.items() if len(v) >= 5}
print(f"{len(obs)} detections, {len(long)} isolated tracklets of >= 5 frames ({sum(map(len, long.values()))} detections)")

# Same: within a tracklet, 1-10 frames apart. Different: two tracklets that coexist in a frame, members 1-10 frames apart.
span = {c: (obs[v[0]]["frame"], obs[v[-1]]["frame"]) for c, v in long.items()}
same, diff = [], []
for c, v in long.items():
    for a in range(len(v)):
        for b in range(a + 1, len(v)):
            if 1 <= obs[v[b]]["frame"] - obs[v[a]]["frame"] <= 10: same.append((v[a], v[b]))
cs = list(long)
for x in range(len(cs)):
    for y in range(x + 1, len(cs)):
        A, B = cs[x], cs[y]
        if span[A][1] < span[B][0] or span[B][1] < span[A][0]: continue
        for i in long[A]:
            for j in long[B]:
                if 1 <= abs(obs[j]["frame"] - obs[i]["frame"]) <= 10 and np.linalg.norm(obs[i]["xy"] - obs[j]["xy"]) < NEAR: diff.append((i, j))
print(f"pairs: same {len(same)}, different {len(diff)}")

def auc(s, d):
    s, d = np.asarray(s), np.asarray(d)
    allv = np.r_[s, d]; ranks = allv.argsort().argsort() + 1
    return (len(s) * len(d) + len(s) * (len(s) + 1) / 2 - ranks[:len(s)].sum()) / (len(s) * len(d))
feats = {k: np.array([o[k] for o in obs if o[k] is not None]) for k in ("shape", "refl")}
sd = {k: v.std(0) + 1e-6 for k, v in feats.items()}
def cue(k):
    if k == "cam": return lambda i, j: None if obs[i]["cam"] is None or obs[j]["cam"] is None else 1 - obs[i]["cam"] @ obs[j]["cam"]
    if k == "refl+shape": return lambda i, j: None if obs[i]["refl"] is None or obs[j]["refl"] is None else \
        np.linalg.norm((obs[i]["refl"] - obs[j]["refl"]) / sd["refl"]) + np.linalg.norm((obs[i]["shape"] - obs[j]["shape"]) / sd["shape"])
    return lambda i, j: None if obs[i][k] is None or obs[j][k] is None else np.linalg.norm((obs[i][k] - obs[j][k]) / sd[k])
for name in args.cues.split(","):
    fn = cue(name)
    for lo, hi in ((0, 8), (8, 30)):
        s = [v for i, j in same if lo <= obs[i]["rng"] < hi and (v := fn(i, j)) is not None]
        dd = [v for i, j in diff if lo <= obs[i]["rng"] < hi and (v := fn(i, j)) is not None]
        if len(s) > 20 and len(dd) > 20:
            print(f"  {name:11s} range {lo:2d}-{hi:2d} m: AUC {auc(s, dd):.3f} (same {len(s)}, different {len(dd)})")
