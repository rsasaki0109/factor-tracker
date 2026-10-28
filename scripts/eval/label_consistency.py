#!/usr/bin/env python3
"""How far a window's labels and poses can be trusted for scoring association.

usage: label_consistency.py <window-dir>... [--coda-root CODa]
  window-dir  setup.json + truth.jsonl (scripts/eval/coda_tracker_eval.sh work
              dir, e.g. work/coda20_9330/20); --coda-root also reads the CODa
              labels of static objects for those frames

Per window:
- teleports: a labelled instance moving more than max(1 m, 3 m/s * dt)
  between two of its labelled scans (an id given to someone else);
- label tracking: a constant-velocity tracker (1 m gate) run on the labelled
  positions themselves and scored against the label ids. On consistent
  labels it is near perfect; every switch it makes is a place where the ids
  disagree with how the labelled positions move;
- static wobble (--coda-root): how much labelled static objects (poles,
  trees, bins, ...) move in the world frame, i.e. the ego-pose error plus
  label noise. Clean windows: about 0.1 m (per-instance std).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mot_metrics import clear_and_id, hota  # noqa: E402

STATIC = {"Pole", "Tree", "TrashCan", "Bench", "Bollard", "Chair", "Table", "Mailbox", "Railing"}


def truth_frames(window):
    frames = []
    for line in open(window / "truth.jsonl"):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            frames.append((r["timestamp_ns"], [(o["truth_id"], np.array(o["position"][:3], float)) for o in r["objects"]]))
    return frames


def teleports(frames):
    last, count = {}, 0
    for t, objs in frames:
        for g, p in objs:
            if g in last:
                tl, pl = last[g]
                count += np.linalg.norm((p - pl)[:2]) > max(1.0, 3.0 * (t - tl) * 1e-9)
            last[g] = (t, p)
    return int(count)


def track_labels(frames, gate=1.0, max_misses=10):
    """Tracks the labelled positions; returns the (truth, tracks) sequence mot_metrics scores."""
    tracks, next_id, seq = [], 1, []
    for t, objs in frames:
        t_s = t * 1e-9
        pred = [tr["p"] + tr["v"] * (t_s - tr["t"]) for tr in tracks]
        pairs = []
        if tracks and objs:
            d = np.linalg.norm(np.array(pred)[:, None, :2] - np.array([p for _, p in objs])[None, :, :2], axis=2)
            cost = np.where(d <= gate, d, 1e6)
            pairs = [(i, j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6]
        used = set()
        for i, j in pairs:
            tr = tracks[i]
            dt = t_s - tr["t"]
            if dt > 0:
                tr["v"] = 0.6 * tr["v"] + 0.4 * (objs[j][1] - tr["p"]) / dt
            tr.update(p=objs[j][1], t=t_s, misses=0)
            used.add(j)
        matched = {i for i, _ in pairs}
        for i, tr in enumerate(tracks):
            if i not in matched:
                tr["misses"] += 1
        for j, (_, p) in enumerate(objs):
            if j not in used:
                tracks.append(dict(id=next_id, p=p, v=np.zeros(3), t=t_s, misses=0))
                next_id += 1
        seq.append((objs, [(tr["id"], tr["p"]) for tr in tracks if tr["misses"] == 0 and tr["t"] == t_s]))
        tracks = [tr for tr in tracks if tr["misses"] <= max_misses]
    return seq


def static_wobble(window, coda_root):
    seq_name = window.name
    setup = json.load(open(window / "setup.json"))
    first = int(window.parent.name.split("_")[1])
    stds, steps = [], []
    positions = defaultdict(dict)
    for k, pose in enumerate(setup["poses"]):
        path = coda_root / "3d_bbox" / "os1" / seq_name / f"3d_bbox_os1_{seq_name}_{first + k}.json"
        if not path.exists():
            continue
        T = np.array(pose["transform"]).reshape(4, 4)
        for b in json.loads(path.read_text())["3dbbox"]:
            if b["classId"].replace(" ", "") in STATIC:
                positions[b["instanceId"]][k] = (T @ np.array([b["cX"], b["cY"], b["cZ"], 1.0]))[:2]
    for scans in positions.values():
        ks = sorted(scans)
        if len(ks) < 20:
            continue
        P = np.array([scans[k] for k in ks])
        if np.linalg.norm(P - np.median(P, axis=0), axis=1).max() > 5.0:
            continue  # an instance id reused for another object
        stds.append(np.linalg.norm(P.std(axis=0)))
        steps += [np.linalg.norm(scans[b] - scans[a]) for a, b in zip(ks, ks[1:]) if b == a + 1]
    return len(stds), float(np.median(stds)) if stds else float("nan"), float(np.percentile(steps, 95)) if steps else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows", nargs="+", type=Path)
    ap.add_argument("--coda-root", type=Path)
    a = ap.parse_args()
    for window in a.windows:
        frames = truth_frames(window)
        seq = track_labels(frames)
        m = {**hota(seq, 1.0), **clear_and_id(seq, 1.0)}
        line = (f"{window.parent.name:14s} ids {len({g for _, objs in frames for g, _ in objs}):4d}  teleports {teleports(frames):4d}  "
                f"label tracking: HOTA {m['HOTA']:.3f} IDSW {int(m['IDSW']):4d}")
        if a.coda_root:
            n, std, step = static_wobble(window, a.coda_root)
            line += f"  | static objects {n:3d}: std {std:.2f} m, scan-to-scan 95% {step:.2f} m"
        print(line)


if __name__ == "__main__":
    main()
