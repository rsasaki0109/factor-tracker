#!/usr/bin/env python3
"""HOTA, IDF1 and MOTA of an engine output against truth3d records.

Similarity of a track and a truth object is distance based (the outputs carry
positions, not boxes): s = max(0, 1 - d / --max-distance) with d the 3D
centre distance. HOTA follows Luiten et al. (IJCV 2020) and TrackEval:
alpha in 0.05..0.95, per-frame Hungarian matching on similarity weighted by
the global alignment of the ids. IDF1 and MOTA use a match when
d <= --max-distance (IDF1 with a global id-to-id assignment, MOTA with
per-frame matching that prefers the previous match). Only confirmed tracks
count unless --all-tracks.

usage: mot_metrics.py output.jsonl truth.jsonl [--max-distance 1.0] [--zone-range-m R --setup setup.json]
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def load(output, truth, all_tracks, zone, setup):
    frames = [json.loads(l) for l in open(output) if '"lidar_frame"' in l]
    gt = {}
    for line in open(truth):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            gt[r["timestamp_ns"]] = [(o["truth_id"], np.array(o["position"][:3], float)) for o in r["objects"]]
    poses = None
    if zone is not None:
        s = json.load(open(setup))
        poses = sorted((p["timestamp_ns"], np.array(p["transform"]).reshape(4, 4)) for p in s.get("poses") or [])
    seq = []
    for f in frames:
        if f["timestamp_ns"] not in gt:
            continue
        objs = gt[f["timestamp_ns"]]
        if zone is not None and poses:
            t = f["timestamp_ns"]
            k = min(range(len(poses)), key=lambda i: abs(poses[i][0] - t))
            origin = poses[k][1][:2, 3]
            objs = [(i, p) for i, p in objs if np.linalg.norm(p[:2] - origin) <= zone]
        tracks = [(t["track_id"], np.array(t["position"][:3], float)) for t in f["tracks"]
                  if all_tracks or t["lifecycle"] == "confirmed"]
        seq.append((objs, tracks))
    return seq


def distance(objs, tracks):
    if not objs or not tracks:
        return np.zeros((len(objs), len(tracks)))
    a = np.array([p for _, p in objs])
    b = np.array([p for _, p in tracks])
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def hota(seq, dmax):
    alphas = np.arange(0.05, 0.96, 0.05)
    gt_ids = sorted({i for objs, _ in seq for i, _ in objs})
    tr_ids = sorted({i for _, tracks in seq for i, _ in tracks})
    gi = {i: k for k, i in enumerate(gt_ids)}
    ti = {i: k for k, i in enumerate(tr_ids)}
    pot = np.zeros((len(gt_ids), len(tr_ids)))
    gt_count = np.zeros((len(gt_ids), 1))
    tr_count = np.zeros((1, len(tr_ids)))
    sims = []
    for objs, tracks in seq:
        s = np.clip(1 - distance(objs, tracks) / dmax, 0, 1)
        sims.append(s)
        g = [gi[i] for i, _ in objs]
        t = [ti[i] for i, _ in tracks]
        if len(g) and len(t):
            denom = s.sum(0)[None, :] + s.sum(1)[:, None] - s
            iou_like = np.divide(s, denom, out=np.zeros_like(s), where=denom > 0)
            pot[np.ix_(g, t)] += iou_like
        for k in g:
            gt_count[k] += 1
        for k in t:
            tr_count[0, k] += 1
    align = pot / np.maximum(1, gt_count + tr_count - pot)
    tp = np.zeros(len(alphas))
    fn = np.zeros(len(alphas))
    fp = np.zeros(len(alphas))
    matches = np.zeros((len(alphas), len(gt_ids), len(tr_ids)))
    loc = np.zeros(len(alphas))
    for (objs, tracks), s in zip(seq, sims):
        g = [gi[i] for i, _ in objs]
        t = [ti[i] for i, _ in tracks]
        if not g or not t:
            fn += len(g)
            fp += len(t)
            continue
        score = align[np.ix_(g, t)] * s
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(alphas):
            ok = s[rows, cols] >= alpha - np.finfo(float).eps
            n = ok.sum()
            tp[a] += n
            fn[a] += len(g) - n
            fp[a] += len(t) - n
            loc[a] += s[rows, cols][ok].sum()
            matches[a, [g[r] for r in rows[ok]], [t[c] for c in cols[ok]]] += 1
    det_a = tp / np.maximum(1, tp + fn + fp)
    ass_a = np.zeros(len(alphas))
    for a in range(len(alphas)):
        m = matches[a]
        ass = m / np.maximum(1, gt_count + tr_count - m)
        ass_a[a] = (m * ass).sum() / max(1, tp[a])
    h = np.sqrt(det_a * ass_a)
    return {"HOTA": h.mean(), "DetA": det_a.mean(), "AssA": ass_a.mean(), "LocA": (loc / np.maximum(1, tp)).mean()}


def clear_and_id(seq, dmax):
    tp = fn = fp = idsw = 0
    last = {}
    pairs = defaultdict(int)
    gt_n = defaultdict(int)
    tr_n = defaultdict(int)
    for objs, tracks in seq:
        for i, _ in objs:
            gt_n[i] += 1
        for i, _ in tracks:
            tr_n[i] += 1
        d = distance(objs, tracks)
        if d.size:
            # MOTA: keep the previous match when still within the gate
            cost = np.where(d <= dmax, d, 1e6)
            for r, (g, _) in enumerate(objs):
                for c, (t, _) in enumerate(tracks):
                    if last.get(g) == t and d[r, c] <= dmax:
                        cost[r, c] = -1.0
            rows, cols = linear_sum_assignment(cost)
            matched = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 1e6]
            for r, c in matched:
                pairs[(objs[r][0], tracks[c][0])] += 1
                g, t = objs[r][0], tracks[c][0]
                if g in last and last[g] != t:
                    idsw += 1
                last[g] = t
            tp += len(matched)
            fn += len(objs) - len(matched)
            fp += len(tracks) - len(matched)
        else:
            fn += len(objs)
            fp += len(tracks)
    total_gt = sum(gt_n.values())
    mota = 1 - (fn + fp + idsw) / max(1, total_gt)
    # IDF1: global one-to-one assignment of gt ids to track ids
    g_ids = sorted(gt_n)
    t_ids = sorted(tr_n)
    w = np.zeros((len(g_ids), len(t_ids)))
    gk = {i: k for k, i in enumerate(g_ids)}
    tk = {i: k for k, i in enumerate(t_ids)}
    for (g, t), n in pairs.items():
        w[gk[g], tk[t]] = n
    idtp = 0
    if w.size:
        rows, cols = linear_sum_assignment(-w)
        idtp = w[rows, cols].sum()
    idf1 = 2 * idtp / max(1, total_gt + sum(tr_n.values()))
    return {"MOTA": mota, "IDF1": idf1, "IDSW": idsw, "FP": fp, "FN": fn, "GT": total_gt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output")
    ap.add_argument("truth")
    ap.add_argument("--max-distance", type=float, default=1.0)
    ap.add_argument("--all-tracks", action="store_true")
    ap.add_argument("--zone-range-m", type=float)
    ap.add_argument("--setup")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    seq = load(a.output, a.truth, a.all_tracks, a.zone_range_m, a.setup)
    m = {**hota(seq, a.max_distance), **clear_and_id(seq, a.max_distance)}
    if a.json:
        print(json.dumps({k: float(v) for k, v in m.items()}))
    else:
        print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))


if __name__ == "__main__":
    main()
