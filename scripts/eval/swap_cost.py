#!/usr/bin/env python3
"""Is an identity swap a model error or a search error?

usage: swap_cost.py <window-dir> <engine-output.jsonl>... [--window 10]
  window-dir  truth.jsonl (scripts/eval/coda_tracker_eval.sh work dir); give
              one engine output per window dir, in pairs: dir out dir out ...

At every transfer (engine track B moves from person g2 to person g, whose
detections track A had), the detections of A and B labelled g or g2 within
+-window scans are partitioned two ways: as the engine associated them, and
by the person they belong to. Both partitions are scored with the fixed-lag
refiner's cost (constant-velocity Kalman negative log-likelihood plus a miss
cost; `--fgo-lag`, crates/factor-tracker-engine/src/fgo.rs). If the engine's
partition is cheaper, no search over associations can fix the swap with this
cost: the model prefers the wrong answer.
"""

import argparse
import json
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def nll(track, sigma, q, v0, miss):
    """Constant-velocity Kalman NLL of [(scan, t, xy)], x and y independent."""
    if not track:
        return 0.0
    k_last, t_last, z0 = track[0]
    x, v = np.array(z0, float), np.zeros(2)
    p00, p01, p11, r, total = sigma**2, 0.0, v0, sigma**2, 0.0
    for k, t, z in track[1:]:
        dt = t - t_last
        p00, p01, p11 = p00 + 2 * dt * p01 + dt * dt * p11 + q * dt**3 / 3, p01 + dt * p11 + q * dt * dt / 2, p11 + q * dt
        x = x + dt * v
        s = p00 + r
        nu = np.array(z) - x
        total += 0.5 * float(nu @ nu) / s + np.log(2 * np.pi * s) + miss * (k - k_last - 1)
        k0, k1 = p00 / s, p01 / s
        x, v = x + k0 * nu, v + k1 * nu
        p00, p01, p11 = (1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01
        k_last, t_last = k, t
    return total


def load(window, output):
    truth = {}
    for line in open(f"{window}/truth.jsonl"):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            truth[r["timestamp_ns"]] = {o["truth_id"]: np.array(o["position"][:3]) for o in r["objects"]}
    frames = [json.loads(line) for line in open(output) if '"lidar_frame"' in line]
    scans = []  # per scan: [(xy, engine track, person)]
    for f in frames:
        observations = [np.array(o) for o in f["observations"]]
        owner = {a["observation_index"]: a["track_id"] for a in f["associations"]}
        people = list(truth.get(f["timestamp_ns"], {}).items())
        if f.get("pose_correction"):
            c = np.array(f["pose_correction"]).reshape(4, 4)
            people = [(g, c[:3, :3] @ p + c[:3, 3]) for g, p in people]
        label = {}
        if observations and people:
            d = np.linalg.norm(np.array([p for _, p in people])[:, None] - np.array(observations)[None], axis=2)
            cost = np.where(d <= 1.0, d, 1e6)
            label = {j: people[i][0] for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6}
        scans.append((f["timestamp_ns"] * 1e-9, dict(people), [(o[:2], owner.get(j), label.get(j)) for j, o in enumerate(observations)]))
    return scans


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--miss", type=float, default=0.3)
    a = ap.parse_args()
    if len(a.pairs) % 2:
        ap.error("give window dirs and engine outputs in pairs")
    counts = Counter()
    for window, output in zip(a.pairs[::2], a.pairs[1::2]):
        scans = load(window, output)
        last_of_track, last_of_person = {}, {}
        for k, (_, people, dets) in enumerate(scans):
            for _, track, person in dets:
                if track is None or person is None:
                    continue
                prev_track, prev_person = last_of_track.get(track), last_of_person.get(person)
                if prev_track and prev_person and prev_track[1] != person and prev_person[1] != track:
                    A, B, g, g2 = prev_person[1], track, person, prev_track[1]
                    lo, hi = max(0, k - a.window), min(len(scans), k + a.window + 1)
                    items = [(q, scans[q][0], xy, tr, pp) for q in range(lo, hi) for xy, tr, pp in scans[q][2]
                             if tr in (A, B) and pp in (g, g2)]
                    engine = sum(nll([(q, t, xy) for q, t, xy, tr, _ in items if tr == x], a.sigma, a.q, 1.0, a.miss) for x in (A, B))
                    right = sum(nll([(q, t, xy) for q, t, xy, _, pp in items if pp == x], a.sigma, a.q, 1.0, a.miss) for x in (g, g2))
                    if g in people and g2 in people:
                        sep = np.linalg.norm((people[g] - people[g2])[:2])
                        band = "< 1 m" if sep < 1 else ("1-1.5 m" if sep < 1.5 else ">= 1.5 m")
                    else:
                        band = "one unlabelled"
                    counts[(band, right < engine - 1e-9)] += 1
                last_of_track[track] = (k, person)
                last_of_person[person] = (k, track)
    total = sum(counts.values())
    right = sum(v for (_, ok), v in counts.items() if ok)
    print(f"transfers {total}: the right association is cheaper in {right} ({right / max(1, total):.0%})")
    for band in ["< 1 m", "1-1.5 m", ">= 1.5 m", "one unlabelled"]:
        n = counts[(band, True)] + counts[(band, False)]
        print(f"  people {band:15s} apart: {counts[(band, True)]:4d} / {n:4d}")


if __name__ == "__main__":
    main()
