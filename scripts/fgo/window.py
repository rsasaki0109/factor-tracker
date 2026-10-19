#!/usr/bin/env python3
"""Multi-frame re-association by factor-graph cost (prototype).

Re-decides which detection belongs to which person over the whole sequence
(`--lag 0`) or over a sliding window of the last `--lag` frames, starting from
the engine's own association (read from a replay/stream output).

Cost of a hypothesis (negative log-likelihood, lower is better):
  per track   birth + trajectory cost + misses * miss_cost
  per detection left out (clutter)   clutter_cost
The trajectory cost is the optimum of the track's factor graph (constant
velocity motion factors + position factors); for this linear-Gaussian graph
it is exactly the Kalman filter's prediction-error decomposition, which is
what is computed. Horizontal motion only; x and y share one covariance.

Local search from the engine's solution, accepting only moves that lower the
cost: tail swap of two tracks at a frame (a steal), merge of two tracks (a
duplicate), split of a track at a frame, and moving one detection between a
track and clutter or another track.

Output: the engine output format, re-estimated causally from the refined
associations (only the association uses the lag): a track is reported from
its first detection, confirmed after two, and coasts for `--coast` seconds.

usage: window.py engine_output.jsonl refined.jsonl [options]
"""

import argparse
import json
import math
from collections import defaultdict


class Costs:
    def __init__(self, a):
        self.q, self.r, self.vvar = a.accel, a.sigma ** 2, a.velocity_variance
        self.birth, self.miss, self.clutter = a.birth_cost, a.miss_cost, a.clutter_cost


def kf_cost(seq, z, t, c):
    """Trajectory cost of observations `seq` [(frame, obs)] (sorted)."""
    f0, o0 = seq[0]
    x = [z[(f0, o0)][0], z[(f0, o0)][1]]
    v = [0.0, 0.0]
    p00, p01, p11 = c.r, 0.0, c.vvar
    tp = t[f0]
    nll = 0.0
    for f, o in seq[1:]:
        dt = t[f] - tp
        tp = t[f]
        # predict (same covariance for x and y)
        p00, p01, p11 = (p00 + 2 * dt * p01 + dt * dt * p11 + c.q * dt ** 3 / 3,
                         p01 + dt * p11 + c.q * dt * dt / 2, p11 + c.q * dt)
        x = [x[0] + dt * v[0], x[1] + dt * v[1]]
        s = p00 + c.r
        k0, k1 = p00 / s, p01 / s
        zz = z[(f, o)]
        for i in range(2):
            nu = zz[i] - x[i]
            nll += 0.5 * (nu * nu / s + math.log(2 * math.pi * s))
            x[i] += k0 * nu
            v[i] += k1 * nu
        p00, p01, p11 = (1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01
    return nll


def track_cost(seq, z, t, c):
    if not seq:
        return 0.0
    misses = (seq[-1][0] - seq[0][0] + 1) - len(seq)
    return c.birth + kf_cost(seq, z, t, c) + misses * c.miss


def valid(seq):
    frames = [f for f, _ in seq]
    return len(frames) == len(set(frames))


class Problem:
    def __init__(self, frames, c):
        self.c = c
        self.t = [f["timestamp_ns"] * 1e-9 for f in frames]
        self.z = {}
        self.tracks = defaultdict(list)
        for i, f in enumerate(frames):
            for o, pos in enumerate(f["observations"]):
                self.z[(i, o)] = pos
            for a in f["associations"]:
                if a["track_id"] is not None:
                    self.tracks[a["track_id"]].append((i, a["observation_index"]))
        assigned = {e for seq in self.tracks.values() for e in seq}
        self.clutter = set(self.z) - assigned
        for seq in self.tracks.values():
            seq.sort()
        self.cache = {}
        self.next_id = max(self.tracks, default=0) + 1

    def cost(self, tid):
        seq = self.tracks.get(tid, [])
        key = (tid, len(seq), seq[0] if seq else None, seq[-1] if seq else None, hash(tuple(seq)))
        if key not in self.cache:
            self.cache[key] = track_cost(seq, self.z, self.t, self.c)
        return self.cache[key]

    def seq_cost(self, seq):
        return track_cost(seq, self.z, self.t, self.c)

    def total(self):
        return sum(self.cost(k) for k in self.tracks) + len(self.clutter) * self.c.clutter

    def neighbours(self, radius, lo=0, hi=None):
        """Pairs of tracks with detections within `radius` at one frame
        (or at frames one apart), restricted to frames [lo, hi]."""
        by_frame = defaultdict(list)
        for tid, seq in self.tracks.items():
            for f, o in seq:
                if f >= lo and (hi is None or f <= hi):
                    by_frame[f].append((tid, self.z[(f, o)]))
        pairs = defaultdict(set)
        for f, entries in by_frame.items():
            nearby = entries + by_frame.get(f + 1, [])
            for i, (a, pa) in enumerate(entries):
                for b, pb in nearby:
                    if a != b and (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 < radius * radius:
                        pairs[(min(a, b), max(a, b))].add(f)
        return pairs


def improve(prob, radius, lo=0, hi=None, frozen_before=0, max_passes=20):
    """Greedy local search; moves only touch frames >= frozen_before."""
    applied = defaultdict(int)
    c = prob.c
    for _ in range(max_passes):
        changed = False
        for (a, b), frames in sorted(prob.neighbours(radius, lo, hi).items()):
            if a not in prob.tracks or b not in prob.tracks:
                continue
            A, B = prob.tracks[a], prob.tracks[b]
            base = prob.cost(a) + prob.cost(b)
            best = (0.0, None)
            cuts = sorted({f for f in frames} | {f + 1 for f in frames})
            for f in cuts:
                if f < frozen_before:
                    continue
                na = [e for e in A if e[0] < f] + [e for e in B if e[0] >= f]
                nb = [e for e in B if e[0] < f] + [e for e in A if e[0] >= f]
                if not valid(na) or not valid(nb) or (na == A and nb == B):
                    continue
                d = prob.seq_cost(na) + prob.seq_cost(nb) - base
                if d < best[0] - 1e-9:
                    best = (d, ("swap", na, nb, []))
            # merge: at frames where both have a detection keep one of them
            for keep_a in (True, False):
                fa = {f: e for e, f in ((e, e[0]) for e in A)}
                fb = {f: e for e, f in ((e, e[0]) for e in B)}
                merged, dropped = [], []
                for f in sorted(set(fa) | set(fb)):
                    if f in fa and f in fb:
                        if f < frozen_before:
                            merged = None
                            break
                        keep, drop = (fa[f], fb[f]) if keep_a else (fb[f], fa[f])
                        merged.append(keep)
                        dropped.append(drop)
                    else:
                        merged.append(fa.get(f, fb.get(f)))
                if merged is None:
                    continue
                d = prob.seq_cost(merged) + len(dropped) * c.clutter - base
                if d < best[0] - 1e-9:
                    best = (d, ("merge", merged, [], dropped))
            if best[1] is None:
                continue
            kind, na, nb, dropped = best[1]
            prob.tracks[a] = na
            if nb:
                prob.tracks[b] = nb
            else:
                del prob.tracks[b]
            prob.clutter.update(dropped)
            applied[kind] += 1
            changed = True
        # single tracks: split at a frame, or drop/add one detection
        for tid in list(prob.tracks):
            seq = prob.tracks[tid]
            base = prob.cost(tid)
            best = (0.0, None)
            for j in range(1, len(seq)):
                if seq[j][0] < frozen_before:
                    continue
                d = prob.seq_cost(seq[:j]) + prob.seq_cost(seq[j:]) - base
                if d < best[0] - 1e-9:
                    best = (d, ("split", seq[:j], seq[j:]))
            for j in range(len(seq)):
                if seq[j][0] < frozen_before or len(seq) == 1:
                    continue
                rest = seq[:j] + seq[j + 1:]
                d = prob.seq_cost(rest) + c.clutter - base
                if d < best[0] - 1e-9:
                    best = (d, ("drop", rest, seq[j]))
            if best[1] is None:
                continue
            if best[1][0] == "split":
                prob.tracks[tid] = best[1][1]
                prob.tracks[prob.next_id] = best[1][2]
                prob.next_id += 1
            else:
                prob.tracks[tid] = best[1][1]
                prob.clutter.add(best[1][2])
            applied[best[1][0]] += 1
            changed = True
        if not changed:
            break
    return applied


def reestimate(frames, prob, coast, c):
    """Causal per-frame outputs of the refined tracks (engine format)."""
    out = [dict(f, tracks=[]) for f in frames]
    t = prob.t
    for tid, seq in prob.tracks.items():
        if not seq:
            continue
        obs = dict(seq)
        f0 = seq[0][0]
        z0 = prob.z[seq[0]]
        x, v = [z0[0], z0[1]], [0.0, 0.0]
        p00, p01, p11 = c.r, 0.0, c.vvar
        heights = [z0[2]]
        count, last_obs_t = 1, t[f0]
        f = f0
        while f < len(frames):
            if f > f0:
                dt = t[f] - t[f - 1]
                p00, p01, p11 = (p00 + 2 * dt * p01 + dt * dt * p11 + c.q * dt ** 3 / 3,
                                 p01 + dt * p11 + c.q * dt * dt / 2, p11 + c.q * dt)
                x = [x[0] + dt * v[0], x[1] + dt * v[1]]
                if f in obs:
                    zz = prob.z[(f, obs[f])]
                    s = p00 + c.r
                    k0, k1 = p00 / s, p01 / s
                    for i in range(2):
                        nu = zz[i] - x[i]
                        x[i] += k0 * nu
                        v[i] += k1 * nu
                    p00, p01, p11 = (1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01
                    heights = (heights + [zz[2]])[-5:]
                    count += 1
                    last_obs_t = t[f]
            if t[f] - last_obs_t > coast:
                if f > seq[-1][0]:
                    break
                f += 1
                continue  # lost: not reported until the next detection
            now_ns = frames[f]["timestamp_ns"]
            last_ns = int(round(last_obs_t * 1e9))
            out[f]["tracks"].append({
                "track_id": int(tid), "lifecycle": "confirmed" if count >= 2 else "tentative",
                "state_timestamp_ns": now_ns, "last_spatial_update_ns": last_ns, "last_camera_support_ns": None,
                "age_since_spatial_update_ns": now_ns - last_ns, "prediction_only": f not in obs,
                "position": [x[0], x[1], sum(heights) / len(heights)], "velocity": [v[0], v[1], 0.0],
                "covariance": [p00 if i % 7 == 0 and i < 14 else 0.0 for i in range(36)],
                "class_label": 2, "class_prior": None, "contributing_sensors": [1],
                "observation_times_count": count,
            })
            f += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--lag", type=int, default=0, help="window in frames; 0 = whole sequence at once")
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--birth-cost", type=float, default=8.0)
    ap.add_argument("--miss-cost", type=float, default=1.6)
    ap.add_argument("--clutter-cost", type=float, default=6.0)
    ap.add_argument("--accel", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--velocity-variance", type=float, default=1.0)
    ap.add_argument("--coast", type=float, default=0.5)
    ap.add_argument("--no-refine", action="store_true", help="only re-estimate the engine's associations")
    args = ap.parse_args()
    records = [json.loads(l) for l in open(args.input) if l.strip()]
    frames = [r for r in records if r["type"] == "lidar_frame"]
    c = Costs(args)
    prob = Problem(frames, c)
    before = prob.total()
    applied = {}
    if not args.no_refine:
        if args.lag <= 0:
            applied = improve(prob, args.radius)
        else:
            applied = defaultdict(int)
            for end in range(len(frames)):
                lo = max(0, end - args.lag)
                for k, v in improve(prob, args.radius, lo=lo, hi=end, frozen_before=lo).items():
                    applied[k] += v
    out = reestimate(frames, prob, args.coast, c)
    with open(args.output, "w") as h:
        for r in out:
            h.write(json.dumps(r) + "\n")
    print(f"cost {before:.0f} -> {prob.total():.0f}; moves {dict(applied)}; "
          f"{len(prob.tracks)} tracks, {len(prob.clutter)} clutter")


if __name__ == "__main__":
    main()
