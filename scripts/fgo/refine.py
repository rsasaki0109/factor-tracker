#!/usr/bin/env python3
"""Fixed-lag identity refinement of engine tracks (prototype).

Reads a replay/stream output (JSONL), re-decides which track pieces belong to
the same person, and writes the same output with corrected track ids.

1. Every track is cut where its observation comes within `--cut-distance` of
   another track's observation, and where it misses an observation. The
   pieces between cuts are unambiguous.
2. Pieces are linked into chains (people) in time order. At each frame, the
   pieces starting there are assigned (Hungarian) to the chains that ended
   within `--max-gap` frames before, or start a new chain.
3. The cost of a link is the change in the chain's factor-graph cost: the
   negative log-likelihood of its observations under a constant-velocity
   model (for this linear-Gaussian graph, the exact optimum's cost is the
   Kalman filter's prediction-error decomposition), plus `--miss-cost` per
   missed frame, against starting a new person (`--birth-cost`). The whole
   piece takes part, so a decision uses every observation up to the next cut:
   that is the lag.

usage: refine.py in.jsonl out.jsonl [options]
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e9


class Model:
    def __init__(self, q, sigma, velocity_variance):
        self.q, self.r, self.vvar = q, sigma * sigma, velocity_variance

    def predict(self, x, p, dt):
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        g3, g2, g1 = dt ** 3 / 3, dt ** 2 / 2, dt
        q = self.q * np.array([[g3, 0, g2, 0], [0, g3, 0, g2], [g2, 0, g1, 0], [0, g2, 0, g1]])
        return f @ x, f @ p @ f.T + q

    def update(self, x, p, z):
        h = np.zeros((2, 4))
        h[0, 0] = h[1, 1] = 1.0
        s = h @ p @ h.T + self.r * np.eye(2)
        nu = z - h @ x
        si = np.linalg.inv(s)
        k = p @ h.T @ si
        nll = 0.5 * (nu @ si @ nu + math.log(np.linalg.det(s)) + 2 * math.log(2 * math.pi))
        return x + k @ nu, (np.eye(4) - k @ h) @ p, nll

    def start(self, z):
        return np.array([z[0], z[1], 0.0, 0.0]), np.diag([self.r, self.r, self.vvar, self.vvar])

    def run(self, x, p, t, obs):
        """Filters observations [(time, z)] from state (x, p) at time t."""
        total = 0.0
        for to, z in obs:
            x, p = self.predict(x, p, to - t)
            x, p, nll = self.update(x, p, z)
            total += nll
            t = to
        return x, p, t, total


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def build_pieces(frames, cut_distance):
    per_track = defaultdict(list)
    for i, frame in enumerate(frames):
        t = frame["timestamp_ns"] * 1e-9
        for a in frame["associations"]:
            if a["track_id"] is None:
                continue
            z = np.array(frame["observations"][a["observation_index"]][:2])
            per_track[a["track_id"]].append((i, t, z))
    by_frame = defaultdict(list)
    for track, obs in per_track.items():
        for i, _, z in obs:
            by_frame[i].append((track, z))
    close = set()
    for i, entries in by_frame.items():
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                if np.linalg.norm(entries[a][1] - entries[b][1]) < cut_distance:
                    close.add((entries[a][0], i))
                    close.add((entries[b][0], i))
    pieces = []
    for track, obs in per_track.items():
        current = [obs[0]]
        for prev, cur in zip(obs, obs[1:]):
            gap = cur[0] - prev[0] > 1
            enters = (track, cur[0]) in close and (track, prev[0]) not in close
            leaves = (track, prev[0]) in close and (track, cur[0]) not in close
            if gap or enters or leaves:
                pieces.append((track, current))
                current = []
            current.append(cur)
        pieces.append((track, current))
    return [
        {"track": track, "first": p[0][0], "last": p[-1][0], "obs": [(t, z) for _, t, z in p],
         "inside": all((track, i) in close for i, _, _ in p)}
        for track, p in pieces
    ]


def end_states(frames, pieces):
    """The base tracker's posterior (x, p, t) at each piece's last frame:
    position, velocity and covariance with the track's whole history."""
    states = {}
    for k, piece in enumerate(pieces):
        frame = frames[piece["last"]]
        for track in frame["tracks"]:
            if track["track_id"] == piece["track"]:
                c = np.array(track["covariance"]).reshape(6, 6)
                idx = [0, 1, 3, 4]
                x = np.array([track["position"][0], track["position"][1], track["velocity"][0], track["velocity"][1]])
                states[k] = (x, c[np.ix_(idx, idx)], frame["timestamp_ns"] * 1e-9)
                break
    return states


def link(frames, pieces, model, max_gap, birth_cost, miss_cost, gate, keep_bonus):
    """Links piece ends to later piece starts by one global assignment.
    The base tracker's own links (consecutive pieces of one track) get
    `keep_bonus`, so they change only on clear evidence.
    Returns {piece index: chain id}."""
    states = end_states(frames, pieces)
    standalone = []
    for piece in pieces:
        x, p = model.start(piece["obs"][0][1])
        standalone.append(model.run(x, p, piece["obs"][0][0], piece["obs"][1:])[3] + birth_cost)
    n = len(pieces)
    by_first = defaultdict(list)
    for k, piece in enumerate(pieces):
        by_first[piece["first"]].append(k)
    rows, cols, costs = [], [], []
    for a, end in enumerate(pieces):
        if a not in states or end["inside"]:
            continue
        x0, p0, t0 = states[a]
        for frame in range(end["last"] + 1, end["last"] + 1 + max_gap):
            for b in by_first.get(frame, []):
                piece = pieces[b]
                if piece["inside"]:
                    continue
                own = piece["track"] == end["track"]
                xp, pp = model.predict(x0, p0, piece["obs"][0][0] - t0)
                d = piece["obs"][0][1] - xp[:2]
                if not own and d @ np.linalg.inv(pp[:2, :2] + model.r * np.eye(2)) @ d > gate:
                    continue
                nll = model.run(x0, p0, t0, piece["obs"])[3]
                delta = nll + (frame - end["last"] - 1) * miss_cost - standalone[b] - (keep_bonus if own else 0.0)
                if delta < 0:
                    rows.append(a)
                    cols.append(b)
                    costs.append(delta)
    successor = {}
    if rows:
        ends = sorted(set(rows))
        starts = sorted(set(cols))
        ri = {a: i for i, a in enumerate(ends)}
        ci = {b: j for j, b in enumerate(starts)}
        matrix = np.zeros((len(ends), len(starts)))  # 0 = not linked
        for a, b, c in zip(rows, cols, costs):
            matrix[ri[a], ci[b]] = c
        for i, j in zip(*linear_sum_assignment(matrix)):
            if matrix[i, j] < 0:
                successor[ends[i]] = starts[j]
    has_predecessor = set(successor.values())
    chain_of = {}
    for k in sorted(range(n), key=lambda k: pieces[k]["first"]):
        if k in has_predecessor or pieces[k]["inside"]:
            continue
        c = len(set(chain_of.values()))
        while k is not None:
            chain_of[k] = c
            k = successor.get(k)
    # Pieces inside an encounter are ambiguous: they take the person of their
    # track's next piece outside the encounter (the identity that piece was
    # linked to), or of the previous one at the end of a track.
    by_track = defaultdict(list)
    for k, piece in enumerate(pieces):
        by_track[piece["track"]].append(k)
    for ks in by_track.values():
        ks.sort(key=lambda k: pieces[k]["first"])
        for j, k in enumerate(ks):
            if not pieces[k]["inside"]:
                continue
            after = [x for x in ks[j + 1:] if not pieces[x]["inside"]]
            before = [x for x in ks[:j] if not pieces[x]["inside"]]
            ref = after[0] if after else (before[-1] if before else None)
            chain_of[k] = chain_of[ref] if ref is not None else len(set(chain_of.values())) + 10**6 + k
    return chain_of


def relabel(frames, pieces, chain_of):
    """Rewrites track ids: each original track's output at a frame takes the
    chain of the latest piece of that track started by then. A prediction
    after the piece's last observation is dropped once the chain has moved
    on to its next piece (which then reports that person)."""
    by_track = defaultdict(list)
    for k, piece in enumerate(pieces):
        by_track[piece["track"]].append(k)
    for ks in by_track.values():
        ks.sort(key=lambda k: pieces[k]["first"])
    next_first = {}
    by_chain = defaultdict(list)
    for k, c in chain_of.items():
        by_chain[c].append(k)
    for ks in by_chain.values():
        ks.sort(key=lambda k: pieces[k]["first"])
        for a, b in zip(ks, ks[1:]):
            next_first[a] = pieces[b]["first"]
    offset = 1_000_000
    out = []
    for i, frame in enumerate(frames):
        tracks = []
        for track in frame["tracks"]:
            ks = [k for k in by_track.get(track["track_id"], []) if pieces[k]["first"] <= i]
            if not ks:
                tracks.append(track)
                continue
            k = ks[-1]
            if i > pieces[k]["last"] and next_first.get(k, len(frames)) <= i:
                continue
            tracks.append(dict(track, track_id=offset + chain_of[k]))
        out.append(dict(frame, tracks=tracks))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--cut-distance", type=float, default=1.0)
    ap.add_argument("--max-gap", type=int, default=15)
    ap.add_argument("--birth-cost", type=float, default=8.0)
    ap.add_argument("--miss-cost", type=float, default=1.0)
    ap.add_argument("--gate", type=float, default=16.0)
    ap.add_argument("--keep-bonus", type=float, default=10.0)
    ap.add_argument("--accel", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--velocity-variance", type=float, default=1.0)
    args = ap.parse_args()
    records = load(args.input)
    frames = [r for r in records if r["type"] == "lidar_frame"]
    others = [r for r in records if r["type"] != "lidar_frame"]
    model = Model(args.accel, args.sigma, args.velocity_variance)
    pieces = build_pieces(frames, args.cut_distance)
    chain_of = link(frames, pieces, model, args.max_gap, args.birth_cost, args.miss_cost, args.gate, args.keep_bonus)
    refined = relabel(frames, pieces, chain_of)
    with open(args.output, "w") as h:
        for r in sorted(refined + others, key=lambda r: r["timestamp_ns"]):
            h.write(json.dumps(r) + "\n")
    tracks_before = len({a["track_id"] for f in frames for a in f["associations"] if a["track_id"] is not None})
    print(f"{len(pieces)} pieces from {tracks_before} tracks -> {len(set(chain_of.values()))} people")


if __name__ == "__main__":
    main()
