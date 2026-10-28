#!/usr/bin/env python3
"""What would perfect identity repair be worth? Relabels a tracker output with
the truth's help and writes it for mot_metrics.py:

  --split   cut each track where the person it matches changes (a track that
            moved to someone else becomes two)
  --stitch  give the pieces of one person the same id when they do not
            overlap in time (fragments joined)

Both together are a perfect swap correction on the tracker's own positions:
nothing is added or removed, only ids change.

usage: oracle_repair.py <output.jsonl> <truth.jsonl> <repaired.jsonl> [--split] [--stitch]
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output")
    ap.add_argument("truth")
    ap.add_argument("repaired")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--stitch", action="store_true")
    a = ap.parse_args()
    truth = {}
    for line in open(a.truth):
        r = json.loads(line)
        if r.get("type") == "truth3d":
            truth[r["timestamp_ns"]] = [(o["truth_id"], np.array(o["position"][:3])) for o in r["objects"]]
    frames = []
    for line in open(a.output):
        f = json.loads(line)
        if f.get("type") != "lidar_frame":
            continue
        c = np.array(f["pose_correction"]).reshape(4, 4) if f.get("pose_correction") else np.eye(4)
        people = [(g, c[:3, :3] @ p + c[:3, 3]) for g, p in truth.get(f["timestamp_ns"], [])]
        tracks = [(t["track_id"], t["position"]) for t in f["tracks"] if t["lifecycle"] == "confirmed"]
        match = {}
        if people and tracks:
            d = np.linalg.norm(np.array([p for _, p in people])[:, None] - np.array([p for _, p in tracks])[None], axis=2)
            cost = np.where(d <= 1.0, d, 1e6)
            match = {tracks[j][0]: people[i][0] for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6}
        frames.append((f, tracks, match))
    # pieces: a track, cut (with --split) wherever its matched person changes
    piece, current, person, n = {}, {}, {}, 0
    for k, (_, tracks, match) in enumerate(frames):
        for i, _ in tracks:
            g = match.get(i)
            if i not in current or (a.split and g is not None and person.get(i) not in (None, g)):
                n += 1
                current[i] = n
            if g is not None:
                person[i] = g
            piece[(k, i)] = current[i]
    final = {p: p for p in set(piece.values())}
    if a.stitch:
        cover, span = defaultdict(Counter), defaultdict(list)
        for (k, i), p in piece.items():
            span[p].append(k)
            g = frames[k][2].get(i)
            if g is not None:
                cover[p][g] += 1
        of_person = defaultdict(list)
        for p, c in cover.items():
            of_person[c.most_common(1)[0][0]].append(p)
        for pieces in of_person.values():
            pieces.sort(key=lambda p: min(span[p]))
            for prev, p in zip(pieces, pieces[1:]):
                if min(span[p]) > max(span[prev]):
                    final[p] = final[prev]
    with open(a.repaired, "w") as out:
        for k, (f, tracks, _) in enumerate(frames):
            record = {"type": "lidar_frame", "timestamp_ns": f["timestamp_ns"],
                      "tracks": [{"track_id": final[piece[(k, i)]], "lifecycle": "confirmed", "position": p} for i, p in tracks]}
            if f.get("pose_correction"):
                record["pose_correction"] = f["pose_correction"]
            out.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
