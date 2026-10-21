#!/usr/bin/env python3
"""Classifies the ID switches of a tracker output against truth.

Matching per frame: Hungarian between truth objects and confirmed tracks
within `--match` metres (as factor-tracker evaluate). For each switch of a
truth object from track A to track B:

  fragment   B is new (first matched within the last few frames) and A is no
             longer reported: the person's track was lost and restarted.
  steal      B was matched to another truth object shortly before: the
             tracks of two people got exchanged or one took over the other.
  return     A still exists and is matched to nothing: the matcher picked
             another track for a while (often a duplicate track).
  other

usage: switch_types.py output.jsonl truth.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("truth")
    ap.add_argument("--match", type=float, default=1.0)
    ap.add_argument("--recent", type=int, default=5, help="frames")
    args = ap.parse_args()
    frames = [json.loads(l) for l in open(args.output) if '"lidar_frame"' in l]
    truth = {json.loads(l)["timestamp_ns"]: json.loads(l) for l in open(args.truth) if '"truth3d"' in l}
    first_seen = {}
    last_owner = {}  # track -> (truth, frame)
    last_match = {}  # truth -> (track, frame)
    alive = defaultdict(set)
    counts = Counter()
    for i, frame in enumerate(frames):
        tr = truth.get(frame["timestamp_ns"])
        tracks = [t for t in frame["tracks"] if t["lifecycle"] == "confirmed"]
        alive[i] = {t["track_id"] for t in frame["tracks"]}
        for t in tracks:
            first_seen.setdefault(t["track_id"], i)
        if not tr or not tracks or not tr["objects"]:
            continue
        objs = tr["objects"]
        d = np.array([[np.linalg.norm(np.subtract(o["position"][:2], t["position"][:2])) for t in tracks] for o in objs])
        rows, cols = linear_sum_assignment(np.where(d < args.match, d, 1e6))
        matched = {}
        for r, c in zip(rows, cols):
            if d[r, c] < args.match:
                matched[objs[r]["truth_id"]] = tracks[c]["track_id"]
        for gt, track in matched.items():
            if gt in last_match and last_match[gt][0] != track:
                old = last_match[gt][0]
                owner = last_owner.get(track)
                if owner and owner[0] != gt and i - owner[1] <= args.recent:
                    counts["steal"] += 1
                elif i - first_seen[track] <= args.recent and old not in alive[i]:
                    counts["fragment"] += 1
                elif old in alive[i] and old not in matched.values():
                    counts["return"] += 1
                else:
                    counts["other"] += 1
            last_match[gt] = (track, i)
            last_owner[track] = (gt, i)
    total = sum(counts.values())
    print(f"{total} switches: " + ", ".join(f"{k} {v} ({v / total:.0%})" for k, v in counts.most_common()))


if __name__ == "__main__":
    main()
