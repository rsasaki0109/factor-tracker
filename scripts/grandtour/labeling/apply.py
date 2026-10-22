"""Apply labelling edits to pre-labels and write engine truth.
usage: apply.py pre.json edits.json seq_dir out_truth.jsonl out_labels.json
Segments are named "<track>@<first frame>" after splits; a bare number is the segment starting at the track's first frame."""
import json, sys
from pathlib import Path
import numpy as np
pre, edits = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); seq = Path(sys.argv[3])
lo, hi = edits["window"]; R = edits["max_range_m"]
splits = {}
for t, f in edits["split"]: splits.setdefault(t, []).append(f)
def seg(d):
    s = max([f for f in splits.get(d["id"], []) if f <= d["frame"]], default=None)
    return f"{d['id']}@{s}" if s is not None else str(d["id"])
parent = {}
def find(x):
    while parent.get(x, x) != x: x = parent[x]
    return x
for a, b in edits["merge"]: parent[find(str(b))] = find(str(a))
ignore = {str(x) for x in edits["ignore"]}
# Unreviewed short pieces (< min_len detections in the window, not merged) are left out.
count = {}
for d in pre:
    if lo <= d["frame"] <= hi and np.hypot(*d["base"][:2]) <= R: count[find(seg(d))] = count.get(find(seg(d)), 0) + 1
merged = {find(str(x)) for pair in edits["merge"] for x in pair}
ignore |= {s for s, n in count.items() if n < edits.get("min_len", 5) and s not in merged}
names, out = {}, []
for d in pre:
    if not (lo <= d["frame"] <= hi) or np.hypot(*d["base"][:2]) > R: continue
    s = seg(d)
    if s in ignore or find(s) in ignore: continue
    out.append(dict(d, person=names.setdefault(find(s), len(names) + 1)))
json.dump(out, open(sys.argv[5], "w"))
times = {json.loads(l)["sequence"]: json.loads(l)["event_time_ns"] for l in open(seq / "inputs.jsonl") if '"lidar_batch"' in l}
with open(sys.argv[4], "w") as handle:
    for f in range(lo, hi + 1):
        objects = [{"truth_id": 9000 + d["person"], "class": 2, "position": d["world"]} for d in out if d["frame"] == f]
        handle.write(json.dumps({"type": "truth3d", "schema_version": 1, "timestamp_ns": times[f], "objects": objects}) + "\n")
print(f"{len(out)} labelled detections, {len(names)} people")
