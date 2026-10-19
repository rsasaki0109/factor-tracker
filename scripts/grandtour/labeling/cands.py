"""Merge candidates: A ends, B starts within 30 frames and 1.5 m (world). usage: cands.py labels.json lo hi R"""
import json, sys, numpy as np
L = json.load(open(sys.argv[1])); lo, hi, R = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
tr = {}
for d in L: tr.setdefault(d["id"], []).append(d)
for v in tr.values(): v.sort(key=lambda d: d["frame"])
out = []
for a, va in tr.items():
    e = va[-1]
    if not (lo <= e["frame"] <= hi) or np.hypot(*e["base"][:2]) > R: continue
    for b, vb in tr.items():
        s = vb[0]; gap = s["frame"] - e["frame"]
        if b == a or not (1 <= gap <= 30): continue
        dist = np.linalg.norm(np.subtract(s["world"][:2], e["world"][:2]))
        if dist < 0.4 + 0.04 * gap: out.append((a, b, e["frame"], s["frame"], round(dist, 2), len(va), len(vb)))
for o in sorted(out): print(o)
print(len(out))
