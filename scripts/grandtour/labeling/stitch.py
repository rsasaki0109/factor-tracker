"""Stitch sheet: for every gap (> 2 frames) inside a track, crops before | after.
usage: stitch.py labels.json out_dir f_lo f_hi max_range"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).parent)); from cams import Cams
SEQ = Path(os.environ["GT_SEQ"])  # directory written by grandtour.py prepare (+ inputs.jsonl)
L = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
lo, hi, R = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
inputs = {json.loads(l)["sequence"]: json.loads(l) for l in open(SEQ / "inputs.jsonl") if '"lidar_batch"' in l}
scan_t = {int(k): v for k, v in json.load(open(SEQ / "frames.json"))["scans"].items()}
C = Cams(); F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
tr = {}
for d in L: tr.setdefault(d["id"], []).append(d)
rows = []
for i, v in sorted(tr.items()):
    v.sort(key=lambda d: d["frame"])
    for a, b in zip(v, v[1:]):
        if b["frame"] - a["frame"] > 2 and lo <= a["frame"] <= hi and min(np.hypot(*a["base"][:2]), np.hypot(*b["base"][:2])) < R:
            k = v.index(a); rows.append((i, v[max(0, k - 2):k + 1], v[k + 1:k + 4], np.linalg.norm(np.subtract(a["world"][:2], b["world"][:2]))))
def crop(d):
    det = inputs[d["frame"]]["detections"][d["k"]]
    im, name = C.crop(scan_t[d["frame"]], np.array(det["shape"]["corners"]).reshape(8, 3) + det["position"], size=(80, 140))
    return Image.fromarray(im) if im is not None else None
W = 6 * 84 + 150
for n0 in range(0, len(rows), 5):
    sheet = Image.new("RGB", (W, 5 * 160), (25, 25, 25)); dr = ImageDraw.Draw(sheet)
    for r, (i, before, after, dist) in enumerate(rows[n0:n0 + 5]):
        y = r * 160
        dr.text((4, y + 4), f"track {i}\ngap {before[-1]['frame']}->{after[0]['frame']}\njump {dist:.2f} m", fill=(255, 255, 0), font=F)
        for n, d in enumerate(before + [None] + after):
            x = 150 + n * 84 if n < len(before) else 150 + (n + (3 - len(before))) * 84
            if d is None: continue
            im = crop(d)
            if im: sheet.paste(im, (x, y))
            else: dr.text((x + 10, y + 60), "no cam", fill=(150, 150, 150), font=F)
            dr.text((x + 2, y + 142), str(d["frame"]), fill=(0, 255, 0), font=F)
        dr.line([150 + 3 * 84 - 2, y, 150 + 3 * 84 - 2, y + 158], fill=(255, 0, 0), width=3)
    sheet.save(out / f"s{n0 // 5:02d}.jpg", quality=85)
print(len(rows), "stitches;", [r[0] for r in rows])
