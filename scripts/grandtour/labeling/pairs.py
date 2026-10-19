"""Pair sheet for merge candidates (A end | B start). usage: pairs.py labels.json out_dir A:B [A:B ...]"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).parent)); from cams import Cams
SEQ = Path(os.environ["GT_SEQ"])  # directory written by grandtour.py prepare (+ inputs.jsonl)
L = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
inputs = {json.loads(l)["sequence"]: json.loads(l) for l in open(SEQ / "inputs.jsonl") if '"lidar_batch"' in l}
scan_t = {int(k): v for k, v in json.load(open(SEQ / "frames.json"))["scans"].items()}
C = Cams(); F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
tr = {}
for d in L: tr.setdefault(d["id"], []).append(d)
for v in tr.values(): v.sort(key=lambda d: d["frame"])
def crop(d):
    det = inputs[d["frame"]]["detections"][d["k"]]
    im, _ = C.crop(scan_t[d["frame"]], np.array(det["shape"]["corners"]).reshape(8, 3) + det["position"], size=(80, 140))
    return Image.fromarray(im) if im is not None else None
pairs = [tuple(map(int, p.split(":"))) for p in sys.argv[3:]]
for n0 in range(0, len(pairs), 5):
    sheet = Image.new("RGB", (6 * 84 + 150, 5 * 160), (25, 25, 25)); dr = ImageDraw.Draw(sheet)
    for r, (a, b) in enumerate(pairs[n0:n0 + 5]):
        y = r * 160; A = tr[a][-3:]; B = tr[b][:3]
        dist = np.linalg.norm(np.subtract(A[-1]["world"][:2], B[0]["world"][:2]))
        dr.text((4, y + 4), f"{a} -> {b}\n{A[-1]['frame']}->{B[0]['frame']}\n{dist:.2f} m\nlen {len(tr[a])}/{len(tr[b])}", fill=(255, 255, 0), font=F)
        for n, d in enumerate(A):
            x = 150 + (3 - len(A) + n) * 84; im = crop(d)
            sheet.paste(im, (x, y)) if im else dr.text((x + 10, y + 60), "no cam", fill=(150, 150, 150), font=F)
            dr.text((x + 2, y + 142), str(d["frame"]), fill=(0, 255, 0), font=F)
        for n, d in enumerate(B):
            x = 150 + (3 + n) * 84; im = crop(d)
            sheet.paste(im, (x, y)) if im else dr.text((x + 10, y + 60), "no cam", fill=(150, 150, 150), font=F)
            dr.text((x + 2, y + 142), str(d["frame"]), fill=(0, 255, 255), font=F)
        dr.line([150 + 3 * 84 - 2, y, 150 + 3 * 84 - 2, y + 158], fill=(255, 0, 0), width=3)
    sheet.save(out / f"p{n0 // 5:02d}.jpg", quality=85)
