"""Per-track review sheet: camera crops along the track + x(t), y(t) with neighbours.
usage: sheet.py labels.json out_dir id [id ...]   (or 'all:N' for every track with >= N frames)"""
import json, os, sys, colorsys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEQ = Path(os.environ["GT_SEQ"])  # directory written by grandtour.py prepare (+ inputs.jsonl)
L = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
setup = json.load(open(SEQ / "setup.json")); K = setup["intrinsics"]
cam = np.linalg.inv(np.array(setup["camera_to_reference"]).reshape(4, 4))
inputs = {json.loads(l)["sequence"]: json.loads(l) for l in open(SEQ / "inputs.jsonl") if '"lidar_batch"' in l}
tr = {}
for d in L: tr.setdefault(d["id"], []).append(d)
ids = []
for a in sys.argv[3:]:
    if a.startswith("all:"): ids += sorted(i for i, v in tr.items() if i is not None and len(v) >= int(a[4:]))
    elif "@" in a: i, r = a.split("@"); lo, hi = map(int, r.split("-")); ids.append((int(i), lo, hi))
    else: ids.append(int(a))
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
c0 = np.array(json.load(open(SEQ / "setup.json"))["poses"][0]["transform"]).reshape(4, 4)[:2, 3]
def col(i): return colorsys.hsv_to_rgb((i * 0.61803) % 1, 0.85, 0.8)
def crop(d):
    det = inputs[d["frame"]]["detections"][d["k"]]
    c = np.array(det["shape"]["corners"]).reshape(8, 3) + det["position"]
    cc = (cam[:3, :3] @ c.T).T + cam[:3, 3]
    if np.any(cc[:, 2] < 0.5): return None
    u = K["fx"] * cc[:, 0] / cc[:, 2] + K["cx"]; v = K["fy"] * cc[:, 1] / cc[:, 2] + K["cy"]
    x1, y1, x2, y2 = u.min(), v.min(), u.max(), v.max()
    if x2 < 0 or x1 > K["image_width"] or y2 < 0 or y1 > K["image_height"]: return None
    w, h = x2 - x1, y2 - y1; x1 -= 0.15 * w; x2 += 0.15 * w; y1 -= 0.1 * h; y2 += 0.05 * h
    img = Image.open(SEQ / f"images/{d['frame']:06d}.jpg").convert("RGB")
    im = img.crop((int(max(x1, 0)), int(max(y1, 0)), int(min(x2, img.width)), int(min(y2, img.height))))
    im = ImageEnhance.Brightness(im).enhance(2.2)
    return im.resize((90, 180))
by_frame = {}
for d in L: by_frame.setdefault(d["frame"], []).append(d)
def patch(d):
    pts = np.fromfile(SEQ / f"scans/{d['frame']:06d}.bin", dtype=np.float32).reshape(-1, 4)
    b = np.array(d["base"]); rel = pts[:, :3] - b
    im = Image.new("RGB", (90, 180), (15, 15, 20)); dr = ImageDraw.Draw(im)
    m = (np.abs(rel[:, 0]) < 1.5) & (np.abs(rel[:, 1]) < 1.5) & (pts[:, 2] > b[2] - 1.2) & (pts[:, 2] < b[2] + 1.3)
    for x, y, z in rel[m]:  # top-down 3 m x 3 m in the top 90x90
        u, v = 45 - y * 30, 45 - x * 30; g = int(np.clip(80 + (z + 1) * 70, 60, 255)); dr.point((u, v), fill=(g, g, 120))
    for e in by_frame[d["frame"]]:
        q = np.array(e["base"]) - b
        if abs(q[0]) < 1.5 and abs(q[1]) < 1.5:
            u, v = 45 - q[1] * 30, 45 - q[0] * 30; c = tuple(int(255 * k) for k in col(e["id"]))
            dr.ellipse([u - 4, v - 4, u + 4, v + 4], outline=c); dr.text((u + 4, v - 6), str(e["id"]), fill=c, font=F)
    s = (np.hypot(rel[:, 0], rel[:, 1]) < 0.45) & (np.abs(rel[:, 2]) < 1.3)
    for x, y, z in rel[s]:  # side view below: lateral vs height
        r = (x * b[1] - y * b[0]) / (np.hypot(b[0], b[1]) + 1e-6)
        dr.point((45 + r * 40, 135 - z * 35), fill=(200, 200, 255))
    dr.line([0, 90, 90, 90], fill=(70, 70, 70))
    return im
sys.path.insert(0, str(Path(__file__).parent)); from cams import Cams
CAMS = Cams(); scan_t = {int(k): v for k, v in json.load(open(SEQ / "frames.json"))["scans"].items()}
def crop(d):
    det = inputs[d["frame"]]["detections"][d["k"]]
    corners = np.array(det["shape"]["corners"]).reshape(8, 3) + det["position"]
    im, name = CAMS.crop(scan_t[d["frame"]], corners)
    if im is None: return None
    im = Image.fromarray(im); ImageDraw.Draw(im).text((2, 2), name, fill=(0, 255, 0), font=F); return im
for i in ids:
    lo, hi = -1, 10**9
    if isinstance(i, tuple): i, lo, hi = i
    v = sorted((d for d in tr[i] if lo <= d["frame"] <= hi), key=lambda d: d["frame"])
    pick = [v[int(round(k))] for k in np.linspace(0, len(v) - 1, min(12, len(v)))]
    strip = Image.new("RGB", (12 * 114, 200), (30, 30, 30)); ds = ImageDraw.Draw(strip)
    for n, d in enumerate(pick):
        im = crop(d)
        if im is not None: strip.paste(im, (n * 114 + 2, 0))
        else: strip.paste(patch(d), (n * 114 + 2, 0))
        ds.text((n * 114 + 4, 183), f"{d['frame']} {np.hypot(*d['base'][:2]):.1f}m", fill=(255, 255, 0), font=F)
    f0, f1 = v[0]["frame"] - 15, v[-1]["frame"] + 15
    xy = np.array([d["world"][:2] for d in v]) - c0
    fig, ax = plt.subplots(1, 3, figsize=(11.3, 3.2))
    for j, w in tr.items():
        if j is None: continue
        w = [d for d in w if f0 <= d["frame"] <= f1]
        if not w: continue
        wxy = np.array([d["world"][:2] for d in w]) - c0
        near = min(np.min(np.linalg.norm(xy - p, axis=1)) for p in wxy)
        if near > 1.5: continue
        f = [d["frame"] for d in w]; lw, ms = (2.5, 3) if j == i else (0.8, 1.5)
        ax[0].plot(f, wxy[:, 0], "-o", color=col(j), lw=lw, ms=ms); ax[0].text(f[-1], wxy[-1, 0], str(j), fontsize=7, color=col(j))
        ax[1].plot(f, wxy[:, 1], "-o", color=col(j), lw=lw, ms=ms); ax[1].text(f[-1], wxy[-1, 1], str(j), fontsize=7, color=col(j))
        ax[2].plot(wxy[:, 1], wxy[:, 0], "-o", color=col(j), lw=lw, ms=ms); ax[2].text(wxy[-1, 1], wxy[-1, 0], str(j), fontsize=7, color=col(j))
    ax[0].set_title(f"track {i}: x(t)"); ax[1].set_title("y(t)"); ax[2].set_title("xy"); ax[2].invert_xaxis(); ax[2].set_aspect("equal", "datalim")
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); fig.savefig(out / "_plot.png", dpi=80); plt.close(fig)
    plot = Image.open(out / "_plot.png")
    sheet = Image.new("RGB", (max(strip.width, plot.width), strip.height + plot.height), (255, 255, 255))
    sheet.paste(strip, (0, 0)); sheet.paste(plot, (0, strip.height)); sheet.save(out / (f"t{i:03d}.jpg" if lo < 0 else f"t{i:03d}_{lo}.jpg"), quality=85)
print("sheets:", len(ids))
