"""All detections with IDs drawn on one camera image. usage: frameview.py labels.json out_dir camera frame [frame ...]"""
import json, os, sys, colorsys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent)); from cams import Cams, M
SEQ = Path(os.environ["GT_SEQ"])  # directory written by grandtour.py prepare (+ inputs.jsonl)
L = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); cam = "alphasense_" + sys.argv[3]
inputs = {json.loads(l)["sequence"]: json.loads(l) for l in open(SEQ / "inputs.jsonl") if '"lidar_batch"' in l}
scan_t = {int(k): v for k, v in json.load(open(SEQ / "frames.json"))["scans"].items()}
C = Cams(); byf = {}
for d in L: byf.setdefault(d["frame"], []).append(d)
def col(i):
    if i is None or i < 0: return (128, 128, 128)
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1, 0.9, 1.0); return (int(b * 255), int(g * 255), int(r * 255))
for f in map(int, sys.argv[4:]):
    img = None
    for d in byf.get(f, []):
        det = inputs[f]["detections"][d["k"]]
        corners = np.array(det["shape"]["corners"]).reshape(8, 3) + det["position"]
        for c, k, (x1, y1, x2, y2), _ in C.boxes(scan_t[f], corners):
            if c != cam: continue
            if img is None: img = cv2.convertScaleAbs(cv2.imread(str(M / "images" / cam / f"{k:06d}.jpeg")), alpha=2.0, beta=10)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col(d["id"]), 3)
            cv2.putText(img, str(d["id"]), (int(x1) + 3, int(y1) + 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col(d["id"]), 3)
            cv2.putText(img, f"{np.hypot(*d['base'][:2]):.1f}m", (int(x1) + 3, int(y2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col(d["id"]), 2)
    if img is not None:
        cv2.putText(img, f"{sys.argv[3]} {f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imwrite(str(out / f"{sys.argv[3]}_{f}.jpg"), cv2.resize(img, (img.shape[1] * 2 // 3, img.shape[0] * 2 // 3)), [cv2.IMWRITE_JPEG_QUALITY, 85])
