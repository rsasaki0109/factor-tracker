"""All five Alphasense cameras for labelling: crops of a base-frame box at a LiDAR time."""
import json, os, sys
from pathlib import Path
import cv2, numpy as np, zarr
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grandtour import Odometry, base_to_sensor, ODOMETRY
M = Path(os.environ["GT_MISSION"])  # GrandTour mission directory (data/ and images/)
CAMS = ["alphasense_front_center", "alphasense_front_left", "alphasense_front_right", "alphasense_left", "alphasense_right"]
class Cams:
    def __init__(self):
        root = zarr.open_group(str(M / "data"), mode="r")
        tf = json.loads((M / "data" / "tf" / ".zattrs").read_text())["tf"]
        self.odo = Odometry(root[ODOMETRY]); self.c = {}
        for c in CAMS:
            if not (M / "images" / c).exists(): continue
            g = root[c]; info = g.attrs["camera_info"]
            self.c[c] = dict(K=np.array(info["K"], float).reshape(3, 3), D=np.array(info["D"], float).reshape(4, 1),
                             A=base_to_sensor(tf, c), t=g["timestamp"][:], w=info["width"], h=info["height"])
    def boxes(self, t_scan, corners_base):
        """Yields (camera, image index, bbox, centrality) for every camera that sees the box."""
        Pw = self.odo.at(t_scan); world = corners_base @ Pw[:3, :3].T + Pw[:3, 3]
        for c, cam in self.c.items():
            k = int(np.argmin(np.abs(cam["t"] - t_scan)))
            if abs(cam["t"][k] - t_scan) > 0.1: continue
            Pi = np.linalg.inv(self.odo.at(cam["t"][k])); base = world @ Pi[:3, :3].T + Pi[:3, 3]
            p = base @ cam["A"][:3, :3].T + cam["A"][:3, 3]
            if np.any(p[:, 2] < 0.3): continue
            ang = np.degrees(np.arctan2(np.hypot(p[:, 0], p[:, 1]), p[:, 2])).max()
            if ang > 80: continue
            uv, _ = cv2.fisheye.projectPoints(p.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam["K"], cam["D"])
            uv = uv.reshape(-1, 2); x1, y1 = uv.min(0); x2, y2 = uv.max(0)
            if x2 < 0 or y2 < 0 or x1 > cam["w"] or y1 > cam["h"]: continue
            yield c, k, (x1, y1, x2, y2), ang
    def crop(self, t_scan, corners_base, size=(110, 180)):
        best = min(self.boxes(t_scan, corners_base), key=lambda b: b[3], default=None)
        if best is None: return None, None
        c, k, (x1, y1, x2, y2), _ = best
        img = cv2.imread(str(M / "images" / c / f"{k:06d}.jpeg"))
        bx = (x1, y1, x2, y2); w, h = x2 - x1, y2 - y1; x1 -= .3 * w; x2 += .3 * w; y1 -= .1 * h; y2 += .05 * h
        x1, y1 = int(max(x1, 0)), int(max(y1, 0)); x2, y2 = int(min(x2, img.shape[1])), int(min(y2, img.shape[0]))
        if x2 - x1 < 4 or y2 - y1 < 4: return None, None
        img = cv2.convertScaleAbs(img, alpha=2.0, beta=10)
        cv2.rectangle(img, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 255), max(2, int((x2 - x1) / 60)))
        crop = img[y1:y2, x1:x2]
        return cv2.cvtColor(cv2.resize(crop, size), cv2.COLOR_BGR2RGB), c.replace("alphasense_", "")
