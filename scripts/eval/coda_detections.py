#!/usr/bin/env python3
"""Pedestrian detection recall and precision of CODa detections.

One-to-one (Hungarian) matching in the bird's-eye view within 0.5 m against
the CODa pedestrian labels of every frame in the given windows; reports
recall, precision and the mean centre error of the matches at several score
thresholds.

usage: coda_detections.py <detections-root> <coda-root> <seq>:<first>:<last> ...
(detections-root/coda<seq>_<first>/3d_bbox_os1_<seq>_<frame>.json)
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def centres(path, threshold=None):
    boxes = json.loads(Path(path).read_text())["3dbbox"]
    return np.array([(b["cX"], b["cY"]) for b in boxes if b["classId"] == "Pedestrian"
                     and (threshold is None or b["score"] >= threshold)]).reshape(-1, 2)


def main():
    dets, coda, windows = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    for th in (0.3, 0.4, 0.5):
        tp = n_det = n_gt = 0
        err = []
        for w in windows:
            seq, first, last = w.split(":")
            for frame in range(int(first), int(last) + 1):
                gt = centres(coda / "3d_bbox" / "os1" / seq / f"3d_bbox_os1_{seq}_{frame}.json")
                det = centres(dets / f"coda{seq}_{first}" / f"3d_bbox_os1_{seq}_{frame}.json", th)
                n_gt += len(gt)
                n_det += len(det)
                if len(gt) and len(det):
                    c = np.linalg.norm(gt[:, None] - det[None], axis=2)
                    rows, cols = linear_sum_assignment(c)
                    ok = c[rows, cols] < 0.5
                    tp += int(ok.sum())
                    err += list(c[rows, cols][ok])
        print(f"detections >= {th}: recall {tp / max(1, n_gt):.3f} precision {tp / max(1, n_det):.3f} "
              f"mean error {np.mean(err) if err else float('nan'):.3f} m")


if __name__ == "__main__":
    main()
