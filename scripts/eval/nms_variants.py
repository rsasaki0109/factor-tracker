#!/usr/bin/env python3
"""Re-applies non-maximum suppression to raw detector output.

Input: per-scan CODa-style JSON written by detect_lidar_cpu.py with
`--nms-threshold 1.0` (no suppression). Output: one detections root per
variant, readable by scripts/eval/coda_tracker_eval.sh.

Variants (greedy, highest score first, across all classes as in the detector):
  iou<T>     rotated bird's-eye-view IoU above T suppresses (detector default 0.1)
  circle<R>  centre distance below R metres suppresses (CenterPoint's circle NMS)

Boxes below --min-score are dropped first; a lower-scoring box never
suppresses a higher one, so this does not change what survives above it.

usage: nms_variants.py <raw-root> <out-root> iou0.1 iou0.3 circle0.3 ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coda"))
from detect_lidar_cpu import rotated_nms  # noqa: E402


def circle_nms(xy: np.ndarray, scores: np.ndarray, radius: float) -> list[int]:
    kept = []
    for i in np.argsort(-scores):
        if all(np.hypot(*(xy[i] - xy[j])) >= radius for j in kept):
            kept.append(i)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--min-score", type=float, default=0.3)
    a = ap.parse_args()
    for scan in sorted(a.raw.glob("*/*.json")):
        boxes = [b for b in json.loads(scan.read_text())["3dbbox"] if b["score"] >= a.min_score]
        arr = np.array([[b["cX"], b["cY"], b["cZ"], b["l"], b["w"], b["h"], b["y"]] for b in boxes]).reshape(-1, 7)
        scores = np.array([b["score"] for b in boxes])
        for variant in a.variants:
            if variant.startswith("iou"):
                keep = rotated_nms(arr, scores, float(variant[3:])) if len(boxes) else []
            elif variant.startswith("circle"):
                keep = circle_nms(arr[:, :2], scores, float(variant[6:]))
            else:
                ap.error(f"unknown variant {variant}")
            target = a.out / variant / scan.parent.name / scan.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"3dbbox": [boxes[i] for i in sorted(keep)]}))


if __name__ == "__main__":
    main()
