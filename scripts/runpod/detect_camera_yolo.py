#!/usr/bin/env python3
"""Runs an Ultralytics COCO detector over KITTI tracking image_02 frames.

Writes one 16-field KITTI 2D detection file per frame (score last) to
<out>/camera/<seq>/<frame>.txt. COCO classes are mapped to the KITTI class
names the engine knows; unmapped COCO classes are dropped. The mapping is an
approximation (for example a COCO `bicycle` box covers the bicycle only, a
KITTI `Cyclist` covers bicycle and rider).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kitti"))

import kitti_tools  # noqa: E402
from ultralytics import YOLO  # noqa: E402

COCO_TO_KITTI = {
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
    "car": "Car",
    "bus": "Truck",
    "truck": "Truck",
    "train": "Tram",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="yolo11x.pt", help="Ultralytics weights (downloaded if missing)")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args()

    split = args.kitti_root / args.split
    sequences = args.sequences or sorted(p.name for p in (split / "image_02").iterdir() if p.is_dir())
    model = YOLO(args.model)

    for seq in sequences:
        images = sorted((split / "image_02" / seq).glob("*.png"))
        out_dir = args.out / "camera" / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        for start in range(0, len(images), args.batch):
            chunk = images[start : start + args.batch]
            results = model.predict([str(p) for p in chunk], conf=args.score_threshold, verbose=False)
            for image, result in zip(chunk, results):
                lines = []
                for box, score, cls in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                ):
                    name = COCO_TO_KITTI.get(result.names[cls])
                    if name is not None:
                        lines.append(kitti_tools.image_box_line(name, box, score))
                (out_dir / f"{image.stem}.txt").write_text("".join(line + "\n" for line in lines))
        elapsed = time.time() - started
        print(f"{seq}: {len(images)} images in {elapsed:.1f} s ({len(images) / max(elapsed, 1e-9):.1f} Hz)")


if __name__ == "__main__":
    main()
