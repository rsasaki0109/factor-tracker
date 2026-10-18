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
import re
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


def frame_name(image: Path) -> str:
    """`000123` for KITTI names, and for names ending in a frame number such as `2d_rect_cam0_20_8650`."""
    digits = re.search(r"(\d+)$", image.stem)
    return f"{int(digits.group(1)):06d}" if digits else image.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path)
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="any directory of images instead of KITTI; output files are named by the trailing frame number",
    )
    parser.add_argument("--out-dir", type=Path, help="output directory for --image-dir")
    parser.add_argument("--split", default="training")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="yolo11x.pt", help="Ultralytics weights (downloaded if missing)")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args()

    if args.image_dir is not None:
        if args.out_dir is None:
            parser.error("--image-dir needs --out-dir")
        jobs = [(args.image_dir.name, sorted(p for p in args.image_dir.iterdir() if p.suffix in (".png", ".jpg")), args.out_dir)]
    else:
        if args.kitti_root is None or args.out is None:
            parser.error("--kitti-root and --out are required without --image-dir")
        split = args.kitti_root / args.split
        sequences = args.sequences or sorted(p.name for p in (split / "image_02").iterdir() if p.is_dir())
        jobs = [(seq, sorted((split / "image_02" / seq).glob("*.png")), args.out / "camera" / seq) for seq in sequences]
    model = YOLO(args.model)

    for seq, images, out_dir in jobs:
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
                (out_dir / f"{frame_name(image)}.txt").write_text("".join(line + "\n" for line in lines))
        elapsed = time.time() - started
        print(f"{seq}: {len(images)} images in {elapsed:.1f} s ({len(images) / max(elapsed, 1e-9):.1f} Hz)")


if __name__ == "__main__":
    main()
