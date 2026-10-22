#!/usr/bin/env python3
"""Converts the KITTI tracking benchmark into the per-sequence layout read by
`factor-tracker prepare` (see docs/datasets.md).

Input (the official archives unpacked into one root):

    <kitti-root>/training/calib/0000.txt
    <kitti-root>/training/oxts/0000.txt
    <kitti-root>/training/label_02/0000.txt      (optional, truth)
    <kitti-root>/training/image_02/0000/000000.png (optional, image size)

Output, one directory per sequence:

    <out>/0000/calib.txt  times.txt  poses.txt  image_size.txt
    <out>/0000/labels_lidar/000000.txt ...
    <out>/0000/detections_lidar/000000.txt ...   (detector output, or oracle labels)
    <out>/0000/detections_camera/000000.txt ...

The tracking benchmark ships no per-frame timestamps, so `times.txt` uses the
nominal 10 Hz frame period. Poses are derived from OXTS with the KITTI devkit
formula and expressed as rectified camera 0 -> world, first frame at the origin.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kitti_tools  # noqa: E402

FRAME_PERIOD_S = 0.1
DEFAULT_IMAGE_SIZE = (1242, 375)


def write_frames(directory: Path, frame_count: int, lines_by_frame: dict[int, list[str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for frame in range(frame_count):
        lines = lines_by_frame.get(frame, [])
        (directory / f"{frame:06d}.txt").write_text("".join(line + "\n" for line in lines))


def copy_detections(source: Path, target: Path, frame_count: int) -> None:
    """Copies detector output, writing an empty file for frames without one."""
    target.mkdir(parents=True, exist_ok=True)
    for frame in range(frame_count):
        name = f"{frame:06d}.txt"
        if (source / name).exists():
            shutil.copyfile(source / name, target / name)
        else:
            (target / name).write_text("")


def convert_sequence(args: argparse.Namespace, sequence: str) -> None:
    split = args.kitti_root / args.split
    out = args.out / sequence
    out.mkdir(parents=True, exist_ok=True)

    calib_path = split / "calib" / f"{sequence}.txt"
    calib = kitti_tools.TrackingCalib.read(calib_path)
    shutil.copyfile(calib_path, out / "calib.txt")

    oxts = kitti_tools.read_oxts(split / "oxts" / f"{sequence}.txt")
    frame_count = len(oxts)
    (out / "times.txt").write_text(
        "".join(f"{frame * FRAME_PERIOD_S:.6f}\n" for frame in range(frame_count))
    )
    poses = kitti_tools.cam0_poses(oxts, calib)
    (out / "poses.txt").write_text("".join(kitti_tools.format_pose(p) + "\n" for p in poses))

    first_image = split / "image_02" / sequence / "000000.png"
    width, height = kitti_tools.png_size(first_image) if first_image.exists() else DEFAULT_IMAGE_SIZE
    (out / "image_size.txt").write_text(f"{width} {height}\n")

    label_path = split / "label_02" / f"{sequence}.txt"
    labels = kitti_tools.split_tracking_labels(label_path) if label_path.exists() else None
    if labels is not None:
        write_frames(out / "labels_lidar", frame_count, labels)

    oracle_objects = None
    if args.oracle:
        if labels is None:
            raise SystemExit(f"--oracle needs labels, {label_path} is missing")
        # Tracking labels are image_02 boxes plus camera 0 3D boxes, so the
        # same object lines serve as camera and LiDAR oracle detections.
        oracle_objects = {
            frame: [kitti_tools.label_to_object_line(line) for line in lines]
            for frame, lines in labels.items()
        }
    for sensor in ("lidar", "camera"):
        source = args.detections_root / sensor / sequence if args.detections_root else None
        if source is not None and source.is_dir():
            copy_detections(source, out / f"detections_{sensor}", frame_count)
        elif oracle_objects is not None:
            write_frames(out / f"detections_{sensor}", frame_count, oracle_objects)
        elif source is not None:
            raise SystemExit(f"missing detection directory {source}")

    print(f"{sequence}: {frame_count} frames, image {width}x{height}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequences", nargs="*", help="e.g. 0000 0001 (default: all)")
    parser.add_argument(
        "--detections-root",
        type=Path,
        help="detector output with lidar/<seq>/<frame>.txt and camera/<seq>/<frame>.txt",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="use the labels as oracle detections (for every sensor missing from --detections-root)",
    )
    args = parser.parse_args()

    sequences = args.sequences or sorted(
        path.stem for path in (args.kitti_root / args.split / "calib").glob("*.txt")
    )
    if not sequences:
        raise SystemExit("no sequences found")
    for sequence in sequences:
        convert_sequence(args, sequence)


if __name__ == "__main__":
    main()
