#!/usr/bin/env python3
r"""Converts a frame range of one CODa sequence (UT Campus Object Dataset,
Clearpath Husky robot) into engine JSONL: setup.json, inputs.jsonl and
truth.jsonl, ready for `factor-tracker replay` / `evaluate`.

Frames and conventions:
- reference frame: the Ouster OS1-128 (`os1`, x forward, y left, z up);
- tracking frame: `world`, from `poses/dense_global/<seq>.txt` (one
  os1->world pose per LiDAR frame, SLAM-optimised, globally consistent);
- camera: stereo-rectified cam0 (`2d_rect`), pinhole from the rectified
  projection matrix, `camera_to_reference = inverse(R_rect * T_os1_cam0)`.
  LiDAR and cam0 are hardware-triggered, so both share the os1 timestamp.

LiDAR detections are either detector output (`--lidar-detections`, CODa-style
JSON with scores, e.g. from detect_lidar_cpu.py) or, by default, the CODa 3D
boxes used as oracle detections (positions at the box centre, shape as
explicit corners). Camera detections are read
from per-frame KITTI 2D detection files (`<frame:06d>.txt`, for example the
output of scripts/runpod/detect_camera_yolo.py --image-dir). Ground truth is
the same 3D boxes with their instance ids, mapped to the world frame.

usage: convert_coda.py --coda-root CODa --sequence 20 --first 8640 --last 8800 \
           --camera-detections det/coda20 --out results/coda20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
LIDAR_SENSOR, CAMERA_SENSOR = 1, 2
# Engine class ids follow the KITTI adapter (1 car, 2 pedestrian, 3 cyclist, 5 truck).
CODA_CLASSES = {
    "Pedestrian": 2,
    "Car": 1,
    "Pickup Truck": 1,
    "Service Vehicle": 1,
    "Utility Vehicle": 1,
    "Golf Cart": 1,
    "Delivery Truck": 5,
    "Bus": 5,
    "Scooter": 3,
    "Segway": 3,
    "Skateboard": 3,
    "Motorcycle": 3,
}
KITTI_NAMES = {"Car": 1, "Van": 4, "Truck": 5, "Pedestrian": 2, "Person_sitting": 6, "Cyclist": 3, "Tram": 7, "Misc": 8}


def read_yaml_matrix(path: Path, key: str) -> np.ndarray:
    """Reads `key: {rows, cols, data: [...]}` from the simple CODa calibration YAML."""
    text = path.read_text()
    block = text[text.index(key + ":") :]
    data = block[block.index("data:") + len("data:") :]
    values = data[data.index("[") + 1 : data.index("]")].replace("\n", " ")
    numbers = np.array([float(v) for v in values.replace(",", " ").split()])
    rows = int(block.split("rows:")[1].split()[0])
    cols = int(block.split("cols:")[1].split()[0])
    return numbers.reshape(rows, cols)


def orthonormalize(transform: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(transform[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, 2] *= -1
        rotation = u @ vt
    out = transform.copy()
    out[:3, :3] = rotation
    out[3] = [0, 0, 0, 1]
    return out


def seconds_to_ns(text: str) -> int:
    whole, _, fraction = text.strip().partition(".")
    return int(whole) * 1_000_000_000 + int((fraction + "000000000")[:9])


def quaternion_pose(values) -> np.ndarray:
    x, y, z, qw, qx, qy, qz = values
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    pose = np.eye(4)
    pose[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    pose[:3, 3] = [x, y, z]
    return pose


def box_corners(box: dict) -> np.ndarray:
    """8x3 corner offsets from the centre: 0-3 bottom, 4-7 top (Box3DShape order)."""
    l, w, h = box["l"], box["w"], box["h"]
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    z = np.array([-h, -h, -h, -h, h, h, h, h]) / 2
    cr, sr, cp, sp, cy, sy = (math.cos(box["r"]), math.sin(box["r"]), math.cos(box["p"]), math.sin(box["p"]),
                              math.cos(box["y"]), math.sin(box["y"]))
    rotation = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    return (rotation @ np.vstack([x, y, z])).T


def truth_id(box: dict) -> int:
    """Instance ids are `<class>:<n>`, numbered per class; make them unique."""
    return CODA_CLASSES[box["classId"]] * 1_000_000 + list(CODA_CLASSES).index(box["classId"]) * 10_000 + int(
        box["instanceId"].rsplit(":", 1)[1]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coda-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--camera-detections", type=Path, help="per-frame KITTI 2D detection files")
    parser.add_argument(
        "--lidar-detections",
        type=Path,
        help="per-frame CODa-style JSON detections with scores (e.g. detect_lidar_cpu.py); default: oracle boxes",
    )
    parser.add_argument("--lidar-score-threshold", type=float, default=0.4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--default-covariance", type=float, default=0.04, help="LiDAR position variance (m^2)")
    parser.add_argument(
        "--max-range",
        type=float,
        default=80.0,
        help="drop boxes farther than this from the LiDAR (some CODa boxes drift to km away)",
    )
    args = parser.parse_args()

    root, seq = args.coda_root, args.sequence
    calib = root / "calibrations" / seq
    os1_to_cam0 = orthonormalize(read_yaml_matrix(calib / "calib_os1_to_cam0.yaml", "extrinsic_matrix"))
    rect = np.eye(4)
    rect[:3, :3] = read_yaml_matrix(calib / "calib_cam0_intrinsics.yaml", "rectification_matrix")
    projection = read_yaml_matrix(calib / "calib_cam0_intrinsics.yaml", "projection_matrix")
    size_text = (calib / "calib_cam0_intrinsics.yaml").read_text()
    width = int(size_text.split("image_width:")[1].split()[0])
    height = int(size_text.split("image_height:")[1].split()[0])
    os1_to_rect = orthonormalize(rect @ os1_to_cam0)
    rect_to_os1 = np.linalg.inv(os1_to_rect)

    timestamps = [seconds_to_ns(line) for line in (root / "timestamps" / f"{seq}.txt").read_text().split()]
    pose_rows = [line.split() for line in (root / "poses" / "dense_global" / f"{seq}.txt").read_text().splitlines() if line.strip()]
    frames = range(args.first, args.last + 1)
    poses = {f: quaternion_pose([float(v) for v in pose_rows[f][1:8]]) for f in frames}

    setup = {
        "tracking_frame": "world",
        "lidar_frame": "os1",
        "camera_frame": "cam0_rect",
        "lidar_to_reference": np.eye(4).reshape(-1).tolist(),
        "camera_to_reference": rect_to_os1.reshape(-1).tolist(),
        "intrinsics": {
            "fx": projection[0, 0], "fy": projection[1, 1], "cx": projection[0, 2], "cy": projection[1, 2],
            "skew": projection[0, 1], "image_width": width, "image_height": height,
        },
        "near_plane": 0.3,
        "poses": [{"timestamp_ns": timestamps[f], "transform": poses[f].reshape(-1).tolist()} for f in frames],
        "pose_max_gap_ns": 200_000_000,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "setup.json").write_text(json.dumps(setup, indent=1))

    inputs, truth = [], []
    for f in frames:
        t = timestamps[f]
        label_path = root / "3d_bbox" / "os1" / seq / f"3d_bbox_os1_{seq}_{f}.json"
        labels = json.loads(label_path.read_text())["3dbbox"] if label_path.exists() else []
        labels = [
            b for b in labels
            if b["classId"] in CODA_CLASSES and math.hypot(b["cX"], b["cY"], b["cZ"]) <= args.max_range
        ]
        if args.lidar_detections is None:
            boxes = labels
        else:
            path = args.lidar_detections / f"3d_bbox_os1_{seq}_{f}.json"
            boxes = json.loads(path.read_text())["3dbbox"] if path.exists() else []
            boxes = [
                b for b in boxes
                if b["classId"] in CODA_CLASSES and b.get("score", 1.0) >= args.lidar_score_threshold
                and math.hypot(b["cX"], b["cY"], b["cZ"]) <= args.max_range
            ]
        detections = []
        for b in boxes:
            detections.append({
                "position": [b["cX"], b["cY"], b["cZ"]],
                "covariance": (np.eye(3) * args.default_covariance).reshape(-1).tolist(),
                "class": CODA_CLASSES[b["classId"]],
                "score": b.get("score", 1.0),
                "shape": {"height": b["h"], "width": b["w"], "length": b["l"], "rotation_y": 0.0,
                          "corners": box_corners(b).reshape(-1).tolist()},
            })
        objects = []
        for b in labels:
            world = poses[f] @ np.array([b["cX"], b["cY"], b["cZ"], 1.0])
            objects.append({"truth_id": truth_id(b), "class": CODA_CLASSES[b["classId"]],
                            "position": world[:3].tolist()})
        inputs.append({"type": "lidar_batch", "schema_version": SCHEMA_VERSION, "sensor_id": LIDAR_SENSOR,
                       "sequence": f, "event_time_ns": t, "arrival_time_ns": t, "frame_id": "os1",
                       "detections": detections})
        truth.append({"type": "truth3d", "schema_version": SCHEMA_VERSION, "timestamp_ns": t, "objects": objects})
        if args.camera_detections is not None:
            camera = []
            path = args.camera_detections / f"{f:06d}.txt"
            for index, line in enumerate(path.read_text().splitlines() if path.exists() else []):
                tokens = line.split()
                camera.append({"detection_id": index, "bbox": [float(v) for v in tokens[4:8]],
                               "class": KITTI_NAMES.get(tokens[0]), "score": float(tokens[-1]) if len(tokens) >= 16 else 1.0})
            inputs.append({"type": "camera_batch", "schema_version": SCHEMA_VERSION, "sensor_id": CAMERA_SENSOR,
                           "sequence": f, "event_time_ns": t, "arrival_time_ns": t, "frame_id": "cam0_rect",
                           "detections": camera})

    with open(args.out / "inputs.jsonl", "w") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in inputs)
    with open(args.out / "truth.jsonl", "w") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in truth)
    print(f"{seq}: frames {args.first}-{args.last}, {len(inputs)} input records, {len(truth)} truth frames -> {args.out}")


if __name__ == "__main__":
    main()
