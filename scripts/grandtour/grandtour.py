#!/usr/bin/env python3
r"""GrandTour (ANYmal + Livox Mid-360 + global-shutter cameras) -> engine JSONL.

GrandTour (https://grand-tour.leggedrobotics.com, MIT licence, Hugging Face
`leggedrobotics/grand_tour_dataset`) is recorded on an ANYmal quadruped. This
script uses one mission's Zarr data (`<mission>/data`, untarred) and one
camera's JPEG images (`<mission>/images/<camera>`).

Frames and conventions (checked on the data, see docs/datasets.md):
- reference frame `base` (robot base, x forward, z up); the Mid-360 is mounted
  upside down and its points are mapped with the static TF;
- tracking frame `world`: ANYmal leg-inertial odometry `anymal_state_odometry`
  (`p_odom = pose * p_base`);
- camera: the Alphasense images are equidistant fisheye; `prepare` undistorts
  them to a pinhole with the same K, so the engine's rectified pinhole model
  holds. Static TF entries map `base` coordinates into the sensor frame
  (`p_sensor = A_sensor p_base`, with `box_base` composed in).

Two steps around the detectors:

  prepare   undistorted images, accumulated Mid-360 scans in the base frame
            (`scans/<scan>.bin`, float32 x y z intensity), setup.json and
            frames.json for a time window
  assemble  engine inputs.jsonl from LiDAR detections (detect_lidar_cpu.py
            JSON per scan) and camera detections (KITTI 2D text per image),
            each batch at its own sensor timestamp

usage:
  grandtour.py prepare  --mission GT/2024-10-01-11-47-44 --camera alphasense_front_center \
                        --first-image 3700 --last-image 3900 --out gt-seq/eth2
  grandtour.py assemble --out gt-seq/eth2 --lidar-detections det/eth2_lidar --camera-detections det/eth2_cam
Requires NumPy, SciPy, OpenCV and zarr.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

LIDAR = "livox_points_undistorted"
ODOMETRY = "anymal_state_odometry"
SCHEMA_VERSION = 1
LIDAR_SENSOR, CAMERA_SENSOR = 1, 2
# Detector class names (convert_coda.py naming) -> engine class ids.
CLASSES = {"Pedestrian": 2, "Car": 1, "Pickup Truck": 1, "Service Vehicle": 1, "Utility Vehicle": 1,
           "Delivery Truck": 5, "Bus": 5, "Scooter": 3, "Segway": 3, "Skateboard": 3, "Motorcycle": 3}
KITTI_NAMES = {"Car": 1, "Van": 4, "Truck": 5, "Pedestrian": 2, "Person_sitting": 6, "Cyclist": 3, "Tram": 7, "Misc": 8}


def se3(translation, rotation_xyzw) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(rotation_xyzw).as_matrix()
    out[:3, 3] = translation
    return out


def base_to_sensor(tf: dict, frame: str) -> np.ndarray:
    """`p_sensor = A p_base` for a static TF entry (box_base composed in)."""
    def entry(name):
        t = tf[name]
        return se3([t["translation"][k] for k in "xyz"], [t["rotation"][k] for k in "xyzw"])

    a = entry(frame)
    if tf[frame]["base_frame_id"] == "box_base":
        a = a @ entry("box_base")
    return a


class Odometry:
    def __init__(self, group):
        self.t = group["timestamp"][:]
        self.p = group["pose_pos"][:]
        self.q = group["pose_orien"][:]

    def at(self, t: float) -> np.ndarray:
        """World-from-base pose, linear/slerp interpolated."""
        k = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        t0, t1 = self.t[k - 1], self.t[k]
        a = (t - t0) / (t1 - t0)
        rotations = Rotation.from_quat([self.q[k - 1], self.q[k]])
        r = Rotation.from_rotvec(rotations[0].as_rotvec()) * Rotation.from_rotvec(
            ((rotations[0].inv() * rotations[1]).as_rotvec()) * a
        )
        out = np.eye(4)
        out[:3, :3] = r.as_matrix()
        out[:3, 3] = (1 - a) * self.p[k - 1] + a * self.p[k]
        return out


def ns(seconds: float) -> int:
    return int(round(seconds * 1e9))


def prepare(args: argparse.Namespace) -> None:
    root = zarr.open_group(str(args.mission / "data"), mode="r")
    tf = json.loads((args.mission / "data" / "tf" / ".zattrs").read_text())["tf"]
    camera = root[args.camera]
    info = camera.attrs["camera_info"]
    k = np.array(info["K"], dtype=float).reshape(3, 3)
    distortion = np.array(info["D"], dtype=float)
    if info["distortion_model"] != "equidistant":
        raise SystemExit(f"unsupported distortion model {info['distortion_model']}")
    image_times = camera["timestamp"][:]
    lidar = root[LIDAR]
    scan_times = lidar["timestamp"][:]
    odometry = Odometry(root[ODOMETRY])

    t_first, t_last = image_times[args.first_image], image_times[args.last_image]
    scans = [i for i in range(len(scan_times)) if t_first - 0.05 <= scan_times[i] <= t_last + 0.05]
    images = list(range(args.first_image, args.last_image + 1))
    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "scans").mkdir(parents=True, exist_ok=True)

    lidar_to_base = np.linalg.inv(base_to_sensor(tf, "livox_lidar"))
    camera_to_base = np.linalg.inv(base_to_sensor(tf, args.camera))

    # Images: undistort the fisheye to a pinhole with the same K.
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, distortion, np.eye(3), k, (info["width"], info["height"]), cv2.CV_16SC2)
    for index in images:
        image = cv2.imread(str(args.mission / "images" / args.camera / f"{index:06d}.jpeg"))
        cv2.imwrite(str(out / "images" / f"{index:06d}.jpg"), cv2.remap(image, map1, map2, cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Scans: accumulate the last N sweeps in the current base frame (static
    # scene aligned by odometry; moving people smear by up to N/10 s of motion).
    # Read the window once: Zarr chunks hold hundreds of sweeps.
    first = max(0, scans[0] - args.accumulate + 1)
    valid_all = lidar["valid"][first : scans[-1] + 1, 0]
    width = int(valid_all.max())
    points_all = lidar["points"][first : scans[-1] + 1, :width]
    intensity_all = lidar["intensity"][first : scans[-1] + 1, :width]
    for index in scans:
        clouds = []
        world_from_base = odometry.at(scan_times[index])
        for j in range(max(0, index - args.accumulate + 1), index + 1):
            valid = int(valid_all[j - first])
            points = points_all[j - first, :valid].astype(np.float64)
            intensity = intensity_all[j - first, :valid].astype(np.float32)
            to_current = np.linalg.inv(world_from_base) @ odometry.at(scan_times[j]) @ lidar_to_base
            xyz = points @ to_current[:3, :3].T + to_current[:3, 3]
            clouds.append(np.hstack([xyz.astype(np.float32), intensity[:, None]]))
        np.concatenate(clouds).astype(np.float32).tofile(out / "scans" / f"{index:06d}.bin")

    t_pose = np.arange(t_first - 1.0, t_last + 1.0, 0.05)
    setup = {
        "tracking_frame": "world",
        "lidar_frame": "base",
        "camera_frame": args.camera,
        "lidar_to_reference": np.eye(4).reshape(-1).tolist(),
        "camera_to_reference": camera_to_base.reshape(-1).tolist(),
        "intrinsics": {"fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2], "skew": k[0, 1],
                       "image_width": int(info["width"]), "image_height": int(info["height"])},
        "near_plane": 0.3,
        "poses": [{"timestamp_ns": ns(t), "transform": odometry.at(t).reshape(-1).tolist()} for t in t_pose],
        "pose_max_gap_ns": 200_000_000,
    }
    (out / "setup.json").write_text(json.dumps(setup, indent=1))
    frames = {"scans": {str(i): float(scan_times[i]) for i in scans},
              "images": {str(i): float(image_times[i]) for i in images}}
    (out / "frames.json").write_text(json.dumps(frames, indent=1))
    (out / "truth.jsonl").write_text("")
    print(f"prepared {len(scans)} scans and {len(images)} images -> {out}")


def box_corners(box: dict) -> np.ndarray:
    l, w, h, yaw = box["l"], box["w"], box["h"], box["y"]
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    z = np.array([-h, -h, -h, -h, h, h, h, h]) / 2
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * x - s * y, s * x + c * y, z], axis=1)


def assemble(args: argparse.Namespace) -> None:
    frames = json.loads((args.out / "frames.json").read_text())
    records = []
    for scan, t in frames["scans"].items():
        path = args.lidar_detections / f"scan_{int(scan):06d}.json"
        boxes = json.loads(path.read_text())["3dbbox"] if path.exists() else []
        boxes = [b for b in boxes if b["classId"] in CLASSES and b["score"] >= args.lidar_score_threshold
                 and math.hypot(b["cX"], b["cY"]) <= args.max_range]
        records.append({"type": "lidar_batch", "schema_version": SCHEMA_VERSION, "sensor_id": LIDAR_SENSOR,
                        "sequence": int(scan), "event_time_ns": ns(t), "arrival_time_ns": ns(t), "frame_id": "base",
                        "detections": [{
                            "position": [b["cX"], b["cY"], b["cZ"]],
                            "covariance": (np.eye(3) * args.default_covariance).reshape(-1).tolist(),
                            "class": CLASSES[b["classId"]], "score": b["score"],
                            "shape": {"height": b["h"], "width": b["w"], "length": b["l"], "rotation_y": 0.0,
                                      "corners": box_corners(b).reshape(-1).tolist()},
                        } for b in boxes]})
    camera_frame = json.loads((args.out / "setup.json").read_text())["camera_frame"]
    for image, t in frames["images"].items():
        path = args.camera_detections / f"{int(image):06d}.txt"
        detections = []
        for index, line in enumerate(path.read_text().splitlines() if path.exists() else []):
            tokens = line.split()
            detections.append({"detection_id": index, "bbox": [float(v) for v in tokens[4:8]],
                               "class": KITTI_NAMES.get(tokens[0]), "score": float(tokens[-1])})
        records.append({"type": "camera_batch", "schema_version": SCHEMA_VERSION, "sensor_id": CAMERA_SENSOR,
                        "sequence": int(image), "event_time_ns": ns(t), "arrival_time_ns": ns(t),
                        "frame_id": camera_frame, "detections": detections})
    records.sort(key=lambda r: (r["arrival_time_ns"], r["type"]))
    with open(args.out / "inputs.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in records)
    print(f"assembled {len(records)} batches -> {args.out / 'inputs.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--mission", type=Path, required=True)
    p.add_argument("--camera", default="alphasense_front_center")
    p.add_argument("--first-image", type=int, required=True)
    p.add_argument("--last-image", type=int, required=True)
    p.add_argument("--accumulate", type=int, default=3, help="Mid-360 sweeps per detector input")
    p.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--lidar-detections", type=Path, required=True)
    a.add_argument("--camera-detections", type=Path, required=True)
    a.add_argument("--lidar-score-threshold", type=float, default=0.4)
    a.add_argument("--max-range", type=float, default=30.0)
    a.add_argument("--default-covariance", type=float, default=0.04)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else assemble(args)


if __name__ == "__main__":
    main()
