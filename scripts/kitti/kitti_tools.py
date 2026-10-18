"""KITTI tracking helpers shared by the conversion and detector scripts.

Only NumPy is required. Frames follow `docs/datasets.md`: KITTI 3D boxes are
in the rectified camera 0 frame, location is the bottom centre of the box and
`rotation_y` is the yaw about the camera y axis.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

# Equatorial radius used by the KITTI devkit (`convertOxtsToPose`).
EARTH_RADIUS_M = 6378137.0


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[: matrix.shape[0], : matrix.shape[1]] = matrix
    return out


class TrackingCalib:
    """Calibration of one KITTI tracking sequence (`calib/<seq>.txt`)."""

    def __init__(self, entries: dict[str, np.ndarray]):
        self.P2 = entries["P2"].reshape(3, 4)
        self.R_rect = entries["R_rect"].reshape(3, 3)
        self.Tr_velo_cam = entries["Tr_velo_cam"].reshape(3, 4)
        self.Tr_imu_velo = entries["Tr_imu_velo"].reshape(3, 4)

    @classmethod
    def read(cls, path: Path) -> "TrackingCalib":
        entries = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            key, _, rest = line.replace(":", " ", 1).partition(" ")
            entries[key] = np.array([float(v) for v in rest.split()])
        return cls(entries)

    @property
    def velo_to_cam0(self) -> np.ndarray:
        """4x4 velodyne -> rectified camera 0 (`R_rect * Tr_velo_cam`)."""
        return _homogeneous(self.R_rect) @ _homogeneous(self.Tr_velo_cam)

    @property
    def imu_to_cam0(self) -> np.ndarray:
        return self.velo_to_cam0 @ _homogeneous(self.Tr_imu_velo)


def read_oxts(path: Path) -> np.ndarray:
    """One row of 30 OXTS values per frame."""
    rows = [
        [float(v) for v in line.split()]
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    return np.array(rows)


def oxts_to_imu_poses(oxts: np.ndarray) -> list[np.ndarray]:
    """IMU -> world poses, following the KITTI devkit `convertOxtsToPose`."""
    scale = math.cos(oxts[0, 0] * math.pi / 180.0)
    poses = []
    for lat, lon, alt, roll, pitch, yaw in oxts[:, :6]:
        tx = scale * lon * math.pi * EARTH_RADIUS_M / 180.0
        ty = scale * EARTH_RADIUS_M * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
        rx = np.array(
            [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
        )
        ry = np.array(
            [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
        )
        rz = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        )
        pose = np.eye(4)
        pose[:3, :3] = rz @ ry @ rx
        pose[:3, 3] = [tx, ty, alt]
        poses.append(pose)
    return poses


def cam0_poses(oxts: np.ndarray, calib: TrackingCalib) -> list[np.ndarray]:
    """Rectified camera 0 -> world poses, with the first pose at the origin.

    This is the convention of the KITTI odometry `poses.txt` that the engine
    adapter expects.
    """
    cam0_to_imu = np.linalg.inv(calib.imu_to_cam0)
    world_from_cam0 = [pose @ cam0_to_imu for pose in oxts_to_imu_poses(oxts)]
    origin = np.linalg.inv(world_from_cam0[0])
    return [origin @ pose for pose in world_from_cam0]


def format_pose(pose: np.ndarray) -> str:
    return " ".join(f"{v:.12e}" for v in pose[:3, :].reshape(-1))


def split_tracking_labels(path: Path) -> dict[int, list[str]]:
    """Groups `label_02/<seq>.txt` lines by frame, dropping `DontCare`."""
    frames: dict[int, list[str]] = {}
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 17 or tokens[2] == "DontCare" or int(tokens[1]) < 0:
            continue
        frames.setdefault(int(tokens[0]), []).append(" ".join(tokens))
    return frames


def label_to_object_line(label_line: str) -> str:
    """Strips `frame track_id` from a tracking label (15-field object line)."""
    return " ".join(label_line.split()[2:17])


def png_size(path: Path) -> tuple[int, int]:
    """Width and height from a PNG header, without an imaging dependency."""
    with open(path, "rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def project(points_cam0: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Projects Nx3 rectified camera 0 points with `P2` to Nx2 pixels."""
    homogeneous = np.hstack([points_cam0, np.ones((len(points_cam0), 1))]) @ p2.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def fov_mask(points_velo: np.ndarray, calib: TrackingCalib, image_size: tuple[int, int]) -> np.ndarray:
    """Points in front of camera 2 that project inside the image."""
    width, height = image_size
    cam = (np.hstack([points_velo[:, :3], np.ones((len(points_velo), 1))]) @ calib.velo_to_cam0.T)[:, :3]
    in_front = cam[:, 2] > 0.1
    pixels = np.full((len(cam), 2), -1.0)
    pixels[in_front] = project(cam[in_front], calib.P2)
    return (
        in_front
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def box_corners_cam0(dims_hwl: np.ndarray, location: np.ndarray, rotation_y: float) -> np.ndarray:
    """8x3 corners of a KITTI camera-frame box (location = bottom centre)."""
    h, w, l = dims_hwl
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2.0
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2.0
    c, s = math.cos(rotation_y), math.sin(rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (rotation @ np.vstack([x, y, z])).T + location


def lidar_boxes_to_kitti_lines(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    calib: TrackingCalib,
    image_size: tuple[int, int],
) -> list[str]:
    """Converts LiDAR-frame boxes to 16-field KITTI detection lines.

    `boxes` is Nx7 `[x, y, z, dx, dy, dz, heading]` in the velodyne frame with
    `z` at the box centre (the OpenPCDet convention).
    """
    width, height = image_size
    lines = []
    for box, score, name in zip(boxes, scores, class_names):
        x, y, z, dx, dy, dz, heading = (float(v) for v in box[:7])
        bottom = calib.velo_to_cam0 @ np.array([x, y, z - dz / 2.0, 1.0])
        location = bottom[:3]
        rotation_y = float(wrap_angle(np.array(-heading - np.pi / 2.0)))
        alpha = float(wrap_angle(np.array(rotation_y - math.atan2(location[0], location[2]))))
        dims = np.array([dz, dy, dx])
        corners = box_corners_cam0(dims, location, rotation_y)
        if np.all(corners[:, 2] > 0.1):
            pixels = project(corners, calib.P2)
            x1, y1 = np.clip(pixels.min(axis=0), 0, [width - 1, height - 1])
            x2, y2 = np.clip(pixels.max(axis=0), 0, [width - 1, height - 1])
        else:
            x1 = y1 = x2 = y2 = -1.0
        lines.append(
            f"{name} -1 -1 {alpha:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"{dz:.4f} {dy:.4f} {dx:.4f} {location[0]:.4f} {location[1]:.4f} {location[2]:.4f} "
            f"{rotation_y:.6f} {float(score):.4f}"
        )
    return lines


def image_box_line(name: str, box_xyxy, score: float) -> str:
    """16-field KITTI 2D detection line (3D fields set to the unknown values)."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    return (
        f"{name} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"-1 -1 -1 -1000 -1000 -1000 -10 {float(score):.4f}"
    )
