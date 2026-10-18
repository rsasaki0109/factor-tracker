"""Unit tests for kitti_tools (run: python3 -m unittest discover -s scripts/kitti)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import kitti_tools

CALIB = """\
P0: 700 0 600 0 0 700 180 0 0 0 1 0
P2: 700 0 600 44 0 700 180 0.2 0 0 1 0.003
R_rect 1 0 0 0 1 0 0 0 1
Tr_velo_cam 0 -1 0 0 0 0 -1 -0.08 1 0 0 -0.27
Tr_imu_velo 1 0 0 -0.8 0 1 0 0.3 0 0 1 -0.8
"""


def write(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


class KittiToolsTest(unittest.TestCase):
    def setUp(self):
        self.calib = kitti_tools.TrackingCalib.read(write(CALIB))

    def test_reads_tracking_calibration(self):
        self.assertEqual(self.calib.P2[0, 3], 44.0)
        np.testing.assert_allclose(self.calib.R_rect, np.eye(3))
        self.assertEqual(self.calib.Tr_imu_velo[0, 3], -0.8)

    def test_lidar_box_round_trips_to_kitti_camera_convention(self):
        # A car 15 m ahead, 2 m to the left, centre 0.9 m below the sensor,
        # heading along the velodyne x axis.
        box = np.array([[15.0, 2.0, -0.9, 4.0, 1.8, 1.5, 0.0]])
        (line,) = kitti_tools.lidar_boxes_to_kitti_lines(box, np.array([0.8]), ["Car"], self.calib, (1242, 375))
        tokens = line.split()
        self.assertEqual(len(tokens), 16)
        self.assertEqual(tokens[0], "Car")
        h, w, l = (float(v) for v in tokens[8:11])
        self.assertEqual((h, w, l), (1.5, 1.8, 4.0))
        location = np.array([float(v) for v in tokens[11:14]])
        bottom_velo = np.linalg.inv(self.calib.velo_to_cam0) @ np.append(location, 1.0)
        np.testing.assert_allclose(bottom_velo[:3], [15.0, 2.0, -0.9 - 0.75], atol=1e-3)
        # Facing forward (velodyne +x == camera +z) is rotation_y = -pi/2.
        self.assertAlmostEqual(float(tokens[14]), -math.pi / 2, places=5)
        self.assertAlmostEqual(float(tokens[15]), 0.8)
        x1, y1, x2, y2 = (float(v) for v in tokens[4:8])
        self.assertTrue(0 <= x1 < x2 < 600 and 0 <= y1 < y2 <= 374)

    def test_first_pose_is_identity_and_driving_east_moves_camera_forward(self):
        lat, lon = 49.0, 8.4
        rows = np.zeros((2, 30))
        rows[:, 0] = lat
        rows[0, 1] = lon
        rows[1, 1] = lon + 1e-5
        poses = kitti_tools.cam0_poses(rows, self.calib)
        np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-9)
        east = math.cos(lat * math.pi / 180) * 1e-5 * math.pi * kitti_tools.EARTH_RADIUS_M / 180
        np.testing.assert_allclose(poses[1][:3, 3], [0.0, 0.0, east], atol=1e-6)

    def test_drops_dont_care_labels_and_strips_frame_and_id(self):
        path = write(
            "0 -1 DontCare -1 -1 -10 1 2 3 4 -1000 -1000 -1000 -10 -1 -1 -1\n"
            "0 3 Car 0 0 -1.5 10 20 30 40 1.5 1.8 4.0 1.0 1.6 12.0 -1.5\n"
            "2 3 Car 0 0 -1.5 10 20 30 40 1.5 1.8 4.0 1.0 1.6 11.0 -1.5\n"
        )
        frames = kitti_tools.split_tracking_labels(path)
        self.assertEqual(sorted(frames), [0, 2])
        self.assertEqual(len(frames[0]), 1)
        self.assertEqual(
            kitti_tools.label_to_object_line(frames[0][0]),
            "Car 0 0 -1.5 10 20 30 40 1.5 1.8 4.0 1.0 1.6 12.0 -1.5",
        )

    def test_fov_mask_keeps_points_in_front_of_the_camera(self):
        points = np.array([[10.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0], [10.0, 50.0, 0.0, 0.0]])
        np.testing.assert_array_equal(
            kitti_tools.fov_mask(points, self.calib, (1242, 375)), [True, False, False]
        )


if __name__ == "__main__":
    unittest.main()
