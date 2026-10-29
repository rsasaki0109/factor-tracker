"""End-to-end: a moving robot sees one standing and one walking person.

Detections are published in the robot frame (`base_link`) while `base_link`
moves through `odom` at 0.5 m/s. The tracks must come out in `odom` with the
standing person at rest and the walker at 1 m/s, which only holds if the node
applies TF at each detection's stamp. Needs ROS 2 and a built factor-tracker
(FACTOR_TRACKER, default: the workspace's target/release binary).
"""

import os
import threading
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")

from derived_object_msgs.msg import ObjectWithCovarianceArray  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose  # noqa: E402

from factor_tracker_ros.tracker_node import TrackerNode  # noqa: E402

CLI = os.environ.get(
    "FACTOR_TRACKER", str(Path(__file__).resolve().parents[3] / "target" / "release" / "factor-tracker"))
ROBOT_SPEED = 0.5


def people_at(t):
    """World (odom) positions at time t (s): a standing and a walking person."""
    return [(5.0, 1.0, 0.9), (3.0, -2.0 + 1.0 * t, 0.9)]


class Scene(Node):
    def __init__(self):
        super().__init__("scene")
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Detection3DArray, "detections", 10)
        self.tracks, self.smoothed = [], []
        self.create_subscription(ObjectWithCovarianceArray, "tracks", self.tracks.append, 10)
        self.create_subscription(ObjectWithCovarianceArray, "tracks/smoothed", self.smoothed.append, 10)
        self.t0 = self.get_clock().now()

    def step(self, publish_detections):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9
        robot_x = ROBOT_SPEED * t
        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = robot_x
        tf.transform.rotation.w = 1.0
        self.tf.sendTransform(tf)
        if not publish_detections:
            return
        msg = Detection3DArray()
        msg.header = tf.header
        msg.header.frame_id = "base_link"
        for x, y, z in people_at(t):
            detection = Detection3D()
            detection.bbox.center.position.x = x - robot_x
            detection.bbox.center.position.y = y
            detection.bbox.center.position.z = z
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = "person"
            hypothesis.hypothesis.score = 0.9
            detection.results.append(hypothesis)
            msg.detections.append(detection)
        self.pub.publish(msg)


def run_scene(*parameters):
    """Runs the node on the scene for 2 s; returns the scene with the messages it received."""
    tracker = TrackerNode(parameter_overrides=[
        Parameter("engine_executable", value=CLI),
        Parameter("tracking_frame", value="odom"),
        *parameters,
    ])
    scene = Scene()
    executor = MultiThreadedExecutor()
    executor.add_node(tracker)
    executor.add_node(scene)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    # TF at 50 Hz; detections at 10 Hz, stamped after the latest TF (as on
    # a real robot, where odometry lags the sensor slightly).
    for tick in range(200):
        scene.step(publish_detections=False)
        time.sleep(0.01)
        if tick % 5 == 4:
            scene.step(publish_detections=True)
        time.sleep(0.01)
    time.sleep(0.5)
    executor.shutdown()
    tracker.destroy_node()
    scene.destroy_node()
    return scene


def test_smoothed_tracks_are_published_late_with_the_same_ids():
    if not Path(CLI).exists():
        pytest.skip(f"factor-tracker not built at {CLI}")
    rclpy.init()
    try:
        scene = run_scene(Parameter("smooth_lag", value=5))
        assert len(scene.tracks) > 20 and len(scene.smoothed) > 15, (len(scene.tracks), len(scene.smoothed))
        # Real-time messages keep coming; smoothed ones trail by five scans
        # (counted in scans: the scene's wall-clock rate varies under load).
        stamps = [(m.header.stamp.sec, m.header.stamp.nanosec) for m in scene.tracks]
        last_smoothed = max((m.header.stamp.sec, m.header.stamp.nanosec) for m in scene.smoothed)
        assert 4 <= len(stamps) - 1 - stamps.index(last_smoothed) <= 6
        by_stamp = {(m.header.stamp.sec, m.header.stamp.nanosec): m for m in scene.tracks}
        compared = 0
        for message in scene.smoothed:
            realtime = by_stamp.get((message.header.stamp.sec, message.header.stamp.nanosec))
            if realtime is not None:
                assert {o.id for o in message.objects} == {o.id for o in realtime.objects}
                compared += 1
        assert compared > 10
        walking = min(scene.smoothed[-1].objects, key=lambda o: abs(o.pose.pose.position.x - 3.0))
        assert abs(walking.twist.twist.linear.y - 1.0) < 0.15
    finally:
        rclpy.shutdown()


def test_tracks_are_in_odom_with_velocity():
    if not Path(CLI).exists():
        pytest.skip(f"factor-tracker not built at {CLI}")
    rclpy.init()
    try:
        tracker = TrackerNode(parameter_overrides=[
            Parameter("engine_executable", value=CLI),
            Parameter("tracking_frame", value="odom"),
        ])
        scene = Scene()
        executor = MultiThreadedExecutor()
        executor.add_node(tracker)
        executor.add_node(scene)
        spinner = threading.Thread(target=executor.spin, daemon=True)
        spinner.start()
        # TF at 50 Hz; detections at 10 Hz, stamped after the latest TF (as on
        # a real robot, where odometry lags the sensor slightly).
        for tick in range(200):
            scene.step(publish_detections=False)
            time.sleep(0.01)
            if tick % 5 == 4:
                scene.step(publish_detections=True)
            time.sleep(0.01)
        time.sleep(0.5)
        executor.shutdown()

        assert len(scene.tracks) > 20, f"only {len(scene.tracks)} track messages"
        last = scene.tracks[-1]
        assert last.header.frame_id == "odom"
        assert len(last.objects) == 2, [o.id for o in last.objects]
        standing = min(last.objects, key=lambda o: abs(o.pose.pose.position.x - 5.0))
        walking = min(last.objects, key=lambda o: abs(o.pose.pose.position.x - 3.0))
        assert abs(standing.pose.pose.position.y - 1.0) < 0.2
        assert abs(standing.twist.twist.linear.x) < 0.15 and abs(standing.twist.twist.linear.y) < 0.15
        assert abs(walking.twist.twist.linear.y - 1.0) < 0.15
        assert abs(walking.twist.twist.linear.x) < 0.15
        assert standing.classification == standing.CLASSIFICATION_PEDESTRIAN
        # Identities stay the same over the run.
        ids = {o.id for message in scene.tracks[len(scene.tracks) // 2:] for o in message.objects}
        assert len(ids) == 2, ids
        tracker.destroy_node()
        scene.destroy_node()
    finally:
        rclpy.shutdown()
