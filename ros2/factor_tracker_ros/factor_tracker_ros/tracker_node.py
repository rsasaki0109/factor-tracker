"""ROS 2 node: 3D detections in, tracked objects (position, velocity, ID) out.

The tracking runs in `factor-tracker stream`, started as a child process.
This node only moves data: it transforms each detection array into the
tracking frame with TF (at the array's stamp), writes it to the engine's
stdin, and publishes every frame the engine writes back. Arrays whose
transform has not arrived yet wait in a queue (in order) for up to
`tf_timeout` seconds instead of blocking the callback.

Subscribes:  detections (vision_msgs/Detection3DArray)
Publishes:   tracks (derived_object_msgs/ObjectWithCovarianceArray)
             tracks/markers (visualization_msgs/MarkerArray)
"""

import json
import os
import subprocess
import tempfile
import threading
from collections import deque

import rclpy
from derived_object_msgs.msg import ObjectWithCovariance, ObjectWithCovarianceArray
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from factor_tracker_ros import convert


def stamp_to_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def ns_to_stamp(ns):
    return Time(nanoseconds=ns).to_msg()


class TrackerNode(Node):
    def __init__(self, **kwargs):
        super().__init__("factor_tracker_node", **kwargs)
        p = self.declare_parameter
        self.tracking_frame = p("tracking_frame", "odom").value
        executable = p("engine_executable", "factor-tracker").value
        mode = p("mode", "lidar_only").value
        # Pedestrian motion prior and association by distance (docs/evaluation.md).
        engine_args = list(p("engine_args", ["--initial-velocity-variance", "1",
                                             "--accel-spectral-density", "0.5",
                                             "--max-association-distance-m", "1.0",
                                             "--recent-track-window-ns", "0"]).value)
        fgo_lag = p("fgo_lag", 0).value
        if fgo_lag > 0:
            engine_args += ["--fgo-lag", str(fgo_lag)]
        self.class_map = convert.parse_class_map(
            p("class_map", ["person:2", "pedestrian:2", "Pedestrian:2"]).value)
        self.keep_unmapped = p("keep_unmapped_classes", False).value
        self.min_score = p("min_score", 0.0).value
        self.default_std = p("default_position_std", 0.15).value
        self.include_tentative = p("publish_tentative", False).value
        self.tf_timeout = Duration(seconds=p("tf_timeout", 0.1).value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.queue = deque()
        self.create_timer(0.02, self.drain_queue)
        self.tracks_pub = self.create_publisher(ObjectWithCovarianceArray, "tracks", 10)
        self.markers_pub = self.create_publisher(MarkerArray, "tracks/markers", 10)
        self.create_subscription(Detection3DArray, "detections", lambda msg: self.queue.append(msg), 10)

        self.setup_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(convert.setup_document(self.tracking_frame), self.setup_file)
        self.setup_file.close()
        command = [executable, "stream", "--setup", self.setup_file.name, "--mode", mode, *engine_args]
        self.get_logger().info("starting engine: " + " ".join(command))
        self.engine = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       text=True, bufsize=1)
        self.sequence = 0
        self.last_marker_ids = set()
        self.reader = threading.Thread(target=self.read_engine, daemon=True)
        self.reader.start()

    def drain_queue(self):
        """Processes queued arrays in arrival order while their transform is
        available; drops the oldest once it has waited longer than
        `tf_timeout`."""
        while self.queue:
            msg = self.queue[0]
            transform = self.lookup(msg)
            if transform is None:
                waited = self.get_clock().now() - Time.from_msg(msg.header.stamp)
                if waited < self.tf_timeout:
                    return
                self.queue.popleft()
                self.get_logger().warn(
                    f"detections dropped: no transform {msg.header.frame_id} -> {self.tracking_frame} "
                    f"within {self.tf_timeout.nanoseconds * 1e-9:.2f} s", throttle_duration_sec=2.0)
                continue
            self.queue.popleft()
            self.on_detections(msg, *transform)

    def lookup(self, msg):
        """(rotation, translation) from the array's frame to the tracking
        frame at its stamp, or None when TF cannot answer yet."""
        if msg.header.frame_id == self.tracking_frame:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0]
        try:
            tf = self.tf_buffer.lookup_transform(self.tracking_frame, msg.header.frame_id,
                                                 Time.from_msg(msg.header.stamp))
        except TransformException:
            return None
        q, t = tf.transform.rotation, tf.transform.translation
        return convert.quaternion_to_matrix(q.x, q.y, q.z, q.w), [t.x, t.y, t.z]

    def on_detections(self, msg, rotation, translation):
        detections = []
        for detection in msg.detections:
            best = max(detection.results, key=lambda r: r.hypothesis.score, default=None)
            label, score = None, 1.0
            if best is not None:
                score = best.hypothesis.score
                label = self.class_map.get(best.hypothesis.class_id)
                if label is None and not self.keep_unmapped:
                    continue
            if score < self.min_score:
                continue
            center = detection.bbox.center.position
            covariance = convert.position_covariance(
                best.pose.covariance if best is not None else [0.0] * 36, self.default_std)
            detections.append({
                "position": convert.transform_point(rotation, translation, [center.x, center.y, center.z]),
                "covariance": convert.rotate_covariance(rotation, covariance),
                "class_label": label,
                "score": score,
            })

        self.sequence += 1
        line = convert.lidar_batch_record(detections, self.tracking_frame, self.sequence,
                                          stamp_to_ns(msg.header.stamp),
                                          self.get_clock().now().nanoseconds)
        try:
            self.engine.stdin.write(line + "\n")
            self.engine.stdin.flush()
        except (BrokenPipeError, ValueError):
            self.get_logger().error("engine process is not running", throttle_duration_sec=2.0)

    def read_engine(self):
        for line in self.engine.stdout:
            frame = convert.tracks_from_output(line, self.include_tentative)
            if frame is not None:
                self.publish(frame)
        self.get_logger().info(f"engine exited with code {self.engine.wait()}")

    def publish(self, frame):
        stamp = ns_to_stamp(frame["timestamp_ns"])
        array = ObjectWithCovarianceArray()
        array.header.stamp = stamp
        array.header.frame_id = self.tracking_frame
        markers = MarkerArray()
        ids = set()
        for track in frame["tracks"]:
            obj = ObjectWithCovariance()
            obj.header = array.header
            obj.id = track["id"] % (1 << 31)  # also a Marker id (int32)
            obj.detection_level = ObjectWithCovariance.OBJECT_TRACKED
            obj.object_classified = track["classification"] != convert.CLASSIFICATION_UNKNOWN
            obj.classification = track["classification"]
            px, py, pz = track["position"]
            vx, vy, vz = track["velocity"]
            obj.pose.pose.position.x, obj.pose.pose.position.y, obj.pose.pose.position.z = px, py, pz
            obj.pose.covariance = convert.covariance_36(track["position_covariance"])
            obj.twist.twist.linear.x, obj.twist.twist.linear.y, obj.twist.twist.linear.z = vx, vy, vz
            obj.twist.covariance = convert.covariance_36(track["velocity_covariance"])
            array.objects.append(obj)
            ids.add(obj.id)
            markers.markers.extend(self.track_markers(array.header, obj.id, (px, py, pz), (vx, vy, vz)))
        for stale in self.last_marker_ids - ids:
            for ns in ("position", "velocity", "label"):
                marker = Marker(header=array.header, ns=ns, id=stale, action=Marker.DELETE)
                markers.markers.append(marker)
        self.last_marker_ids = ids
        self.tracks_pub.publish(array)
        self.markers_pub.publish(markers)

    @staticmethod
    def track_markers(header, track_id, position, velocity):
        px, py, pz = position
        body = Marker(header=header, ns="position", id=track_id, type=Marker.CYLINDER, action=Marker.ADD)
        body.pose.position.x, body.pose.position.y, body.pose.position.z = px, py, pz
        body.pose.orientation.w = 1.0
        body.scale.x = body.scale.y = 0.5
        body.scale.z = 1.7
        body.color.r, body.color.g, body.color.b, body.color.a = 0.2, 0.6, 1.0, 0.6
        arrow = Marker(header=header, ns="velocity", id=track_id, type=Marker.ARROW, action=Marker.ADD)
        arrow.points = [Point(x=px, y=py, z=pz), Point(x=px + velocity[0], y=py + velocity[1], z=pz + velocity[2])]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.08, 0.16, 0.2
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.5, 0.0, 1.0
        label = Marker(header=header, ns="label", id=track_id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = px, py, pz + 1.1
        label.pose.orientation.w = 1.0
        label.scale.z = 0.3
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = str(track_id)
        return [body, arrow, label]

    def destroy_node(self):
        if self.engine.poll() is None:
            self.engine.stdin.close()
            try:
                self.engine.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.engine.kill()
        os.unlink(self.setup_file.name)
        super().destroy_node()


def main():
    rclpy.init()
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
