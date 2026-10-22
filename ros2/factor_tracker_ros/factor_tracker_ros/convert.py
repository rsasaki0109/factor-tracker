"""Conversions between ROS-side data and factor-tracker stream records.

Pure Python (no ROS imports) so it can be tested without a ROS install.
Matrices are plain nested lists or flat row-major lists.
"""

import json
import math

SCHEMA_VERSION = 1
LIDAR_SENSOR_ID = 1

# Engine class ids (see scripts/grandtour/grandtour.py) -> derived_object_msgs
# Object.CLASSIFICATION_* values.
CLASSIFICATION = {1: 6, 2: 4, 3: 5, 4: 6, 5: 7, 6: 4}
CLASSIFICATION_UNKNOWN = 0


def quaternion_to_matrix(x, y, z, w):
    """Rotation matrix (3x3 nested list) of a unit quaternion."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def transform_point(rotation, translation, point):
    """`rotation * point + translation`."""
    return [sum(rotation[i][k] * point[k] for k in range(3)) + translation[i] for i in range(3)]


def rotate_covariance(rotation, covariance):
    """`R C R^T` for a 3x3 covariance."""
    return _matmul(_matmul(rotation, covariance), _transpose(rotation))


def position_covariance(pose_covariance_36, default_std):
    """The 3x3 position block of a ROS 6x6 covariance, or `default_std^2 I`
    when the publisher left it empty."""
    block = [[pose_covariance_36[6 * i + j] for j in range(3)] for i in range(3)]
    if all(block[i][i] > 0.0 for i in range(3)):
        return block
    variance = default_std * default_std
    return [[variance if i == j else 0.0 for j in range(3)] for i in range(3)]


def parse_class_map(entries):
    """`["person:2", "pedestrian:2"]` -> `{"person": 2, "pedestrian": 2}`."""
    out = {}
    for entry in entries:
        name, _, value = entry.rpartition(":")
        if not name:
            raise ValueError(f"class_map entry '{entry}' is not 'name:id'")
        out[name] = int(value)
    return out


def lidar_batch_record(detections, frame_id, sequence, event_time_ns, arrival_time_ns):
    """One `lidar_batch` input line.

    `detections` holds dicts with `position` (3 floats, already in
    `frame_id`), `covariance` (3x3), `class_label` (int or None) and `score`.
    """
    records = []
    for d in detections:
        record = {
            "position": list(d["position"]),
            "covariance": [v for row in d["covariance"] for v in row],
            "score": d["score"],
        }
        if d["class_label"] is not None:
            record["class"] = d["class_label"]
        records.append(record)
    return json.dumps(
        {
            "type": "lidar_batch",
            "schema_version": SCHEMA_VERSION,
            "sensor_id": LIDAR_SENSOR_ID,
            "sequence": sequence,
            "event_time_ns": event_time_ns,
            "arrival_time_ns": arrival_time_ns,
            "frame_id": frame_id,
            "detections": records,
        }
    )


def setup_document(tracking_frame):
    """An engine setup whose LiDAR frame is the tracking frame.

    The node transforms detections into the tracking frame itself (with TF,
    at each detection's stamp), so the engine needs no extrinsics or ego
    poses. The camera entries are placeholders: the node runs LiDAR-only.
    """
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return {
        "tracking_frame": tracking_frame,
        "lidar_frame": tracking_frame,
        "camera_frame": "unused_camera",
        "lidar_to_reference": identity,
        "camera_to_reference": identity,
        "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "skew": 0.0,
                       "image_width": 1, "image_height": 1},
        "near_plane": 0.1,
    }


def tracks_from_output(line, include_tentative=False):
    """Tracks of one `lidar_frame` output line, or None for other records.

    Each track: `id`, `timestamp_ns`, `position`, `velocity`,
    `position_covariance` and `velocity_covariance` (3x3), `classification`,
    `prediction_only`.
    """
    record = json.loads(line)
    if record.get("type") != "lidar_frame":
        return None
    keep = {"confirmed", "tentative"} if include_tentative else {"confirmed"}
    tracks = []
    for track in record["tracks"]:
        if track["lifecycle"] not in keep:
            continue
        c = track["covariance"]
        tracks.append(
            {
                "id": track["track_id"],
                "timestamp_ns": track["state_timestamp_ns"],
                "position": track["position"],
                "velocity": track["velocity"],
                "position_covariance": [[c[6 * i + j] for j in range(3)] for i in range(3)],
                "velocity_covariance": [[c[6 * (i + 3) + j + 3] for j in range(3)] for i in range(3)],
                "classification": CLASSIFICATION.get(track["class_label"], CLASSIFICATION_UNKNOWN),
                "prediction_only": track["prediction_only"],
            }
        )
    return {"timestamp_ns": record["timestamp_ns"], "tracks": tracks}


def covariance_36(block):
    """A ROS 6x6 covariance with `block` (3x3) in the linear part. The
    tracker estimates no orientation, so the angular part gets a large
    variance."""
    out = [0.0] * 36
    for i in range(3):
        for j in range(3):
            out[6 * i + j] = block[i][j]
    for i in range(3, 6):
        out[6 * i + i] = 1e6
    return out
