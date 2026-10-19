import json
import math

from perception_engine_ros import convert


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_quaternion_yaw_90_rotates_x_to_y():
    s = math.sqrt(0.5)
    rotation = convert.quaternion_to_matrix(0.0, 0.0, s, s)
    assert close(convert.transform_point(rotation, [1.0, 2.0, 0.0], [1.0, 0.0, 0.0]), [1.0, 3.0, 0.0])


def test_rotate_covariance_swaps_axes():
    s = math.sqrt(0.5)
    rotation = convert.quaternion_to_matrix(0.0, 0.0, s, s)
    out = convert.rotate_covariance(rotation, [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 9.0]])
    assert close([out[0][0], out[1][1], out[2][2]], [1.0, 4.0, 9.0])


def test_position_covariance_falls_back_when_empty():
    assert convert.position_covariance([0.0] * 36, 0.2)[1][1] == 0.2 * 0.2
    full = [0.0] * 36
    full[0], full[7], full[14] = 1.0, 2.0, 3.0
    assert convert.position_covariance(full, 0.2)[2][2] == 3.0


def test_class_map():
    assert convert.parse_class_map(["person:2", "a:b:3"]) == {"person": 2, "a:b": 3}


def test_lidar_batch_record_uses_engine_field_names():
    line = convert.lidar_batch_record(
        [{"position": [1, 2, 3], "covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "class_label": 2, "score": 0.9},
         {"position": [4, 5, 6], "covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "class_label": None, "score": 0.5}],
        "odom", 7, 100, 120)
    record = json.loads(line)
    assert record["type"] == "lidar_batch" and record["frame_id"] == "odom"
    assert record["detections"][0]["class"] == 2
    assert "class" not in record["detections"][1]
    assert len(record["detections"][0]["covariance"]) == 9


def test_tracks_from_output_filters_and_splits_covariance():
    covariance = [float(i) for i in range(36)]
    track = {"track_id": 3, "lifecycle": "confirmed", "state_timestamp_ns": 5, "position": [1, 2, 3],
             "velocity": [0.5, 0, 0], "covariance": covariance, "class_label": 2, "prediction_only": False}
    line = json.dumps({"type": "lidar_frame", "timestamp_ns": 5,
                       "tracks": [track, dict(track, track_id=4, lifecycle="tentative")]})
    frame = convert.tracks_from_output(line)
    assert [t["id"] for t in frame["tracks"]] == [3]
    out = frame["tracks"][0]
    assert out["classification"] == 4
    assert out["position_covariance"][0] == [0.0, 1.0, 2.0]
    assert out["velocity_covariance"][0] == [21.0, 22.0, 23.0]
    assert len(convert.tracks_from_output(line, include_tentative=True)["tracks"]) == 2
    assert convert.tracks_from_output(json.dumps({"type": "camera_frame"})) is None
