# perception_engine_ros

ROS 2 node for perception-engine: 3D detections in, tracked objects with
position, velocity, covariance and a stable ID out. Tested on ROS 2 Jazzy.

The tracking runs in `perception-cli stream` (started by the node as a child
process); the node only converts messages and applies TF.

| | topic | type |
| --- | --- | --- |
| in | `detections` | `vision_msgs/Detection3DArray` (any frame with TF to `tracking_frame`) |
| out | `tracks` | `derived_object_msgs/ObjectWithCovarianceArray` in `tracking_frame` |
| out | `tracks/markers` | `visualization_msgs/MarkerArray` (body, velocity arrow, ID) |

Each detection is placed at its box centre and transformed into
`tracking_frame` (default `odom`) with TF at the array's stamp, so the robot's
own motion does not show up as people moving. Arrays wait up to `tf_timeout`
for their transform.

## Build and run

```bash
cargo build --release -p perception-cli          # from the repository root
cd ~/ros2_ws/src && ln -s /path/to/perception-engine/ros2/perception_engine_ros .
sudo apt install ros-jazzy-vision-msgs ros-jazzy-derived-object-msgs
cd ~/ros2_ws && colcon build --packages-select perception_engine_ros && source install/setup.bash
ros2 launch perception_engine_ros tracker.launch.py \
    detections:=/your/detector/output \
    engine_executable:=/path/to/perception-engine/target/release/perception-cli
```

The detector is not included: any node publishing `Detection3DArray` works
(for the Mid-360 see `docs/mid360.md`).

## Parameters

| name | default | meaning |
| --- | --- | --- |
| `tracking_frame` | `odom` | frame the tracks are estimated and published in |
| `engine_executable` | `perception-cli` | path to the built CLI |
| `mode` | `lidar_only` | engine mode |
| `engine_args` | pedestrian motion prior, 1 m distance association | extra `perception-cli stream` flags |
| `fgo_lag` | `0` | re-decide associations over the last N scans (fixed-lag, factor-graph cost); tracks are then published N scans late. 0 = immediate output |
| `class_map` | `person:2`, `pedestrian:2`, `Pedestrian:2` | detector `class_id` to engine class |
| `keep_unmapped_classes` | `false` | track detections whose class is not in `class_map` |
| `min_score` | `0.0` | drop detections below this score |
| `default_position_std` | `0.15` | position std (m) when the detection carries no covariance |
| `publish_tentative` | `false` | also publish unconfirmed tracks |
| `tf_timeout` | `0.1` | seconds an array may wait for its transform |

## Tests

```bash
source /opt/ros/jazzy/setup.bash
cd ros2/perception_engine_ros && python3 -m pytest test/
```

`test_node_e2e.py` runs the node against a robot driving at 0.5 m/s past one
standing and one walking person and checks the tracks in `odom` (standing
person at rest, walker at 1 m/s, two stable IDs).
