"""Starts the tracker node. Remap `detections` to your detector's topic."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("detections", default_value="detections"),
        DeclareLaunchArgument("tracking_frame", default_value="odom"),
        DeclareLaunchArgument("engine_executable", default_value="perception-cli"),
        Node(
            package="perception_engine_ros",
            executable="tracker_node",
            name="perception_engine_tracker",
            parameters=[{
                "tracking_frame": LaunchConfiguration("tracking_frame"),
                "engine_executable": LaunchConfiguration("engine_executable"),
            }],
            remappings=[("detections", LaunchConfiguration("detections"))],
            output="screen",
        ),
    ])
