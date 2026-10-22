from glob import glob

from setuptools import setup

package_name = "factor_tracker_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rsasaki0109",
    maintainer_email="rsasaki0109@gmail.com",
    description="ROS 2 wrapper for factor-tracker.",
    license="MIT OR Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tracker_node = factor_tracker_ros.tracker_node:main",
        ],
    },
)
