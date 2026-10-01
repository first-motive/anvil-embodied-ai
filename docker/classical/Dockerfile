# Classical can-on-paper baseline. Runs on the robot PC itself: no GPU, no
# lerobot, just ROS2 Jazzy, OpenCV and the loader's message types.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rosbag2-py \
    ros-jazzy-rosbag2-storage-mcap \
    ros-jazzy-tf2-ros-py \
    ros-jazzy-std-srvs \
    python3-opencv \
    python3-scipy \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# Prebuilt anvil_msgs install tree, extracted from the loader image by
# scripts/vendor_anvil_msgs.sh. It is proprietary and not published anywhere,
# so it is vendored at build time rather than committed. Same distro, Python and
# architecture as the loader, so the type hashes match.
COPY vendor/anvil_msgs /opt/anvil_msgs
ENV AMENT_PREFIX_PATH=/opt/anvil_msgs:${AMENT_PREFIX_PATH} \
    PYTHONPATH=/opt/anvil_msgs/lib/python3.12/site-packages:${PYTHONPATH} \
    LD_LIBRARY_PATH=/opt/anvil_msgs/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace

COPY ros2/src/classical_control_msgs /workspace/src/classical_control_msgs
COPY ros2/src/classical_control /workspace/src/classical_control
COPY configs/cyclonedds /workspace/configs/cyclonedds

RUN . /opt/ros/jazzy/setup.sh && colcon build

# The inference entrypoint already sources the workspace and wires CycloneDDS
# from the environment; nothing here differs.
COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "classical_control", "classical.launch.py"]
