FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
    v4l-utils \
    ros-humble-cv-bridge \
    ros-humble-rqt-image-view \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/auv_ws

COPY src/ src/

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/auv_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /workspace/auv_ws/install/setup.bash && ros2 launch auv_vision vision.launch.py"]