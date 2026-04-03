# AUV Vision Pipeline using ROS 2(Humble)

## Installation & Setup

### Prerequisites

- Standard Docker installed and running.
- A built-in or USB webcam mapped to `/dev/video0`.
- **GUI Access:** If you are on Linux (Fedora/Ubuntu), you must allow Docker to access your screen by running this on your **host machine** terminal:
  ```bash
  xhost +local:docker
  ```

### 1. Build the Laptop Image

Open a terminal in the project root and compile the environment:

```bash
sudo docker build -f Dockerfile -t auv_vision_laptop .
```

### 2. Run the Pipeline

Launch the container. We include specific flags to pass your laptop's **Display** and **Camera** into the isolated environment:

```bash
sudo docker run -it --rm \
  --network=host \
  --device /dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e QT_X11_NO_MITSHM=1 \
  auv_vision_laptop
```

## Viewing the Live Video Feed

Open a new terminal window and enter the docker container by using this command.

```bash
# Enter the running container
sudo docker exec -it $(sudo docker ps -q -l) /bin/bash
```

after the prompt changes, enter this command to view the image.

```bash
ros2 run rqt_image_view rqt_image_view
```

---

## Verifying the System

```bash
# Enter the running container
sudo docker exec -it $(sudo docker ps -q -l) /bin/bash

# Check the frequency
ros2 topic hz /auv/camera/raw
```

---

## Changing Exposure (Hardware Tuning)

```bash
# 1. Disable auto-exposure
v4l2-ctl -d /dev/video0 -c exposure_auto=1

# 2. Set manual exposure time (tune 100-500 based on lighting)
v4l2-ctl -d /dev/video0 -c exposure_absolute=300
```
