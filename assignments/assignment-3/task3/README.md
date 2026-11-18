# Assignment 3 — Task 3: Visual SLAM and Navigation

## 1. Objective

Implement visual simultaneous localization and mapping (SLAM) for a simulated TurtleBot3 using StellaVSLAM and prepare the system for autonomous navigation with ROS 2 Nav2.

The objective is to generate a map from RGB-D camera observations rather than rely on a preloaded map.

**Status: In progress — visual feature detection and SLAM initialization remain unresolved.**

## 2. Development Environment

- Ubuntu running under WSL
- Visual Studio Code
- Docker and Docker Compose
- ROS 2 Jazzy
- Gazebo simulation
- TurtleBot3 Waffle
- StellaVSLAM
- Zenoh communication
- Nav2
- Python and OpenCV

Git branch: `assignment-3-task3`

## 3. Simulation and Communication

The project uses the following Docker Compose services:

| Service | Purpose |
|---|---|
| `zenoh-router` | Zenoh communication router |
| `demo-world` | Gazebo TurtleBot simulation |
| `demo-slam` | StellaVSLAM RGB-D processing |

The Zenoh ROS 2 bridge is started manually:

```bash
docker compose exec demo-world \
  zenoh-bridge-ros2dds client --connect tcp/localhost:7447
```

The bridge must be restarted whenever the `demo-world` container is recreated.

The simulation and Zenoh communication were tested successfully.

## 4. RGB-D Camera Investigation

The TurtleBot camera publishes:

| Parameter | Value |
|---|---|
| RGB resolution | 320 × 240 |
| RGB encoding | `rgb8` |
| Depth encoding | `32FC1` |
| Expected frequency | 5 Hz |
| Measured frequency | 5.001 Hz |

The original camera image appeared nearly uniform gray.

The camera sensor pose in `tb_worlds/urdf/gz_waffle.sdf.xacro` was changed to:

```xml
<pose>0 0 0 0 0 0</pose>
```

This removed an additional sensor offset and improved the camera image and depth measurements.

The original file was backed up as `gz_waffle.sdf.xacro.backup`.

## 5. Visual Feature Detection

OpenCV ORB was used to evaluate visual features in camera images.

The diagnostic detector used:

```python
orb = cv2.ORB_create(nfeatures=1000, fastThreshold=3)
```

### Results

| Camera test | ORB keypoints | Brightness standard deviation |
|---|---:|---:|
| Initial view with ArUco markers enabled | 0 | 15.91 |
| After robot rotation | 5 | 21.10 |

Earlier camera tests also detected only 0–1 keypoints.

The camera images showed large, nearly featureless gray surfaces.

**Finding:** The current camera views contain too few visual features for reliable StellaVSLAM initialization and tracking.

## 6. StellaVSLAM Processing

The `demo-slam` service successfully received and processed RGB-D frames.

Logs showed more than 3,000 frames processed.

Some logs contained the warning:

```text
preprocess: cannot extract any keypoints
```

A previous direct Zenoh subscription to `tb/slam/pose` received no pose messages during the observation interval.

Processing camera frames does not necessarily mean StellaVSLAM has successfully initialized or is tracking the robot.

**Current status:** Camera pose publication and visual map generation have not been demonstrated.

## 7. ArUco Marker Investigation

The simulation includes an ArUco marker spawner:

`tb_worlds/launch/aruco_marker_spawner.launch.py`

Two markers are configured:

| Marker | X (m) | Y (m) | Z