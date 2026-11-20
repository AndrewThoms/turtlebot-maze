
# Task 1 — Implement a VSLAM pipeline in Python

## Overview

This task implements monocular visual odometry using Python, OpenCV, and the KITTI dataset. The goal is to estimate camera movement from images and compare the estimated trajectory with the actual trajectory. The implementation is based on the Monocular Visual Odometry C++ example (https://github.com/thehummingbird/robotics_demos/tree/main/monocular_vo).

## Implementation

1. ORB Feature Matching: Detects and matches features between consecutive camera images.
2. Essential Matrix and Pose Estimation: Calculates the camera's relative rotation and translation direction.
3. 3D Triangulation: Estimates the locations of 3D landmarks using matched features.
4. Camera Trajectory: Estimates the camera path and compares it with KITTI ground truth.

## Dataset

- KITTI Odometry Dataset 

## Results - saved in "task1/results/"

| Measurement | Result |
|---|---:|
| ORB feature matches | 1,221 |
| Triangulated 3D landmarks | 820 |
| Successful trajectory updates | 199 |
| Trajectory RMSE | 10.117 m |
| Final position error | 16.407 m |

-ORB Feature Matching - shown in "task1/results/orb_matches.png"
-3D Landmark Triangulation - shown in "task1/results/landmarks_3d.png"
-Camera Trajectory Comparison - shown in "task1/results/trajectory_comparison.png"
 The estimated trajectory follows the general shape of the actual path but shows some drift as the camera moves.  The implementation uses KITTI ground-truth movement distances to provide translation scale because monocular visual odometry cannot determine absolute scale from two images alone.

## Python Scripts

-python orb_features.py
-python essential_matrix.py
-python triangulation.py
-python camera_trajectory.py

## References

- KITTI Odometry Dataset - https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- Monocular Visual Odometry Reference Code - https://github.com/thehummingbird/robotics_demos/tree/main/monocular_vo




# Task 2: VSLAM on Your Own Indoor Space

## Overview

This task uses StellaVSLAM to perform VSLAM using the AIST Living Lab dataset and a video of my apartment recorded with a laptop webcam.

## Environment

- Ubuntu 24.04 (WSL)
- Docker
- StellaVSLAM with SocketViewer
- Python, OpenCV, NumPy, and Matplotlib

## Step 1: validate setup 

StellaVSLAM was run on the `aist_living_lab_1` panoramic video using the provided camera configuration and ORB vocabulary.

The system successfully generated a 3D map and estimated the camera trajectory.

### Results - saved in "task2/results/"

| Measurement | Result |
|---|---|
| Camera poses | 1,359 |
| Keyframes | 221 |
| Trajectory duration | 135.80 seconds |
 
-SocketViewer Mapping - shown in "task2/results/assignment 3, task 2 results - aist socketviewer mapping.png"
-Camera Trajectory  - shown in "task2/results/aist_trajectory_3d.png"

## Step 2: Indoor Webcam

A video was recorded indoors using my laptop's camera. The camera was calibrated using OpenCV and an 8 × 6 internal-corner checkerboard. StellaVSLAM successfully initialized, generated a sparse 3D map, and estimated the camera trajectory. 

### Results - saved in "task2/results/"

| Measurement | Result |
|---|---|
| Camera poses | 238 |
| Keyframes | 27 |
| Trajectory duration | 11.85 seconds |
| Calibration RMS error | 0.353 pixels |

### Camera Trajectory  - shown in "task2/results/indoor/indoor_trajectory_3d.png"

### Youtube video documenting Step2  - https://www.youtube.com/watch?v=3lRUYHLyx9Y

## Comparison

Both experiments successfully generated maps and camera trajectories. The indoor webcam experiment produced a shorter trajectory, indicating that tracking was more limited than in the AIST reference experiment.

## Python Scripts

-calibrate_webcam.py
-plot_aist_trajectory.py
-analyze_aist_tracking.py
-plot_indoor_trajectory.py
-analyze_indoor_tracking.py

## References

- StellaVSLAM "https://github.com/stella-cv/stella_vslam"
- SocketViewer "https://github.com/stella-cv/socket_viewer"
- Assignment 3 — Visual SLAM "https://aegean.ai/aiml-common/projects/robotics/visual-slam/visual-slam"

# Task 3: Integrate StellaVSLAM with Nav2

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
