
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
