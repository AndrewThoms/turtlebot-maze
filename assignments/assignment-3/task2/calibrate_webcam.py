
from pathlib import Path
import cv2
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent
VIDEO = Path.home() / "slam-tools/datasets/indoor/calibration_video.mp4"
RESULTS = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)

# Number of INNER checkerboard corners
PATTERN_SIZE = (8, 6)

# Square size in millimeters.
# Change this if your printed squares have a different size.
SQUARE_SIZE = 25.8

# 3D coordinates of checkerboard corners
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:PATTERN_SIZE[0],
    0:PATTERN_SIZE[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE

object_points = []
image_points = []

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video frames: {frame_count}")
print(f"Video FPS: {fps:.2f}")

# Sample approximately two frames per second
sample_interval = max(1, round(fps / 2))

frame_index = 0
detected = 0
image_size = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_index % sample_interval != 0:
        frame_index += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        object_points.append(objp.copy())
        image_points.append(corners)
        detected += 1

        # Save a few images showing detected corners
        if detected <= 5:
            preview = frame.copy()
            cv2.drawChessboardCorners(
                preview, PATTERN_SIZE, corners, found
            )
            cv2.imwrite(
                str(RESULTS / f"calibration_corners_{detected}.png"),
                preview
            )

    frame_index += 1

cap.release()

print(f"Checkerboard detections: {detected}")

if detected < 10:
    raise RuntimeError(
        "Too few checkerboard detections. "
        "Try a clearer calibration recording."
    )

# Camera calibration
rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None
)

print("\nCalibration results")
print("-------------------")
print(f"RMS reprojection error: {rms:.4f} pixels")
print("\nCamera matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(dist_coeffs)

np.savez(
    RESULTS / "webcam_calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_size=np.array(image_size),
    rms=rms
)

print("\nSaved: results/webcam_calibration.npz")
