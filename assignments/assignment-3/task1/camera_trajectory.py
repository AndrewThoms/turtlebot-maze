
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "kitti_dataset/sequences/00/image_0"
CALIB_FILE = BASE_DIR / "kitti_dataset/sequences/00/calib.txt"
POSE_FILE = BASE_DIR / "kitti_dataset/poses/00.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_FRAMES = 200


# Load KITTI camera intrinsics
def load_camera_matrix(path):
    with open(path) as f:
        for line in f:
            if line.startswith("P0:"):
                values = np.array(
                    [float(x) for x in line.split()[1:]]
                )
                return values.reshape(3, 4)[:, :3]
    raise ValueError("P0 calibration not found")


# Load KITTI ground-truth camera poses
def load_ground_truth(path):
    return np.loadtxt(path).reshape(-1, 3, 4)


K = load_camera_matrix(CALIB_FILE)
ground_truth = load_ground_truth(POSE_FILE)

image_paths = sorted(IMAGE_DIR.glob("*.png"))
num_frames = min(MAX_FRAMES, len(image_paths), len(ground_truth))

if num_frames < 2:
    raise RuntimeError("Not enough KITTI frames.")

# ORB detector and matcher
orb = cv2.ORB_create(nfeatures=3000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Camera-to-world pose initialized at identity
R_wc = np.eye(3)
p_w = np.zeros(3)

estimated_positions = [p_w.copy()]

gt_positions = ground_truth[:num_frames, :, 3].copy()
gt_positions -= gt_positions[0].copy()

prev_img = cv2.imread(
    str(image_paths[0]),
    cv2.IMREAD_GRAYSCALE
)

if prev_img is None:
    raise RuntimeError("Unable to load first image.")

# Initialize diagnostic counters BEFORE the loop
successful_updates = 0
skipped_updates = 0

# Process consecutive frames
for i in range(1, num_frames):

    curr_img = cv2.imread(
        str(image_paths[i]),
        cv2.IMREAD_GRAYSCALE
    )

    if curr_img is None:
        raise RuntimeError(f"Unable to load frame {i}")

    kp1, des1 = orb.detectAndCompute(prev_img, None)
    kp2, des2 = orb.detectAndCompute(curr_img, None)

    if des1 is None or des2 is None:
        skipped_updates += 1
        estimated_positions.append(p_w.copy())
        prev_img = curr_img
        continue

    matches = matcher.match(des1, des2)

    if len(matches) < 8:
        skipped_updates += 1
        estimated_positions.append(p_w.copy())
        prev_img = curr_img
        continue

    pts1 = np.float64([
        kp1[m.queryIdx].pt for m in matches
    ])

    pts2 = np.float64([
        kp2[m.trainIdx].pt for m in matches
    ])

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        skipped_updates += 1
        estimated_positions.append(p_w.copy())
        prev_img = curr_img
        continue

    try:
        inliers, R, t, pose_mask = cv2.recoverPose(
            E, pts1, pts2, K, mask=mask
        )
    except cv2.error:
        skipped_updates += 1
        estimated_positions.append(p_w.copy())
        prev_img = curr_img
        continue

    # Ground-truth step length supplies metric scale.
    # This is a scale-assisted VO demonstration.
    scale = np.linalg.norm(
        gt_positions[i] - gt_positions[i - 1]
    )

    # Update pose only when enough inliers are available
    if inliers >= 30 and scale > 0.01:

        R_wc = R_wc @ R.T
        p_w = p_w - scale * (R_wc @ t.reshape(3))

        successful_updates += 1

    else:
        skipped_updates += 1
        print(
            f"Skipped frame {i}: "
            f"inliers={inliers}, scale={scale:.4f}"
        )

    # Save estimated position for EVERY frame
    estimated_positions.append(p_w.copy())

    # Move to next frame
    prev_img = curr_img

    if i % 20 == 0:
        print(f"Processed frame {i}/{num_frames - 1}")


# Convert list to NumPy array
estimated_positions = np.array(estimated_positions)

# Verify trajectory lengths
assert len(estimated_positions) == len(gt_positions)

# Print diagnostic summary
print("\n--- Visual Odometry Diagnostics ---")
print(f"Total frame transitions: {num_frames - 1}")
print(f"Successful pose updates: {successful_updates}")
print(f"Skipped pose updates: {skipped_updates}")

# Save trajectory coordinates
output = np.column_stack(
    (estimated_positions, gt_positions)
)

np.savetxt(
    RESULTS_DIR / "trajectory.csv",
    output,
    delimiter=",",
    header="est_x,est_y,est_z,gt_x,gt_y,gt_z",
    comments=""
)

# Plot X-Z ground-plane trajectory
plt.figure(figsize=(10, 8))

plt.plot(
    gt_positions[:, 0],
    gt_positions[:, 2],
    label="KITTI Ground Truth",
    linewidth=2
)

plt.plot(
    estimated_positions[:, 0],
    estimated_positions[:, 2],
    label="Estimated Visual Odometry",
    linewidth=2
)

plt.xlabel("X position (m)")
plt.ylabel("Z position (m)")
plt.title("KITTI Sequence 00 - Camera Trajectory")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "trajectory_comparison.png",
    dpi=200
)
plt.close()

print("\nSaved results/trajectory.csv")
print("Saved results/trajectory_comparison.png")

# Position error without trajectory alignment
errors = np.linalg.norm(
    estimated_positions - gt_positions,
    axis=1
)

rmse = np.sqrt(np.mean(errors ** 2))

print(f"Trajectory RMSE: {rmse:.3f} meters")
print("\n--- Final Position Comparison ---")

print("Ground truth final position:")
print(gt_positions[-1])

print("Estimated final position:")
print(estimated_positions[-1])

final_error = np.linalg.norm(
    estimated_positions[-1] - gt_positions[-1]
)

print(f"Final position error: {final_error:.3f} meters")