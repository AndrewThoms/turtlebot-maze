
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "kitti_dataset/sequences/00/image_0"
CALIB_FILE = BASE_DIR / "kitti_dataset/sequences/00/calib.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load camera calibration matrix
def load_camera_matrix(path):
    with open(path) as f:
        for line in f:
            if line.startswith("P0:"):
                values = np.array(
                    [float(v) for v in line.split()[1:]]
                )
                return values.reshape(3, 4)[:, :3]
    raise ValueError("P0 calibration not found")


K = load_camera_matrix(CALIB_FILE)

# Load two consecutive KITTI images
img1 = cv2.imread(
    str(IMAGE_DIR / "000000.png"),
    cv2.IMREAD_GRAYSCALE
)
img2 = cv2.imread(
    str(IMAGE_DIR / "000001.png"),
    cv2.IMREAD_GRAYSCALE
)

if img1 is None or img2 is None:
    raise FileNotFoundError("KITTI images not found")

# Detect ORB features
orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    raise RuntimeError("ORB descriptors missing")

# Match features
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(
    matcher.match(des1, des2),
    key=lambda m: m.distance
)

pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

# Estimate essential matrix
E, mask = cv2.findEssentialMat(
    pts1, pts2, K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None:
    raise RuntimeError("Essential matrix estimation failed")

# Recover camera pose
num_inliers, R, t, pose_mask = cv2.recoverPose(
    E, pts1, pts2, K, mask=mask
)

# Select geometrically consistent matches
valid = pose_mask.ravel() != 0
pts1_inliers = pts1[valid]
pts2_inliers = pts2[valid]

print(f"Pose inliers: {num_inliers}")

# Construct projection matrices
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, t))

# Triangulate 3D landmarks
points_4d = cv2.triangulatePoints(
    P1,
    P2,
    pts1_inliers.T,
    pts2_inliers.T
)

# Convert homogeneous coordinates to Euclidean coordinates
w = points_4d[3]
valid_w = np.abs(w) > 1e-10

points_3d = (
    points_4d[:3, valid_w] / w[valid_w]
).T

# Remove non-finite points
points_3d = points_3d[
    np.all(np.isfinite(points_3d), axis=1)
]

print(f"Triangulated landmarks: {len(points_3d)}")

# Save coordinates
np.savetxt(
    RESULTS_DIR / "landmarks_3d.csv",
    points_3d,
    delimiter=",",
    header="X,Y,Z",
    comments=""
)

# Plot landmarks in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    points_3d[:, 0],
    points_3d[:, 1],
    points_3d[:, 2],
    s=3,
    alpha=0.6
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("KITTI Sequence 00 - Triangulated 3D Landmarks")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "landmarks_3d.png", dpi=200)
plt.close()

print("Saved results/landmarks_3d.csv")
print("Saved results/landmarks_3d.png")
