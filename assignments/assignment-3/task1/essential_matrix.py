
import cv2
import numpy as np
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "kitti_dataset/sequences/00/image_0"
CALIB_FILE = BASE_DIR / "kitti_dataset/sequences/00/calib.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load KITTI camera calibration
def load_camera_matrix(calib_file):
    with open(calib_file, "r") as f:
        for line in f:
            if line.startswith("P0:"):
                values = np.array(
                    [float(x) for x in line.split()[1:]]
                )
                P = values.reshape(3, 4)
                return P[:, :3]

    raise ValueError("P0 camera calibration not found.")


K = load_camera_matrix(CALIB_FILE)

print("Camera intrinsic matrix K:")
print(K)

# Load consecutive images
img1 = cv2.imread(
    str(IMAGE_DIR / "000000.png"),
    cv2.IMREAD_GRAYSCALE
)

img2 = cv2.imread(
    str(IMAGE_DIR / "000001.png"),
    cv2.IMREAD_GRAYSCALE
)

if img1 is None or img2 is None:
    raise FileNotFoundError("KITTI images not found.")

# ORB feature detection
orb = cv2.ORB_create(nfeatures=2000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    raise RuntimeError("ORB descriptors not found.")

# Feature matching
matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

matches = sorted(
    matcher.match(des1, des2),
    key=lambda m: m.distance
)

# Extract matched pixel coordinates
pts1 = np.float64([
    kp1[m.queryIdx].pt for m in matches
])

pts2 = np.float64([
    kp2[m.trainIdx].pt for m in matches
])

if len(pts1) < 8:
    raise RuntimeError("Not enough feature matches.")

# Estimate essential matrix using RANSAC
E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None:
    raise RuntimeError("Essential matrix estimation failed.")

# Recover relative camera rotation and translation
inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    pts1,
    pts2,
    K,
    mask=mask
)

print("\nEssential Matrix E:")
print(E)

print("\nRotation Matrix R:")
print(R)

print("\nTranslation Direction t:")
print(t)

print(f"\nTotal feature matches: {len(matches)}")
print(f"Pose inliers: {inliers}")

# Save results
np.savetxt(RESULTS_DIR / "essential_matrix.txt", E)
np.savetxt(RESULTS_DIR / "rotation_matrix.txt", R)
np.savetxt(RESULTS_DIR / "translation_vector.txt", t)

# Save inlier match visualization
inlier_matches = [
    m for m, keep in zip(matches, pose_mask.ravel())
    if keep != 0
]

inlier_image = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    inlier_matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(RESULTS_DIR / "essential_inliers.png"),
    inlier_image
)

print("\nResults saved in results/")
