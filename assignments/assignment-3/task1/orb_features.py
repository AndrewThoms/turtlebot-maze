
import cv2
from pathlib import Path

# -------------------------------------------------
# Task 1: ORB Feature Detection and Matching
# KITTI Odometry Dataset - Sequence 00
# -------------------------------------------------

# Use the script's directory so paths work in VS Code
BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "kitti_dataset" / "sequences" / "00" / "image_0"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Load two consecutive grayscale images
img1 = cv2.imread(str(IMAGE_DIR / "000000.png"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(str(IMAGE_DIR / "000001.png"), cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("Unable to load KITTI images.")

print("KITTI images loaded successfully.")
print(f"Image dimensions: {img1.shape}")

# Step 2: Initialize ORB detector
orb = cv2.ORB_create(nfeatures=2000)

# Step 3: Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print(f"Frame 0 keypoints: {len(kp1)}")
print(f"Frame 1 keypoints: {len(kp2)}")

if des1 is None or des2 is None:
    raise RuntimeError("ORB descriptors could not be computed.")

# Step 4: Match ORB descriptors
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)

# Step 5: Sort matches by descriptor distance
matches = sorted(matches, key=lambda match: match.distance)

print(f"Total feature matches: {len(matches)}")

# Step 6: Draw the best 100 matches
matched_image = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Step 7: Save the results
output_path = RESULTS_DIR / "orb_matches.png"

if not cv2.imwrite(str(output_path), matched_image):
    raise RuntimeError("Failed to save matched image.")

print(f"Feature matching image saved to: {output_path}")
