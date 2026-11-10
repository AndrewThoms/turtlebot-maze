
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

times = np.loadtxt(RESULTS_DIR / "track_times.txt")

mean_time = np.mean(times)
median_time = np.median(times)
min_time = np.min(times)
max_time = np.max(times)
std_time = np.std(times)

print("StellaVSLAM - AIST Tracking Performance")
print("-------------------------------------")
print(f"Number of frames: {len(times)}")
print(f"Average tracking time: {mean_time:.4f} seconds")
print(f"Median tracking time: {median_time:.4f} seconds")
print(f"Minimum tracking time: {min_time:.4f} seconds")
print(f"Maximum tracking time: {max_time:.4f} seconds")
print(f"Standard deviation: {std_time:.4f} seconds")
print(f"Estimated tracking throughput: {1 / mean_time:.2f} FPS")
