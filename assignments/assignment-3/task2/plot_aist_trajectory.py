
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

trajectory_file = RESULTS_DIR / "frame_trajectory.txt"
trajectory = np.loadtxt(trajectory_file)

timestamps = trajectory[:, 0]
x = trajectory[:, 1]
y = trajectory[:, 2]
z = trajectory[:, 3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot(x, y, z, color="blue", linewidth=1.5,
        label="Estimated camera trajectory")

ax.scatter(x[0], y[0], z[0], color="green",
           s=70, label="Start")

ax.scatter(x[-1], y[-1], z[-1], color="red",
           s=70, label="End")

ax.set_xlabel("X (arbitrary scale)")
ax.set_ylabel("Y (arbitrary scale)")
ax.set_zlabel("Z (arbitrary scale)")
ax.set_title("StellaVSLAM - AIST Living Lab Camera Trajectory")
ax.legend()

plt.tight_layout()

output_file = RESULTS_DIR / "aist_trajectory_3d.png"
plt.savefig(output_file, dpi=200)

print(f"Number of camera poses: {len(trajectory)}")
print(f"Number of columns: {trajectory.shape[1]}")
print(f"Trajectory duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
print(f"Start position: {x[0]:.4f}, {y[0]:.4f}, {z[0]:.4f}")
print(f"End position: {x[-1]:.4f}, {y[-1]:.4f}, {z[-1]:.4f}")
print(f"Saved plot: {output_file}")

plt.show()
