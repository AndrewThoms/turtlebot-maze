import math
import numpy as np


FX = 277.13
FY = 277.13
CX = 160.0
CY = 120.0


def back_project(u, v, depth):
    """Back-project an image pixel into the camera optical frame."""
    x_o = (u - CX) * depth / FX
    y_o = (v - CY) * depth / FY
    z_o = depth

    return np.array([x_o, y_o, z_o, 1.0])


def optical_to_body(point_optical):
    """Convert optical axes (right, down, forward) to ROS body axes."""
    R_co = np.array([
        [0.0,  0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 1.0],
    ])

    return R_co @ point_optical


def camera_to_base(point_body):
    """Apply the fixed base_link -> camera translation from the URDF."""
    T_bc = np.eye(4)
    T_bc[0, 3] = 0.064
    T_bc[1, 3] = -0.065
    T_bc[2, 3] = 0.094

    return T_bc @ point_body


def base_to_map(point_base, pose):
    """Transform a base_link point using the synchronized robot pose."""
    x, y, yaw = pose

    c = math.cos(yaw)
    s = math.sin(yaw)

    T_mb = np.array([
        [c, -s, 0.0, x],
        [s,  c, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    return T_mb @ point_base


def pixel_to_map(u, v, depth, pose):
    """Apply the complete optical -> camera -> base -> map chain."""
    p_optical = back_project(u, v, depth)
    p_body = optical_to_body(p_optical)
    p_base = camera_to_base(p_body)
    p_map = base_to_map(p_base, pose)

    return p_map[:3]
