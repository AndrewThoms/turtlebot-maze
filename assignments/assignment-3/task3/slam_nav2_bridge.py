
#!/usr/bin/env python3
"""Publish StellaVSLAM camera poses from Zenoh into ROS 2.

Visualization-only bridge. Does not publish TF or command the robot.
"""

import json
import threading

import numpy as np
import rclpy
import zenoh

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def rotation_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to a quaternion (x, y, z, w)."""
    r = rotation
    trace = np.trace(r)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    return q


class SlamPoseBridge(Node):
    def __init__(self):
        super().__init__("slam_pose_bridge")

        self.publisher = self.create_publisher(
            PoseStamped, "/stella_vslam/camera_pose_raw", 10
        )

        config = zenoh.Config()
        config.insert_json5(
            "connect/endpoints", json.dumps(["tcp/localhost:7447"])
        )
        self.session = zenoh.open(config)

        self.subscriber = self.session.declare_subscriber(
            "tb/slam/pose", self.on_pose
        )

        self.get_logger().info(
            "Listening on Zenoh tb/slam/pose; publishing "
            "/stella_vslam/camera_pose_raw"
        )

    def on_pose(self, sample):
        try:
            record = json.loads(bytes(sample.payload).decode("utf-8"))
            values = np.asarray(record["pose"], dtype=float)

            if values.shape != (12,) or not np.isfinite(values).all():
                return

            matrix = np.eye(4)
            matrix[:3, :] = values.reshape(3, 4)

            # Publish the raw matrix as received, without inversion
            # or optical-frame conversion. Coordinate convention is
            # still to be verified before navigation integration.
            position = matrix[:3, 3]
            quaternion = rotation_to_quaternion(matrix[:3, :3])

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "stella_raw"

            msg.pose.position.x = float(position[0])
            msg.pose.position.y = float(position[1])
            msg.pose.position.z = float(position[2])

            msg.pose.orientation.x = float(quaternion[0])
            msg.pose.orientation.y = float(quaternion[1])
            msg.pose.orientation.z = float(quaternion[2])
            msg.pose.orientation.w = float(quaternion[3])

            self.publisher.publish(msg)

        except Exception as exc:
            self.get_logger().warning(f"Invalid SLAM pose: {exc}")

    def close(self):
        self.subscriber.undeclare()
        self.session.close()


def main():
    rclpy.init()
    node = None

    try:
        node = SlamPoseBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
