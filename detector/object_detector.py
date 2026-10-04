#!/usr/bin/env python3
"""
Zenoh-based YOLOv8 object detector with CLIP embedding extraction.

Task 1:
- YOLOv8n COCO detections at confidence >= 0.3
- odometry-driven keyframes
- optional OpenCLIP ViT-B-32 embeddings
- publishes JSON detections over Zenoh

Task 2:
- buffers color, depth, and odometry by ROS timestamp
- matches nearest depth/odometry samples within 200 ms
- estimates robust object depth from the central 50% of each bounding box
- projects detections into map coordinates using task2_geometry.pixel_to_map
"""

import argparse
import base64
import json
import math
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import cv2
import numpy as np
import torch
import zenoh
from PIL import Image as PILImage
from pycdr2 import IdlStruct
from pycdr2.types import float64, int32, uint8, uint32
from ultralytics import YOLO

from task2_geometry import pixel_to_map


@dataclass
class Time(IdlStruct, typename="builtin_interfaces/msg/Time"):
    sec: int32
    nanosec: uint32


@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    stamp: Time
    frame_id: str


@dataclass
class Image(IdlStruct, typename="sensor_msgs/msg/Image"):
    header: Header
    height: uint32
    width: uint32
    encoding: str
    is_bigendian: uint8
    step: uint32
    data: List[uint8]


@dataclass
class Point(IdlStruct, typename="geometry_msgs/msg/Point"):
    x: float64
    y: float64
    z: float64


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs/msg/Quaternion"):
    x: float64
    y: float64
    z: float64
    w: float64


@dataclass
class Pose(IdlStruct, typename="geometry_msgs/msg/Pose"):
    position: Point
    orientation: Quaternion


@dataclass
class PoseWithCovariance(IdlStruct, typename="geometry_msgs/msg/PoseWithCovariance"):
    pose: Pose
    covariance: List[float64]


@dataclass
class Twist(IdlStruct, typename="geometry_msgs/msg/Twist"):
    linear: Point
    angular: Point


@dataclass
class TwistWithCovariance(IdlStruct, typename="geometry_msgs/msg/TwistWithCovariance"):
    twist: Twist
    covariance: List[float64]


@dataclass
class Odometry(IdlStruct, typename="nav_msgs/msg/Odometry"):
    header: Header
    child_frame_id: str
    pose: PoseWithCovariance
    twist: TwistWithCovariance


def quaternion_to_yaw(q: Quaternion) -> float:
    """Extract planar yaw from a quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)
def camera_pose_in_map(robot_pose):
    """Return camera position and orientation in the map frame."""
    x, y, yaw = robot_pose

    # Camera translation relative to the robot base.
    tx = 0.064
    ty = -0.065
    tz = 0.094

    c = math.cos(yaw)
    s = math.sin(yaw)

    # Rotate the fixed camera offset by the robot yaw.
    camera_x = x + c * tx - s * ty
    camera_y = y + s * tx + c * ty
    camera_z = tz

    # Camera link has no additional rotation relative to the base.
    # Convert the robot yaw to a quaternion.
    camera_qx = 0.0
    camera_qy = 0.0
    camera_qz = math.sin(yaw / 2.0)
    camera_qw = math.cos(yaw / 2.0)

    return (
        camera_x,
        camera_y,
        camera_z,
        camera_qx,
        camera_qy,
        camera_qz,
        camera_qw,
    )

def main():
    parser = argparse.ArgumentParser(description="Zenoh YOLOv8 Object Detector")
    parser.add_argument("-e", "--connect", type=str, default="")
    parser.add_argument("-m", "--model", type=str, default="yolov8n.pt")
    parser.add_argument("-c", "--confidence", type=float, default=0.3)
    parser.add_argument(
        "--image-key",
        type=str,
        default="camera/color/image_raw",
        help="Zenoh key for color images",
    )
    parser.add_argument(
        "--detection-key",
        type=str,
        default="tb/detections",
        help="Zenoh key to publish detections to",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=10.0,
        help="Maximum accepted color-image rate",
    )
    parser.add_argument(
        "--enable-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b79k",
    )
    parser.add_argument("--odom-key", type=str, default="odom")
    parser.add_argument(
        "--depth-key",
        type=str,
        default="camera/depth/image_rect_raw",
    )
    parser.add_argument("--keyframe-dist", type=float, default=0.5)
    parser.add_argument("--keyframe-angle", type=float, default=15.0)
    parser.add_argument("--run-id", type=str, default="")
    args = parser.parse_args()

    # D435i intrinsics for the assignment's 320x240 stream.
    FX = 277.13
    FY = 277.13
    CX = 160.0
    CY = 120.0

    run_id = (
        args.run_id
        if args.run_id
        else f"{socket.gethostname()}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    print(f"Run ID: {run_id}")

    model = YOLO(args.model)
    print(f"Loaded model: {args.model}")
    print(f"Subscribing to color: {args.image_key}")
    print(f"Subscribing to depth: {args.depth_key}")
    print(f"Subscribing to odom:  {args.odom_key}")
    print(f"Publishing to:        {args.detection_key}")

    clip_model = None
    clip_preprocess = None
    clip_device = None
    clip_tag = None

    if args.enable_embeddings:
        import open_clip

        clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            args.clip_model,
            pretrained=args.clip_pretrained,
        )
        clip_model = clip_model.to(clip_device).eval()

        clip_embedding_dim = clip_model.visual.output_dim
        if clip_embedding_dim != 512:
            raise ValueError(
                f"Task 1 requires 512-D CLIP embeddings, got {clip_embedding_dim}"
            )

        clip_tag = f"{args.clip_model}/{args.clip_pretrained}"
        print(
            f"CLIP embeddings enabled: {clip_tag} "
            f"({clip_embedding_dim}-dim) on {clip_device}"
        )

    min_interval = 1.0 / args.max_fps
    last_color_accept_time = 0.0

    keyframe_dist_thresh = args.keyframe_dist
    keyframe_angle_thresh = math.radians(args.keyframe_angle)
    last_keyframe_pose = None
    keyframe_id = 0

    # Recent timestamped sensor data.
    depth_buffer = []       # [(timestamp_ns, depth_array), ...]
    odom_buffer = []        # [(timestamp_ns, (x, y, yaw)), ...]
    color_buffer = []       # [(timestamp_ns, Image), ...]
    max_buffer_size = 100
    sync_limit_ns = 200_000_000

    print(
        f"Keyframe gating: dist={keyframe_dist_thresh}m, "
        f"angle={args.keyframe_angle}deg"
    )

    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps([args.connect]))

    session = zenoh.open(conf)
    pub = session.declare_publisher(args.detection_key)

    def ros_time_ns(stamp: Time) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def find_nearest(buffer, timestamp_ns):
        """Return nearest buffered sample only when it is within 200 ms."""
        if not buffer:
            return None

        nearest = min(buffer, key=lambda item: abs(item[0] - timestamp_ns))
        if abs(nearest[0] - timestamp_ns) > sync_limit_ns:
            return None
        return nearest

    def is_keyframe(pose) -> bool:
        """Accept first synchronized frame, then gate on motion."""
        nonlocal last_keyframe_pose, keyframe_id

        if pose is None:
            return False

        if last_keyframe_pose is None:
            last_keyframe_pose = pose
            keyframe_id += 1
            return True

        dx = pose[0] - last_keyframe_pose[0]
        dy = pose[1] - last_keyframe_pose[1]
        dist = math.hypot(dx, dy)

        dyaw_raw = pose[2] - last_keyframe_pose[2]
        dyaw = abs(math.atan2(math.sin(dyaw_raw), math.cos(dyaw_raw)))

        if dist >= keyframe_dist_thresh or dyaw >= keyframe_angle_thresh:
            last_keyframe_pose = pose
            keyframe_id += 1
            return True

        return False

    def image_to_bgr(img_msg):
        """Decode rgb8/bgr8 ROS Image data, respecting row stride."""
        if img_msg.encoding not in ("rgb8", "bgr8"):
            print(f"Unsupported color encoding: {img_msg.encoding}")
            return None

        row_bytes = int(img_msg.width) * 3
        raw = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)

        if int(img_msg.step) < row_bytes:
            print(
                f"Invalid color image step={img_msg.step}; "
                f"expected at least {row_bytes}"
            )
            return None

        try:
            rows = raw.reshape(int(img_msg.height), int(img_msg.step))
        except ValueError:
            print("Color image data size does not match height/step")
            return None

        frame = rows[:, :row_bytes].reshape(
            int(img_msg.height),
            int(img_msg.width),
            3,
        )

        if img_msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def depth_image_to_numpy(depth_msg):
        """Decode Gazebo 32FC1 depth in metres, respecting row stride."""
        if depth_msg.encoding != "32FC1":
            print(f"Unsupported depth encoding: {depth_msg.encoding}")
            return None

        if int(depth_msg.is_bigendian) != 0:
            print("Unsupported big-endian depth image")
            return None

        row_bytes = int(depth_msg.width) * 4
        if int(depth_msg.step) < row_bytes or int(depth_msg.step) % 4 != 0:
            print(
                f"Invalid depth image step={depth_msg.step}; "
                f"expected at least {row_bytes} and divisible by 4"
            )
            return None

        raw = np.frombuffer(bytes(depth_msg.data), dtype="<f4")
        floats_per_row = int(depth_msg.step) // 4

        try:
            rows = raw.reshape(int(depth_msg.height), floats_per_row)
        except ValueError:
            print("Depth image data size does not match height/step")
            return None

        return rows[:, : int(depth_msg.width)]

    def detection_to_3d(bbox, depth_image):
        """
        Estimate optical-frame object position.

        Uses the central 50% of the bounding box and requires at least
        10 finite, positive depth samples. The median is used for depth.
        """
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        width = x2 - x1
        height = y2 - y1

        rx1 = int(x1 + 0.25 * width)
        rx2 = int(x2 - 0.25 * width)
        ry1 = int(y1 + 0.25 * height)
        ry2 = int(y2 - 0.25 * height)

        rx1 = max(0, min(rx1, depth_image.shape[1]))
        rx2 = max(0, min(rx2, depth_image.shape[1]))
        ry1 = max(0, min(ry1, depth_image.shape[0]))
        ry2 = max(0, min(ry2, depth_image.shape[0]))

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        region = depth_image[ry1:ry2, rx1:rx2]
        valid_depth = region[np.isfinite(region) & (region > 0.0)]

        if valid_depth.size < 10:
            return None

        d = float(np.median(valid_depth))
        x_o = (u - CX) * d / FX
        y_o = (v - CY) * d / FY
        z_o = d
        return (x_o, y_o, z_o)

    def process_color_frame(img_msg):
        """Synchronize one buffered color frame and process it if keyframed."""
        image_timestamp_ns = ros_time_ns(img_msg.header.stamp)

        depth_match = find_nearest(depth_buffer, image_timestamp_ns)
        odom_match = find_nearest(odom_buffer, image_timestamp_ns)

        if depth_match is None or odom_match is None:
            print("Skipping color frame: no depth/odom match within 200 ms")
            return

        depth_timestamp_ns, matched_depth = depth_match
        odom_timestamp_ns, matched_pose = odom_match

        depth_diff_ms = abs(depth_timestamp_ns - image_timestamp_ns) / 1_000_000.0
        odom_diff_ms = abs(odom_timestamp_ns - image_timestamp_ns) / 1_000_000.0

        if not is_keyframe(matched_pose):
            return

        frame = image_to_bgr(img_msg)
        if frame is None:
            return

        results = model.predict(frame, conf=args.confidence, verbose=False)

        detections = []
        crops = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                bbox = [round(float(x), 1) for x in box.xyxy[0].tolist()]

                point_3d = detection_to_3d(bbox, matched_depth)
                if point_3d is None:
                    continue

                det = {
                    "class": model.names[cls_id],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": bbox,
                }

                x1, y1, x2, y2 = bbox
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                depth = point_3d[2]

                # task2_geometry implements the homogeneous
                # T_mb * T_bc * R_co * p_optical chain.
                point_map = pixel_to_map(
                    u,
                    v,
                    depth,
                    matched_pose,
                )

                det["camera_x"] = round(point_3d[0], 4)
                det["camera_y"] = round(point_3d[1], 4)
                det["camera_z"] = round(point_3d[2], 4)
                det["map_x"] = round(float(point_map[0]), 4)
                det["map_y"] = round(float(point_map[1]), 4)
                det["map_z"] = round(float(point_map[2]), 4)

                # Horizontal range and bearing relative to the robot/camera heading.
                #
                # bearing_rad = 0 straight ahead
                # positive bearing = counter-clockwise / left
                #
                # range_m is horizontal XY distance, not 3D Euclidean distance.
                forward = point_3d[2]   # optical z = forward
                left = -point_3d[0]     # optical x = right, therefore -x = left

                det["bearing_rad"] = round(math.atan2(left, forward), 4)
                det["range_m"] = round(math.hypot(forward, left), 4)

                detections.append(det)

                # Keep crops aligned one-for-one with accepted detections.
                if clip_model is not None:
                    crop_x1 = max(0, int(bbox[0]))
                    crop_y1 = max(0, int(bbox[1]))
                    crop_x2 = min(frame.shape[1], int(bbox[2]))
                    crop_y2 = min(frame.shape[0], int(bbox[3]))

                    crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop_bgr.size > 0:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        crops.append(PILImage.fromarray(crop_rgb))
                    else:
                        crops.append(None)

        # Process all valid crops in one CLIP batch.
        if clip_model is not None and crops:
            valid_indices = [i for i, crop in enumerate(crops) if crop is not None]

            if valid_indices:
                batch = torch.stack(
                    [clip_preprocess(crops[i]) for i in valid_indices]
                ).to(clip_device)

                with torch.no_grad():
                    embeddings = clip_model.encode_image(batch)

                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings_np = embeddings.cpu().numpy()

                for emb_idx, det_idx in enumerate(valid_indices):
                    vec = embeddings_np[emb_idx]
                    detections[det_idx]["embedding"] = base64.b64encode(
                        vec.astype(np.float32).tobytes()
                    ).decode("ascii")
                    detections[det_idx]["embedding_dim"] = int(vec.shape[0])
                    detections[det_idx]["embedding_model"] = clip_tag

        camera_pose = camera_pose_in_map(matched_pose)

        pose_data = {
            # In this simulation map->odom is identity, so this synchronized
            # odometry pose is also the robot pose in the map frame.
            "map_x": round(matched_pose[0], 4),
            "map_y": round(matched_pose[1], 4),
            "map_yaw": round(matched_pose[2], 4),

            # Camera pose expressed in the map frame.
            "camera_map_x": round(camera_pose[0], 4),
            "camera_map_y": round(camera_pose[1], 4),
            "camera_map_z": round(camera_pose[2], 4),
            "camera_qx": round(camera_pose[3], 6),
            "camera_qy": round(camera_pose[4], 6),
            "camera_qz": round(camera_pose[5], 6),
            "camera_qw": round(camera_pose[6], 6),
        }

        for idx, det in enumerate(detections):
            det["det_id"] = f"{run_id}_kf{keyframe_id}_d{idx}"

        envelope = {
            "run_id": run_id,
            "keyframe_id": keyframe_id,
            "timestamp_ns": image_timestamp_ns,
            **pose_data,
            "detections": detections,
        }

        pub.put(json.dumps(envelope).encode())

        if detections:
            classes = [d["class"] for d in detections]
            print(
                f"KF#{keyframe_id} Detected: {classes} "
                f"@ ({pose_data['map_x']}, {pose_data['map_y']}) "
                f"[sync depth={depth_diff_ms:.1f}ms, odom={odom_diff_ms:.1f}ms]"
            )

    def process_ready_color_frames():
        """
        Process color frames only after depth AND odometry have reached their
        timestamps. This avoids treating callback arrival order as synchronization.
        """
        if not color_buffer or not depth_buffer or not odom_buffer:
            return
        #print(
         #   f"BUFFER DEBUG: color={len(color_buffer)}, "
         #   f"depth={len(depth_buffer)}, odom={len(odom_buffer)}, "
         #   f"color_ts={color_buffer[0][0]}, "
         #   f"depth_ts={depth_buffer[-1][0]}, "
         #   f"odom_ts={odom_buffer[-1][0]}"
        #)
        newest_depth_ts = depth_buffer[-1][0]
        newest_odom_ts = odom_buffer[-1][0]

        ready_through = min(newest_depth_ts, newest_odom_ts)
        ready = []
        waiting = []

        for item in color_buffer:
            if item[0] <= ready_through:
                ready.append(item)
            else:
                waiting.append(item)

        color_buffer[:] = waiting[-max_buffer_size:]

        for _, img_msg in ready:
            process_color_frame(img_msg)

    def odom_callback(sample):
        try:
            odom_msg = Odometry.deserialize(bytes(sample.payload))
        except Exception as exc:
            print(f"Odom CDR deserialize error: {exc}")
            return

        if not hasattr(odom_callback, "frame_printed"):
            print(
                f"Odom frame: '{odom_msg.header.frame_id}', "
                f"child frame: '{odom_msg.child_frame_id}'"
            )
            odom_callback.frame_printed = True

        p = odom_msg.pose.pose.position
        yaw = quaternion_to_yaw(odom_msg.pose.pose.orientation)
        pose = (p.x, p.y, yaw)
        #print(
        #    f"ODOM DEBUG: x={p.x:.3f}, y={p.y:.3f}, "
        #    f"yaw={math.degrees(yaw):.1f} deg"
        #)
        timestamp_ns = ros_time_ns(odom_msg.header.stamp)

        odom_buffer.append((timestamp_ns, pose))
        if len(odom_buffer) > max_buffer_size:
            odom_buffer.pop(0)

        process_ready_color_frames()

    def depth_callback(sample):
        try:
            depth_msg = Image.deserialize(bytes(sample.payload))
        except Exception as exc:
            print(f"Depth CDR deserialize error: {exc}")
            return

        if not hasattr(depth_callback, "first_printed"):
            print(
                f"Depth received: encoding='{depth_msg.encoding}', "
                f"size={depth_msg.width}x{depth_msg.height}, "
                f"frame='{depth_msg.header.frame_id}'"
            )
            depth_callback.first_printed = True

        depth = depth_image_to_numpy(depth_msg)
        if depth is None:
            return

        timestamp_ns = ros_time_ns(depth_msg.header.stamp)
        depth_buffer.append((timestamp_ns, depth))

        if len(depth_buffer) > max_buffer_size:
            depth_buffer.pop(0)

        process_ready_color_frames()

    def image_callback(sample):
        nonlocal last_color_accept_time

        # Limit accepted color input to at most args.max_fps.
        now = time.monotonic()
        if now - last_color_accept_time < min_interval:
            return
        last_color_accept_time = now

        try:
            img_msg = Image.deserialize(bytes(sample.payload))
        except Exception as exc:
            print(f"Color CDR deserialize error: {exc}")
            return

        image_timestamp_ns = ros_time_ns(img_msg.header.stamp)
        color_buffer.append((image_timestamp_ns, img_msg))

        if len(color_buffer) > max_buffer_size:
            # An extremely old unprocessed frame is no longer useful.
            color_buffer.pop(0)

        process_ready_color_frames()

    odom_sub = session.declare_subscriber(args.odom_key, odom_callback)
    depth_sub = session.declare_subscriber(args.depth_key, depth_callback)
    img_sub = session.declare_subscriber(args.image_key, image_callback)

    print("Detector running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        img_sub.undeclare()
        depth_sub.undeclare()
        odom_sub.undeclare()
        pub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
