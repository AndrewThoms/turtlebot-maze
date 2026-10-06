#!/usr/bin/env python3
"""
Zenoh-to-pgvector ingest worker.

Subscribes to tb/detections on the Zenoh router, decodes CLIP embeddings
from the JSON payload, and writes them to the pgvector PostgreSQL service.
"""

import argparse
import base64
import json
import math
import time

import numpy as np
import psycopg2
import zenoh


def decode_embedding(b64_str: str) -> list[float]:
    """Decode a base64-encoded float32 vector to a Python list."""
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=np.float32).tolist()


def main():
    parser = argparse.ArgumentParser(description="Zenoh → pgvector ingest worker")
    parser.add_argument(
        "-e",
        "--connect",
        type=str,
        default="tcp/localhost:7447",
        help="Zenoh endpoint to connect to",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="tb/detections",
        help="Zenoh key to subscribe to",
    )
    parser.add_argument(
        "--pg-host", type=str, default="localhost", help="PostgreSQL host"
    )
    parser.add_argument(
        "--pg-port", type=int, default=5436, help="PostgreSQL port"
    )
    parser.add_argument(
        "--pg-db", type=str, default="vectordb", help="PostgreSQL database"
    )
    parser.add_argument(
        "--pg-user", type=str, default="postgres", help="PostgreSQL user"
    )
    parser.add_argument(
        "--pg-password", type=str, default="postgres", help="PostgreSQL password"
    )
    parser.add_argument(
        "--association-distance",
        type=float,
        default=1.0,
        help="Maximum map distance in metres for matching the same object",
    )
    args = parser.parse_args()
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=args.pg_host,
        port=args.pg_port,
        dbname=args.pg_db,
        user=args.pg_user,
        password=args.pg_password,
    )
    conn.autocommit = False
    cur = conn.cursor()
    print(f"Connected to PostgreSQL: {args.pg_host}:{args.pg_port}/{args.pg_db}")

    insert_count = 0
#task 3 update
    observation_count = 0

    def find_matching_object(class_name, x, y, z, embedding):
        cur.execute(
            """
            SELECT
                object_id,
                map_x,
                map_y,
                map_z,
                observation_count,
                mean_embedding::text,
                first_seen_ns,
                mean_embedding <=> %s::vector AS cosine_distance
            FROM semantic_objects
            WHERE class_name = %s
            AND sqrt(
                power(map_x - %s, 2) +
                power(map_y - %s, 2) +
                power(map_z - %s, 2)
            ) <= %s
            ORDER BY cosine_distance ASC
            """,
            (
                str(embedding),
                class_name,
                x,
                y,
                z,
                args.association_distance,
            ),
        )

        candidates = cur.fetchall()

        if not candidates:
            return None

        return candidates[0]

    def create_semantic_object(class_name, x, y, z, embedding, seen_at_ns):
        cur.execute(
            """
            INSERT INTO semantic_objects
                (class_name, map_x, map_y, map_z,
                 observation_count, mean_embedding,
                 first_seen_ns, last_seen_ns)
            VALUES (%s, %s, %s, %s, 1, %s, %s, %s)
            RETURNING object_id
            """,
            (
                class_name,
                x,
                y,
                z,
                str(embedding),
                seen_at_ns,
                seen_at_ns,
            ),
        )
        return cur.fetchone()[0]

    def update_semantic_object(row, x, y, z, embedding, seen_at_ns):
        """
        Update position and embedding using incremental arithmetic means.
        """
        (
            object_id,
            old_x,
            old_y,
            old_z,
            count,
            old_embedding_text,
            first_seen_ns,
            cosine_distance,
        ) = row

        new_count = count + 1

        new_x = (old_x * count + x) / new_count
        new_y = (old_y * count + y) / new_count
        new_z = (old_z * count + z) / new_count

        old_embedding = np.fromstring(
            old_embedding_text.strip("[]"),
            sep=",",
            dtype=np.float32,
        )
        mean_embedding = (
            old_embedding * count + np.asarray(embedding, dtype=np.float32)
        ) / new_count

        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        cur.execute(
            """
            UPDATE semantic_objects
            SET map_x = %s,
                map_y = %s,
                map_z = %s,
                observation_count = %s,
                mean_embedding = %s,
                last_seen_ns = %s
            WHERE object_id = %s
            """,
            (
                new_x,
                new_y,
                new_z,
                new_count,
                str(mean_embedding.tolist()),
                seen_at_ns,
                object_id,
            ),
        )

        return object_id

    insert_count = 0
    observation_count = 0
    def detection_callback(sample):
        nonlocal insert_count, observation_count
        try:
            msg = json.loads(sample.payload.to_bytes())
        except Exception as e:
            print(f"JSON decode error: {e}")
            return

        # Handle envelope format: {keyframe_id, timestamp, map_x, map_y, map_yaw, detections: [...]}
        if isinstance(msg, dict) and "detections" in msg:
            detections = msg["detections"]
            kf_id = msg.get("keyframe_id")
            run_id = msg.get("run_id")
            map_x = msg.get("map_x")
            map_y = msg.get("map_y")
            map_yaw = msg.get("map_yaw")
            
            timestamp_ns = msg.get("timestamp_ns")

            camera_map_x = msg.get("camera_map_x")
            camera_map_y = msg.get("camera_map_y")
            camera_map_z = msg.get("camera_map_z")
            camera_qx = msg.get("camera_qx")
            camera_qy = msg.get("camera_qy")
            camera_qz = msg.get("camera_qz")
            camera_qw = msg.get("camera_qw")
        elif isinstance(msg, list):
            # Legacy flat list format
            detections = msg
            kf_id = None
            run_id = None
            map_x = map_y = map_yaw = None
        else:
            return

        for det in detections:
            embedding_b64 = det.get("embedding")
            if embedding_b64 is None:
                continue

            embedding = decode_embedding(embedding_b64)
            det_id = det.get("det_id")
            cur.execute(
                """INSERT INTO detection_embeddings
                   (run_id, det_id, keyframe_id, class_name, confidence, bbox,
                    map_x, map_y, map_yaw, embedding_model, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (det_id) DO NOTHING""",
                (
                    run_id,
                    det_id,
                    kf_id,
                    det["class"],
                    det["confidence"],
                    det["bbox"],
                    map_x,
                    map_y,
                    map_yaw,
                    det.get("embedding_model"),
                    str(embedding),
                ),
            )

            # Task 3 requires complete geometry and a 512-D embedding.
            required_values = [
                timestamp_ns,
                map_x,
                map_y,
                map_yaw,
                camera_map_x,
                camera_map_y,
                camera_map_z,
                camera_qx,
                camera_qy,
                camera_qz,
                camera_qw,
                det.get("map_x"),
                det.get("map_y"),
                det.get("map_z"),
                det.get("bearing_rad"),
                det.get("range_m"),
            ]

            if any(value is None for value in required_values):
                print(f"Skipping Task 3 row for {det_id}: missing geometry")
                continue

            if len(embedding) != 512:
                print(
                    f"Skipping Task 3 row for {det_id}: "
                    f"embedding has {len(embedding)} dimensions"
                )
                continue

            object_row = find_matching_object(
                det["class"],
                det["map_x"],
                det["map_y"],
                det["map_z"],
                embedding,
            )

            if object_row is None:
                object_id = create_semantic_object(
                    det["class"],
                    det["map_x"],
                    det["map_y"],
                    det["map_z"],
                    embedding,
                    timestamp_ns,
                )
            else:
                object_id = update_semantic_object(
                    object_row,
                    det["map_x"],
                    det["map_y"],
                    det["map_z"],
                    embedding,
                    timestamp_ns,
                )

            bbox = det["bbox"]

            # Preserve the whole detected box when converting the floating
            # YOLO coordinates to the integer database columns.
            bbox_xmin = math.floor(bbox[0])
            bbox_ymin = math.floor(bbox[1])
            bbox_xmax = math.ceil(bbox[2])
            bbox_ymax = math.ceil(bbox[3])

            cur.execute(
                """
                INSERT INTO object_observations
                    (object_id, seen_at_ns,
                     map_x, map_y, map_yaw,
                     camera_x, camera_y, camera_z,
                     camera_qx, camera_qy, camera_qz, camera_qw,
                     bearing_rad, range_m,
                     bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
                     confidence, embedding)
                VALUES
                    (%s, %s,
                     %s, %s, %s,
                     %s, %s, %s,
                     %s, %s, %s, %s,
                     %s, %s,
                     %s, %s, %s, %s,
                     %s, %s)
                """,
                (
                    object_id,
                    timestamp_ns,
                    map_x,
                    map_y,
                    map_yaw,
                    camera_map_x,
                    camera_map_y,
                    camera_map_z,
                    camera_qx,
                    camera_qy,
                    camera_qz,
                    camera_qw,
                    det["bearing_rad"],
                    det["range_m"],
                    bbox_xmin,
                    bbox_ymin,
                    bbox_xmax,
                    bbox_ymax,
                    det["confidence"],
                    str(embedding),
                ),
            )

            observation_count += 1
            conn.commit()
            insert_count += 1

        if detections:
            classes = [d["class"] for d in detections if d.get("embedding")]
            if classes:
                print(
                    f"KF#{kf_id} [{run_id}] Ingested {len(classes)} embeddings "
                    f"(total: {insert_count}): {classes}"
                )

    # Open Zenoh session
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps([args.connect]))

    session = zenoh.open(conf)
    sub = session.declare_subscriber(args.key, detection_callback)
    print(f"Subscribed to: {args.key}")
    print("Ingest worker running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sub.undeclare()
        session.close()
        cur.close()
        conn.close()
        print(f"Shutdown. Total embeddings ingested: {insert_count}")


if __name__ == "__main__":
    main()
