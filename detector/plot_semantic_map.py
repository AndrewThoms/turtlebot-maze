import argparse

import matplotlib.pyplot as plt
import psycopg2


def main():
    parser = argparse.ArgumentParser(description="Plot semantic map")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5436)
    parser.add_argument("--database", default="vectordb")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--output", default="semantic_map.png")
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
    )
    cur = conn.cursor()

    # Fused semantic object positions.
    cur.execute(
        """
        SELECT object_id, class_name, map_x, map_y
        FROM semantic_objects
        ORDER BY object_id;
        """
    )
    objects = cur.fetchall()

    # Accepted robot poses, ordered by observation time.
    cur.execute(
        """
        SELECT map_x, map_y
        FROM object_observations
        ORDER BY seen_at_ns;
        """
    )
    robot_poses = cur.fetchall()

    cur.close()
    conn.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    if robot_poses:
        route_x = [row[0] for row in robot_poses]
        route_y = [row[1] for row in robot_poses]

        ax.plot(
            route_x,
            route_y,
            marker=".",
            label="Accepted robot poses",
        )

    for object_id, class_name, map_x, map_y in objects:
        ax.scatter(map_x, map_y, s=70)
        ax.annotate(
            f"{class_name} {object_id}",
            (map_x, map_y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.set_title("Semantic Object Map")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved map plot to {args.output}")


if __name__ == "__main__":
    main()