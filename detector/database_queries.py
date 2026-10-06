import argparse
import psycopg2

def find_nearest_object(cur, class_name, map_x, map_y):
    cur.execute(
        """
        SELECT
            object_id,
            class_name,
            map_x,
            map_y,
            map_z,
            sqrt(
                power(map_x - %s, 2) +
                power(map_y - %s, 2)
            ) AS distance
        FROM semantic_objects
        WHERE class_name = %s
        ORDER BY distance
        LIMIT 1;
        """,
        (map_x, map_y, class_name),
    )

    return cur.fetchone()

def list_objects_in_region(cur, min_x, max_x, min_y, max_y):
    cur.execute(
        """
        SELECT
            object_id,
            class_name,
            map_x,
            map_y,
            map_z,
            observation_count
        FROM semantic_objects
        WHERE map_x BETWEEN %s AND %s
          AND map_y BETWEEN %s AND %s
        ORDER BY class_name, object_id;
        """,
        (min_x, max_x, min_y, max_y),
    )

    return cur.fetchall()

def get_class_histogram(cur):
    cur.execute(
        """
        SELECT
            class_name,
            count(*) AS object_count
        FROM semantic_objects
        GROUP BY class_name
        ORDER BY object_count DESC, class_name;
        """
    )

    return cur.fetchall()

def get_object_observations(cur, object_id):
    cur.execute(
        """
        SELECT
            observation_id,
            seen_at_ns,
            map_x,
            map_y,
            map_yaw,
            camera_x,
            camera_y,
            camera_z,
            bearing_rad,
            range_m,
            confidence
        FROM object_observations
        WHERE object_id = %s
        ORDER BY seen_at_ns;
        """,
        (object_id,),
    )

    return cur.fetchall()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5436)
    parser.add_argument("--database", default="vectordb")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
    )

    print("Connected to semantic database.")
    cur = conn.cursor()

    print("\nClass histogram:")
    for row in get_class_histogram(cur):
        print(row)

    print("\nNearest chair to (0, 0):")
    print(find_nearest_object(cur, "chair", 0.0, 0.0))    

    print("\nObjects inside region (0 to 10, 0 to 10):")
    for row in list_objects_in_region(cur, 0.0, 10.0, 0.0, 10.0):
        print(row)

    print("\nObservation history for object 1:")
    for row in get_object_observations(cur, 1):
        print(row)

    cur.close()

    conn.close()


if __name__ == "__main__":
    main()
