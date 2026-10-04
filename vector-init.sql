-- Initialize pgvector extension for embedding storage
CREATE EXTENSION IF NOT EXISTS vector;

-- Detection embeddings table (referenced by assignment-3)
CREATE TABLE IF NOT EXISTS detection_embeddings (
  det_pk bigserial PRIMARY KEY,
  run_id text,
  det_id text UNIQUE,
  keyframe_id integer,
  class_name text NOT NULL,
  confidence real NOT NULL,
  bbox real[] NOT NULL,
  map_x real,
  map_y real,
  map_yaw real,
  embedding_model text,
  embedding vector(512),
  ingested_at timestamptz NOT NULL DEFAULT now()
);

-- One row represents the current estimate of one physical object
CREATE TABLE IF NOT EXISTS semantic_objects (
    object_id BIGSERIAL PRIMARY KEY,
    class_name TEXT NOT NULL,
    map_x DOUBLE PRECISION NOT NULL,
    map_y DOUBLE PRECISION NOT NULL,
    map_z DOUBLE PRECISION NOT NULL,
    observation_count INTEGER NOT NULL CHECK (observation_count > 0),
    mean_embedding vector(512) NOT NULL,
    first_seen_ns BIGINT NOT NULL,
    last_seen_ns BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS semantic_objects_class_name_idx
    ON semantic_objects (class_name);

-- Individual observations used to form each semantic object estimate
CREATE TABLE IF NOT EXISTS object_observations (
    observation_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL
        REFERENCES semantic_objects (object_id)
        ON DELETE CASCADE,
    seen_at_ns BIGINT NOT NULL,

    map_x DOUBLE PRECISION NOT NULL,
    map_y DOUBLE PRECISION NOT NULL,
    map_yaw DOUBLE PRECISION NOT NULL,

    camera_x DOUBLE PRECISION NOT NULL,
    camera_y DOUBLE PRECISION NOT NULL,
    camera_z DOUBLE PRECISION NOT NULL,
    camera_qx DOUBLE PRECISION NOT NULL,
    camera_qy DOUBLE PRECISION NOT NULL,
    camera_qz DOUBLE PRECISION NOT NULL,
    camera_qw DOUBLE PRECISION NOT NULL,

    bearing_rad DOUBLE PRECISION NOT NULL,
    range_m DOUBLE PRECISION NOT NULL,

    bbox_xmin INTEGER NOT NULL,
    bbox_ymin INTEGER NOT NULL,
    bbox_xmax INTEGER NOT NULL,
    bbox_ymax INTEGER NOT NULL,
    confidence REAL NOT NULL,
    embedding vector(512) NOT NULL
);

CREATE INDEX IF NOT EXISTS object_observations_object_id_idx
    ON object_observations (object_id);