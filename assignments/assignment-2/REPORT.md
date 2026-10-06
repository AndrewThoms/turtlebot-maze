Task 1: Detector node
Changes Made
object_detector.py
- Updated the detector to meet the Task 1 requirements
- Set the detection confidence to 0.3
- Added movement-based keyframes using 0.5 m movement or 15° rotation
- Added YOLOv8 object detection and CLIP embeddings for detected objects
- Prevented new keyframes from being created while the robot is stationary
docker-compose.yaml
- Updated the detector configuration to run on my CPU-only system.
- Fixed file permissions needed to download and store the model files
- Added a writable location for the CLIP model cache
Testing
- KF#1 was created at startup
- KF#2 appeared after rotating the robot
- KF#3 appeared after moving the robot forward
- YOLO and CLIP both loaded and ran successfully

Task 2: From bounding box to map coordinates
Changes Made
object_detector.py
-Added depth and odom data to convert 2D object detection into a 3D position
-RGB, depth, and odom are matched by timestamp
-camera-to-robot-to-map transformations are applied
Testing and Verification
-Tested the coordinate transformation with known values using a unit test, which passed
-The full system was also tested in simulation by moving and rotating the robot
-confirmed that detected objects such as chairs and tables were successfully published with 3D map coordinates

Task 3: Semantic object database
Changes Made
-Added the required semantic_objects and object_observations PostgreSQL tables.
object_detector.py
-Updated the detector to show: robot pose, camera pose, object map position, bearing and range, 512-D embedding	
embedding_ingest.py
- Store every observation.
- Match repeated detections to existing objects.
- Update object position, observation count, embedding, and timestamps.
Testing and Verification
-both database tables exist.
-detections are successfully stored in PostgreSQL.
-Observed the same objects from different robot positions.
-observation_count increased
-stored embeddings are 512 dimensions.
-observation counts match the number of stored observations.

Task 4 – Deduplication
Changes made
embedding_ingest.py
- Added a distance radius for matching detections to existing objects.
- Only objects with the same class can match.
- If multiple objects are nearby, CLIP similarity chooses the best match.
- When objects match, their position and embedding are averaged.
- Every detection is saved in object_observations.
Testing and Verification
- Tested repeated detections of the dining chairs
- Measured about 4–9 cm of range variation
- Different viewpoints produced up to about 0.45 m position difference
- Tested different association radii:
  - 0.30 m: 8 chair observations → 4 chair objects
  - 0.75 m: so far, 6 chair observations → 2 chair objects (only two physical chairs were observed in this part of the test)


Task 5: Querying the database
changes made
- Added database_queries.py
- Added the 4 required database queries
- All SQL inputs parameterized
- Added plot_semantic_map.py
- Generated semantic_map.png 
Testing and Verification
- All 4 database queries worked
- Map plot generated successfully
- Database currently contains 2 chairs and 1 umbrella
- House ground truth has 8 chairs
- Chair results: 100% precision, 25% recall
