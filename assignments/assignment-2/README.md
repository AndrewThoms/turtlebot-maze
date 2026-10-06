# Assignment 2 — Object detection and semantic localization in the house world

commands are below to run the implementation against the compose services.

Tasks 1 and 2

    Start the simulation infrastructure:
    docker compose up -d demo-world-house zenoh-router zenoh-bridge rosbridge
    
    wait, then open a second terminal:
    cd ~/turtlebot-maze
    docker compose up detector
    
    wait, Open a third terminal:
    cd ~/turtlebot-maze
    source .venv-robot/bin/activate
    
    move robot
    python move_robot.py left
    python move_robot.py forward
    

Task 3

    start database
    docker compose up -d vector
    
    start detector
    docker compose up --force-recreate detector
    
    start ingest
    docker compose run --rm embedding-ingest
    
    move robot
    source .venv-robot/bin/activate
    python move_robot.py left
    python move_robot.py forward
    

Task 4

    Start embedding-ingest
    docker compose run --rm embedding-ingest
    
    start detector
    docker compose up --force-recreate detector
    
    move robot
    source .venv-robot/bin/activate
    python move_robot.py left
    python move_robot.py forward
    
    Run with a 0.30 m radius:
    docker compose run --rm embedding-ingest python embedding_ingest.py --association-distance 0.30
    
    Run with a 0.75 m radius:
    docker compose run --rm embedding-ingest python embedding_ingest.py --association-distance 0.75
    

Task 5

    Run the database-query demonstration:
    docker compose run --rm embedding-ingest python database_queries.py
    
    Generate the semantic map:
    docker compose run --rm embedding-ingest python plot_semantic_map.py --output semantic_map.png

