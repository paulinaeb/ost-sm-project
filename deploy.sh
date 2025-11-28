#!/usr/bin/env bash
set -Eeuo pipefail

COMPOSE_FILE="docker-compose.yml"
LINKEDIN_KEYSPACE="linkedin_jobs"
LINKEDIN_TABLE="jobs"
CASSANDRA_CONTAINER="cassandra-dev"
KAFKA_CONTAINER="kafka"
APP_CONTAINER="python-app"
STREAMLIT_URL="http://localhost:8501"
SCHEMA_FILE="preprocessing/ECSF/keyspace_tables_creation.sql"
ECSF_KEYSPACE="ecsf"

have() { command -v "$1" >/dev/null 2>&1; }

resolve_compose() {
  if have docker && docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  elif have docker-compose; then
    echo "docker-compose"
  else
    echo "ERROR: docker compose not installed." >&2
    exit 1
  fi
}

wait_health() {
  local name="$1" timeout="${2:-300}" waited=0
  echo "Waiting for $name to be healthy (timeout ${timeout}s)..."
  until [ "$(docker inspect -f '{{.State.Health.Status}}' "$name" 2>/dev/null || echo starting)" = "healthy" ]; do
    sleep 5
    waited=$((waited+5))
    if [ "$waited" -ge "$timeout" ]; then
      echo "ERROR: $name did not become healthy in time." >&2
      docker logs "$name" --tail 50 || true
      exit 1
    fi
  done
  echo "$name is healthy."
}

main() {
  echo "[1/7] Prechecks..."
  have docker || { echo "ERROR: docker not found." >&2; exit 1; }
  docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not running." >&2; exit 1; }
  COMPOSE="$(resolve_compose)"
  [ -f "$COMPOSE_FILE" ] || { echo "ERROR: $COMPOSE_FILE not found." >&2; exit 1; }
  [ -f "requirements.txt" ] || { echo "ERROR: requirements.txt not found." >&2; exit 1; }
  [ -f "forecasting/streamlit_app.py" ] || { echo "ERROR: forecasting/streamlit_app.py not found." >&2; exit 1; }

  echo "[2/7] Building app image..."
  $COMPOSE -f "$COMPOSE_FILE" build python-app

  echo "[3/7] Pulling external images..."
  $COMPOSE -f "$COMPOSE_FILE" pull cassandra-dev kafka kafka-ui cassandra-web

  echo "[4/7] Starting services..."
  $COMPOSE -f "$COMPOSE_FILE" up -d cassandra-dev kafka kafka-ui cassandra-web python-app

  echo "[5/7] Waiting for dependencies..."
  wait_health "$CASSANDRA_CONTAINER" 300
  wait_health "$KAFKA_CONTAINER" 300

   # Give Cassandra extra time to fully initialize
   echo "Waiting for Cassandra to be fully ready..."
   sleep 10

  echo "[6/7] Initializing Cassandra schema (ECSF keyspace)..."
  
  if docker exec -i "$CASSANDRA_CONTAINER" cqlsh -e "DESCRIBE KEYSPACE $ECSF_KEYSPACE" >/dev/null 2>&1; then
    echo "Keyspace '$ECSF_KEYSPACE' already exists."
  else
    if [ -f "$SCHEMA_FILE" ]; then
      for i in 1 2 3; do
        echo "Attempting schema init (try $i/3)..."
        if docker exec -i "$CASSANDRA_CONTAINER" cqlsh < "$SCHEMA_FILE" 2>/dev/null; then
          echo "Schema initialized from $SCHEMA_FILE."
          break
        else
          if [ "$i" -lt 3 ]; then
            echo "Schema init failed, retrying in 5s..."
            sleep 5
          else
            echo "ERROR: Schema init failed after 3 attempts."
            exit 1
          fi
        fi
      done
    else
      echo "WARNING: $SCHEMA_FILE not found. Skipping schema init."
    fi
  fi
  echo "Checking if ECSF data already loaded..."
  # Check if work_role_by_id table has data (adjust table name if needed)
  # 1. Extract the specific number from the database (digits only)
  # We use grep -o to get only numbers, and head -n 1 to take the first number (the count)
  ROW_COUNT=$(docker exec -i "$CASSANDRA_CONTAINER" cqlsh --no-color -e "SELECT COUNT(*) FROM $ECSF_KEYSPACE.work_role_by_id LIMIT 10" 2>/dev/null | grep -o '[0-9]\+' | head -n 1)

  # 2. Safety: If the connection failed and ROW_COUNT is empty, treat it as 0
  ROW_COUNT=${ROW_COUNT:-0}

  # 3. Numeric Check: Is ROW_COUNT less than (-lt) 10?
  if [ "$ROW_COUNT" -lt 10 ]; then
    echo "No ECSF data found. Loading..."
    NETWORK_NAME=$($COMPOSE -f "$COMPOSE_FILE" ps -q cassandra-dev | xargs docker inspect -f '{{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}' | head -1 | xargs docker network inspect -f '{{.Name}}')
    
    docker run --rm --network "$NETWORK_NAME" \
      -e CASSANDRA_HOSTS=cassandra-dev \
      -e CASSANDRA_PORT=9042 \
      -e CASSANDRA_KEYSPACE=ecsf \
      csoma-streamlit:latest \
      python preprocessing/ECSF/load_ecsf.py
    echo "ECSF data loaded."
  else
    echo "ECSF data already present. Skipping load."
  fi

  echo "Checking if LinkedIn jobs data already loaded..."
  LINKEDIN_KEYSPACE="linkedin_jobs"
  LINKEDIN_TABLE="jobs"
  
  # Check if keyspace and table exist with data
  if docker exec -i "$CASSANDRA_CONTAINER" cqlsh -e "SELECT COUNT(*) FROM $LINKEDIN_KEYSPACE.$LINKEDIN_TABLE LIMIT 1" 2>/dev/null | grep -q " 0 "; then
    echo "No LinkedIn jobs data found. Starting streaming pipeline..."
    
    $COMPOSE -f "$COMPOSE_FILE" up -d kafka-consumer
    echo "Waiting for Consumer to be fully ready..."
    sleep 10
    $COMPOSE -f "$COMPOSE_FILE" up -d kafka-producer
    
    
    if [ "$(docker inspect -f '{{.State.ExitCode}}' kafka-producer)" != "0" ]; then
      echo "ERROR: Producer failed."
      docker logs kafka-producer
      exit 1
    fi
    
    if [ "$(docker inspect -f '{{.State.ExitCode}}' kafka-consumer)" != "0" ]; then
      echo "ERROR: Consumer failed."
      docker logs kafka-consumer
      exit 1
    fi
    
  elif docker exec -i "$CASSANDRA_CONTAINER" cqlsh -e "DESCRIBE TABLE $LINKEDIN_KEYSPACE.$LINKEDIN_TABLE" >/dev/null 2>&1; then
    echo "LinkedIn jobs data already present. Skipping streaming pipeline."
  else
    echo "LinkedIn jobs keyspace/table doesn't exist. Starting consumer to create it..."
    
    $COMPOSE -f "$COMPOSE_FILE" up -d kafka-consumer
    echo "Waiting for Consumer to be fully ready..."
    sleep 10
    $COMPOSE -f "$COMPOSE_FILE" up -d kafka-producer
    
    if [ "$(docker inspect -f '{{.State.ExitCode}}' kafka-producer)" != "0" ]; then
      echo "ERROR: Producer failed."
      docker logs kafka-producer
      exit 1
    fi
    
    if [ "$(docker inspect -f '{{.State.ExitCode}}' kafka-consumer)" != "0" ]; then
      echo "ERROR: Consumer failed."
      docker logs kafka-consumer
      exit 1
    fi
  fi

  echo "[7/7] Waiting for Streamlit app..."
  wait_health "$APP_CONTAINER" 300
  
  echo
  echo "All services are up:"
  echo "- Streamlit: $STREAMLIT_URL"
  echo "- Kafka UI: http://localhost:8080"
  echo "- Cassandra Web: http://localhost:8081"
  echo
  $COMPOSE -f "$COMPOSE_FILE" ps
}

main "$@"