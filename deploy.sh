# bash
// filepath: deploy.sh
#!/usr/bin/env bash
set -Eeuo pipefail

COMPOSE_FILE="docker-compose.yml"
KEYSPACE="linkedin_jobs"
CASSANDRA_CONTAINER="cassandra-dev"
KAFKA_CONTAINER="kafka"
APP_CONTAINER="python-app"
STREAMLIT_URL="http://localhost:8501"
SCHEMA_FILE="preprocessing/ECSF/keyspace_tables_creation.sql"

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
  $COMPOSE -f "$COMPOSE_FILE" up -d

  echo "[5/7] Waiting for dependencies..."
  wait_health "$CASSANDRA_CONTAINER" 300
  wait_health "$KAFKA_CONTAINER" 300

  echo "[6/7] Initializing Cassandra schema (idempotent)..."
  if docker exec -i "$CASSANDRA_CONTAINER" cqlsh -e "DESCRIBE KEYSPACE $KEYSPACE" >/dev/null 2>&1; then
    echo "Keyspace '$KEYSPACE' already exists. Skipping schema init."
  else
    if [ -f "$SCHEMA_FILE" ]; then
      docker exec -i "$CASSANDRA_CONTAINER" cqlsh < "$SCHEMA_FILE"
      echo "Schema initialized from $SCHEMA_FILE."
    else
      echo "WARNING: $SCHEMA_FILE not found. Skipping schema init."
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