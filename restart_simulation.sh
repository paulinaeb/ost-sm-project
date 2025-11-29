#!/usr/bin/env bash
set -Eeuo pipefail

COMPOSE_FILE="docker-compose.yml"
CASSANDRA_CONTAINER="cassandra-dev"
LINKEDIN_KEYSPACE="linkedin_jobs"
LINKEDIN_TABLE="jobs"

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

main() {
  echo "[1/4] Prechecks..."
  have docker || { echo "ERROR: docker not found." >&2; exit 1; }
  docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not running." >&2; exit 1; }
  COMPOSE="$(resolve_compose)"
  [ -f "$COMPOSE_FILE" ] || { echo "ERROR: $COMPOSE_FILE not found." >&2; exit 1; }

  echo "[2/4] Truncating LinkedIn jobs data..."
  if docker exec -i "$CASSANDRA_CONTAINER" cqlsh -e "TRUNCATE $LINKEDIN_KEYSPACE.$LINKEDIN_TABLE" 2>/dev/null; then
    echo "LinkedIn jobs table truncated successfully."
  else
    echo "WARNING: Could not truncate table. It may not exist yet."
  fi

  echo "[3/4] Starting Kafka consumer..."
  $COMPOSE -f "$COMPOSE_FILE" up -d kafka-consumer
  echo "Waiting for Consumer to be fully ready..."
  sleep 3

  echo "[4/4] Starting Kafka producer..."
  $COMPOSE -f "$COMPOSE_FILE" up -d kafka-producer

  echo
  echo "Stream mining process restarted successfully!"
  echo
  echo "To monitor the streaming process:"
  echo "  - Consumer logs: docker logs -f kafka-consumer"
  echo "  - Producer logs: docker logs -f kafka-producer"
  echo "  - Kafka UI: http://localhost:8080"
  echo
}

main "$@"