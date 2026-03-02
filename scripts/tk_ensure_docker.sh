#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/compose.yaml"

if ! command -v docker >/dev/null 2>&1; then
  echo "[tk_ensure_docker] docker not found; skipping startup."
  exit 0
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "[tk_ensure_docker] $COMPOSE_FILE not found; skipping startup."
  exit 0
fi

if ! docker info >/dev/null 2>&1; then
  echo "[tk_ensure_docker] docker daemon unavailable; skipping startup."
  exit 0
fi

docker compose -f "$COMPOSE_FILE" up -d >/dev/null 2>&1 || {
  echo "[tk_ensure_docker] docker compose startup failed; skipping."
  exit 0
}

echo "[tk_ensure_docker] docker compose is up."
