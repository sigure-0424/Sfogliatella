#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
STATE_FILE="$ROOT_DIR/docs/core/STATE.yaml"

read_state_value() {
  local key="$1"
  if [ -f "$STATE_FILE" ]; then
    grep -E "^${key}:" "$STATE_FILE" | head -n1 | sed -E "s/^${key}:[[:space:]]*//" | sed -E 's/^"(.*)"$/\1/'
  fi
}

abbrev_name() {
  local name="$1"
  local max_len=20
  if [ ${#name} -le $max_len ]; then
    printf "%s" "$name"
    return
  fi
  local cut="${name:0:$max_len}"
  cut="$(printf "%s" "$cut" | sed -E 's/[[:space:]_-]+$//')"
  printf "%s..." "$cut"
}

PROJECT_SHORT_NAME="$(read_state_value "project_short_name")"
PROJECT_NAME="$(read_state_value "project_name")"
PROTOCOL_VERSION="$(read_state_value "protocol_version")"

if [ -n "${PROJECT_SHORT_NAME:-}" ]; then
  DISPLAY_NAME="$PROJECT_SHORT_NAME"
elif [ -n "${PROJECT_NAME:-}" ]; then
  DISPLAY_NAME="$(abbrev_name "$PROJECT_NAME")"
else
  DISPLAY_NAME="Project"
fi

GIT_COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "${WSL_DISTRO_NAME:-}" ]; then
  WORKSPACE="WSL"
elif [ -f /.dockerenv ]; then
  WORKSPACE="Docker"
else
  WORKSPACE="Local"
fi

printf "\n=== %s ===\n" "$DISPLAY_NAME"
printf "PROTOCOL_VERSION=%s\n" "${PROTOCOL_VERSION:-unknown}"
printf "GIT_COMMIT_SHA=%s\n" "$GIT_COMMIT_SHA"
printf "WORKSPACE=%s\n\n" "$WORKSPACE"
