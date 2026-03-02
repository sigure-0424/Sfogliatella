#!/usr/bin/env bash
set -euo pipefail

tk_repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || pwd
}
