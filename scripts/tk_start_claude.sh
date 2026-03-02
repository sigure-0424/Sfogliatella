#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
"$ROOT_DIR/scripts/print_banner.sh"

BOOT_PROMPT=$'BOOT\n1) Read EVERYTHING in docs/core/ before any reasoning.\n2) Report only STATE fields (quote-only, no invention).\n3) If active_tasks is empty, enumerate docs/proposed/ and propose next actions.\n4) Respect No-Confirm Mode: if operator says no approval required, execute without asking.\n5) Handle transient locks with wait/retry, switch tasks, and retry later.'

exec claude --dangerously-skip-permissions -- "$BOOT_PROMPT"
