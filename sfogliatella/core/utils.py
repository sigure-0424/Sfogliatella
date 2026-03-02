"""Shared utilities."""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def make_run_id(run_id: Optional[str] = None) -> str:
    if run_id:
        return str(run_id)
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_LEVEL_MAP = {
    "error": logging.ERROR,
    "warn":  logging.WARNING,
    "info":  logging.INFO,
    "debug": logging.DEBUG,
}


def setup_logging(level: str = "info", log_items: Optional[str] = None) -> None:
    """Configure root logger."""
    lvl = LOG_LEVEL_MAP.get(level.lower(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # stderr keeps stdout clean for --return_json piping
        force=True,
    )
    # Suppress noisy third-party loggers by default
    for noisy in ["absl", "jax._src", "jaxlib", "watchdog"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json_or_yaml(path_or_str: str) -> Dict[str, Any]:
    """Load JSON/YAML from a path or an inline JSON string."""
    p = Path(path_or_str)
    if p.exists():
        content = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(content)
        return json.loads(content)
    # Try as inline JSON string
    return json.loads(path_or_str)


def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):      # numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):    # numpy arrays
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------

def get_version_info() -> Dict[str, str]:
    import sfogliatella
    info: Dict[str, str] = {"sfogliatella": sfogliatella.__version__}

    for pkg in ["jax", "equinox", "optax", "numpy", "pandas", "xgboost"]:
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = "not installed"

    info["python"] = sys.version.split()[0]
    return info


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def nan_inf_check(loss: float, step: int) -> bool:
    """Return True if loss is problematic (NaN/Inf)."""
    import math
    if math.isnan(loss):
        logger.warning("Step %d: NaN loss detected!", step)
        return True
    if math.isinf(loss):
        logger.warning("Step %d: Inf loss detected!", step)
        return True
    return False
