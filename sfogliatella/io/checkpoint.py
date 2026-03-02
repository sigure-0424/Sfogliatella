"""Checkpoint save and load for JAX/equinox models and XGBoost."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "weights.pkl"
OPT_FILE     = "opt_state.pkl"
META_FILE    = "checkpoint_meta.json"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: Any,
    opt_state: Any,
    step: int,
    epoch: int,
    config: Dict[str, Any],
    ckpt_dir: Path,
    run_id: str,
    rng: Optional[Any] = None,
) -> Path:
    """Save model weights, optimizer state, and meta to ckpt_dir/run_id/step/."""
    ckpt_path = ckpt_dir / run_id / f"step_{step:08d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    from sfogliatella.models.xgboost_model import XGBoostModel

    if isinstance(model, XGBoostModel):
        model.save(str(ckpt_path / "xgb_model"))
    else:
        # Use equinox's tree_serialise_leaves for portable, structure-aware saving
        import equinox as eqx
        eqx.tree_serialise_leaves(str(ckpt_path / WEIGHTS_FILE), model)

    # Optimizer state — use equinox serialization for consistency
    try:
        import equinox as eqx
        eqx.tree_serialise_leaves(str(ckpt_path / OPT_FILE), opt_state)
    except Exception as e:
        logger.warning("Could not save optimizer state: %s", e)

    # Meta
    meta = {
        "step": step,
        "epoch": epoch,
        "run_id": run_id,
        "config": config,
    }
    if rng is not None:
        try:
            meta["rng"] = np.array(rng).tolist()
        except Exception:
            pass

    with open(ckpt_path / META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    logger.debug("Checkpoint saved: %s", ckpt_path)
    return ckpt_path


def get_latest_checkpoint(ckpt_dir: Path, run_id: str) -> Optional[Path]:
    """Find the latest checkpoint directory."""
    base = ckpt_dir / run_id
    if not base.exists():
        return None
    steps = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    return steps[-1] if steps else None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    model_template: Any,
    ckpt_path: Path,
    opt_state_template: Optional[Any] = None,
) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """Load model (and optionally optimizer state) from checkpoint.

    Returns (model, opt_state, meta).
    """
    ckpt_path = Path(ckpt_path)

    # Load meta
    meta_file = ckpt_path / META_FILE
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
    else:
        meta = {}

    from sfogliatella.models.xgboost_model import XGBoostModel

    if isinstance(model_template, XGBoostModel):
        model_template.load(str(ckpt_path / "xgb_model"))
        return model_template, None, meta

    # Load equinox model using tree_deserialise_leaves (requires model template)
    import equinox as eqx
    weights_file = ckpt_path / WEIGHTS_FILE
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    model = eqx.tree_deserialise_leaves(str(weights_file), model_template)

    # Load optimizer state using equinox deserialization (needs template for structure)
    opt_state = opt_state_template
    opt_file = ckpt_path / OPT_FILE
    if opt_state_template is not None and opt_file.exists():
        try:
            import equinox as eqx
            opt_state = eqx.tree_deserialise_leaves(str(opt_file), opt_state_template)
        except Exception as e:
            logger.warning("Could not restore optimizer state: %s", e)

    logger.debug("Checkpoint loaded: %s", ckpt_path)
    return model, opt_state, meta
