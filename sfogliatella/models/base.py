"""Base model interface and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Common model config keys (shared across all models)
# ---------------------------------------------------------------------------

COMMON_CONFIG_KEYS = {
    "task", "num_classes", "lookback", "horizon",
    "input_dim", "output_dim", "seed",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    run_id: str
    model_dir: str
    train_losses: List[float] = field(default_factory=list)
    val_losses:   List[float] = field(default_factory=list)
    metrics:      Dict[str, Any] = field(default_factory=dict)
    config:       Dict[str, Any] = field(default_factory=dict)
    status:       str = "done"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":       self.run_id,
            "model_dir":    self.model_dir,
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "metrics":      self.metrics,
            "config":       self.config,
            "status":       self.status,
        }


@dataclass
class PredictResult:
    run_id:       str
    predictions:  np.ndarray
    output_path:  Optional[str] = None
    metrics:      Dict[str, Any] = field(default_factory=dict)
    status:       str = "done"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":      self.run_id,
            "output_path": self.output_path,
            "metrics":     self.metrics,
            "n_predictions": len(self.predictions),
            "status":      self.status,
        }


# ---------------------------------------------------------------------------
# Head helpers: build output projection sizes
# ---------------------------------------------------------------------------

def output_dim_for_task(task: str, num_classes: int = 2, horizon: int = 1) -> int:
    """Number of raw output units for the given task.

    Binary classification uses 1 output unit (sigmoid BCE).
    Multiclass uses num_classes output units (softmax CE).
    """
    if task == "classification":
        return 1 if num_classes <= 2 else num_classes
    elif task == "embedding":
        return 64  # default embedding dim
    elif task in ("ranking", "anomaly_score", "probability"):
        return horizon  # scalar per step
    elif task == "clustering":
        return 64  # embedding for k-means
    else:
        return horizon  # regression


def apply_output_head(logits, task: str, num_classes: int = 2):
    """Convert raw logits to task output (JAX)."""
    import jax
    import jax.numpy as jnp

    if task == "classification":
        if num_classes <= 2:
            return jax.nn.sigmoid(logits)   # binary proba — logits shape (..., 1) or (...)
        else:
            return jax.nn.softmax(logits, axis=-1)    # multiclass proba
    elif task in ("regression", "ranking", "anomaly_score", "probability"):
        return logits                                   # raw float
    elif task in ("embedding", "clustering"):
        norm = jnp.linalg.norm(logits, axis=-1, keepdims=True)
        return logits / jnp.maximum(norm, 1e-8)        # L2 normalized
    return logits
