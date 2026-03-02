"""Loss function loader: built-in registry + custom plugin."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Built-in loss registry
_REGRESSION_LOSSES = {
    "mse":       "sfogliatella.losses.regression:mse_loss",
    "mae":       "sfogliatella.losses.regression:mae_loss",
    "huber":     "sfogliatella.losses.regression:huber_loss",
    "log_cosh":  "sfogliatella.losses.regression:log_cosh_loss",
    "quantile":  "sfogliatella.losses.regression:quantile_loss",
    "smape":     "sfogliatella.losses.regression:smape_loss",
    "mape":      "sfogliatella.losses.regression:mape_loss",
}

_CLASSIFICATION_LOSSES = {
    "cross_entropy":        "sfogliatella.losses.classification:cross_entropy_loss",
    "binary_cross_entropy": "sfogliatella.losses.classification:binary_cross_entropy_loss",
    "focal":                "sfogliatella.losses.classification:focal_loss",
}

_ALL_LOSSES = {**_REGRESSION_LOSSES, **_CLASSIFICATION_LOSSES}

_TASK_DEFAULTS = {
    "regression":    "mse",
    "classification": "binary_cross_entropy",  # overridden for multiclass; see get_default_loss
    "ranking":       "mse",
    "anomaly_score": "mse",
    "probability":   "mse",
    "clustering":    "mse",
    "embedding":     "mse",
}


def get_default_loss(task: str, num_classes: int = 2) -> str:
    if task == "classification":
        return "binary_cross_entropy" if num_classes <= 2 else "cross_entropy"
    return _TASK_DEFAULTS.get(task, "mse")


def load_loss_fn(
    loss_name: Optional[str] = None,
    loss_path: Optional[str] = None,
    task: str = "regression",
    config: Optional[dict] = None,
) -> Callable:
    """Load a loss function by name or from an external Python file.

    Custom file must expose either:
        build_loss(config) -> callable  [preferred]
        loss_fn(y_pred, y_true, **kwargs)
    """
    if loss_path:
        return _load_custom_loss(loss_path, config or {})

    num_classes = int((config or {}).get("num_classes", 2))
    name = loss_name or get_default_loss(task, num_classes)
    if name not in _ALL_LOSSES:
        raise ValueError(f"Unknown loss: {name!r}. Available: {list(_ALL_LOSSES)}")

    module_path, fn_name = _ALL_LOSSES[name].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    logger.debug("Loaded built-in loss: %s", name)
    return fn


def _load_custom_loss(loss_path: str, config: dict) -> Callable:
    path = Path(loss_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Custom loss file not found: {path}")

    spec = importlib.util.spec_from_file_location("_custom_loss", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "build_loss"):
        fn = mod.build_loss(config)
        logger.info("Loaded custom loss via build_loss() from %s", path)
        return fn
    elif hasattr(mod, "loss_fn"):
        logger.info("Loaded custom loss via loss_fn() from %s", path)
        return mod.loss_fn
    else:
        raise AttributeError(
            f"Custom loss file {path} must expose 'build_loss(config)' or 'loss_fn(y_pred, y_true, **kwargs)'"
        )
