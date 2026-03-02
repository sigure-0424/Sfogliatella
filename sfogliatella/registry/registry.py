"""Model registry: name -> builder function."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a model builder."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_model_builder(name: str) -> Callable:
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_models() -> list:
    return sorted(_REGISTRY.keys())


def build_model(name: str, config: Dict[str, Any], rng_key=None, sample_x=None):
    """Build and return a model instance (or None if deferred init)."""
    builder = get_model_builder(name)
    return builder(config=config, rng_key=rng_key, sample_x=sample_x)


# ---------------------------------------------------------------------------
# Trigger registration by importing model modules
# ---------------------------------------------------------------------------

def _ensure_models_registered():
    import sfogliatella.models.mlp          # noqa: F401
    import sfogliatella.models.lstm         # noqa: F401
    import sfogliatella.models.rnn          # noqa: F401
    import sfogliatella.models.transformer  # noqa: F401
    import sfogliatella.models.cnn          # noqa: F401
    import sfogliatella.models.xgboost_model  # noqa: F401
    import sfogliatella.models.timesfm_model  # noqa: F401
