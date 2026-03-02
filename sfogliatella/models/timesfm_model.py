"""TimesFM model wrapper (inference-only + fine-tunable stub).

TimesFM (google-research/timesfm) is an optional dependency.
If not installed, importing this module succeeds but building the model raises
an ImportError with installation instructions.

Fine-tuning variant (timesfm_ft) wraps the same class with a learned linear
adapter head on top of frozen TimesFM embeddings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from sfogliatella.registry.registry import register

logger = logging.getLogger(__name__)

_TIMESFM_INSTALL_MSG = (
    "TimesFM is not installed. Install it with:\n"
    "  pip install timesfm\n"
    "See: https://github.com/google-research/timesfm"
)


# ---------------------------------------------------------------------------
# TimesFM wrapper (inference-only)
# ---------------------------------------------------------------------------

class TimesFMWrapper:
    """Wraps the google-research TimesFM model for inference.

    The model is treated as a frozen foundation model. At predict time it
    returns point forecasts from the TimesFM API.

    For fine-tuning (timesfm_ft) a small equinox adapter head is appended
    to the TimesFM embedding output and trained with the standard trainer.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback = int(config.get("lookback", 512))
        self.horizon = int(config.get("horizon", 1))
        self.task = config.get("task", "regression")
        self._tfm = None  # lazy-loaded

    def _load(self):
        if self._tfm is not None:
            return
        try:
            import timesfm  # noqa: F401
        except ImportError as e:
            raise ImportError(_TIMESFM_INSTALL_MSG) from e

        context_len = self.config.get("timesfm_context_len", max(self.lookback, 512))
        horizon_len = self.horizon

        try:
            import timesfm
            self._tfm = timesfm.TimesFm(
                context_len=context_len,
                horizon_len=horizon_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend="cpu",
            )
            self._tfm.load_from_checkpoint(
                repo_id=self.config.get(
                    "timesfm_checkpoint", "google/timesfm-1.0-200m"
                )
            )
            logger.info("TimesFM loaded (context=%d, horizon=%d)", context_len, horizon_len)
        except Exception as e:
            raise RuntimeError(f"Failed to load TimesFM checkpoint: {e}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference. X: (N, lookback, n_feat); returns (N, horizon)."""
        self._load()
        # TimesFM expects (batch, context) univariate series.
        # Use only the first feature dimension (or mean across features).
        n_feat = X.shape[2] if X.ndim == 3 else 1
        if X.ndim == 3:
            series = X[:, :, 0]  # (N, lookback) — first feature
        else:
            series = X

        freq = [0] * len(series)  # 0 = default frequency
        point_forecast, _ = self._tfm.forecast(series.tolist(), freq=freq)
        preds = np.array(point_forecast)  # (N, horizon)
        return preds

    def save(self, path: str) -> None:
        """Save adapter weights (stub: saves config only for frozen model)."""
        import json, pathlib
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "timesfm_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info("TimesFM config saved to %s", p)

    def load(self, path: str) -> None:
        """Load config (weights are loaded lazily from HF checkpoint)."""
        import json, pathlib
        cfg_file = pathlib.Path(path) / "timesfm_config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                saved = json.load(f)
            self.config.update(saved)
        self._load()


# ---------------------------------------------------------------------------
# Fine-tuning variant: TimesFM + trainable linear adapter (equinox)
# ---------------------------------------------------------------------------

class TimesFMFTWrapper(TimesFMWrapper):
    """TimesFM with a trainable linear adapter head (fine-tuning variant).

    The frozen TimesFM produces embeddings; the adapter projects them to the
    target output dimension. The adapter is an equinox Linear layer trained
    by the standard Sfogliatella trainer.

    NOTE: Because the TimesFM backbone is not JAX-native, gradient flow
    through the backbone is not supported. Only the adapter head is trained.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._adapter = None
        self._adapter_opt_state = None

    def _build_adapter(self, embedding_dim: int, out_dim: int, key):
        import equinox as eqx
        self._adapter = eqx.nn.Linear(embedding_dim, out_dim, key=key)
        logger.info("TimesFM-FT adapter: %d -> %d", embedding_dim, out_dim)

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract TimesFM embeddings for the input windows."""
        self._load()
        if X.ndim == 3:
            series = X[:, :, 0]
        else:
            series = X
        freq = [0] * len(series)
        _, embeddings = self._tfm.forecast(series.tolist(), freq=freq)
        return np.array(embeddings)


# ---------------------------------------------------------------------------
# Registry builders
# ---------------------------------------------------------------------------

@register("timesfm")
def build_timesfm(config: Dict[str, Any], rng_key=None, sample_x=None):
    """Build a TimesFM inference wrapper."""
    logger.info("Building TimesFM model (inference-only, optional dependency)")
    return TimesFMWrapper(config)


@register("timesfm_ft")
def build_timesfm_ft(config: Dict[str, Any], rng_key=None, sample_x=None):
    """Build a TimesFM fine-tuning wrapper."""
    logger.info("Building TimesFM-FT model (adapter head, optional dependency)")
    return TimesFMFTWrapper(config)
