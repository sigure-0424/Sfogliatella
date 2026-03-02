"""XGBoost model wrapper (no PyTorch/TF; uses xgboost package directly)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from sfogliatella.registry.registry import register

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost wrapper that follows the same train/predict interface as JAX models.

    Unlike equinox models, XGBoost is not a pytree and uses a different save/load mechanism.
    The trainer handles this via isinstance checks.
    """

    def __init__(self, config: Dict[str, Any]):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required. Install with: pip install xgboost") from e

        self.config = config
        self.task = config.get("task", "regression")
        self.num_classes = int(config.get("num_classes", 2))
        self.booster = None

        params = self._build_params(config)
        self.params = params
        self.n_estimators = int(config.get("n_estimators", 300))

    def _build_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        task = config.get("task", "regression")
        if task == "regression":
            objective = "reg:squarederror"
        elif task == "classification":
            nc = int(config.get("num_classes", 2))
            objective = "binary:logistic" if nc == 2 else "multi:softmax"
        else:
            objective = "reg:squarederror"

        params = {
            "objective":        objective,
            "max_depth":        int(config.get("max_depth", 6)),
            "learning_rate":    float(config.get("learning_rate", config.get("lr", 0.1))),
            "subsample":        float(config.get("subsample", 0.8)),
            "colsample_bytree": float(config.get("colsample_bytree", 0.8)),
            "seed":             int(config.get("seed", 42)),
            "verbosity":        0,
        }
        if task == "classification" and self.num_classes > 2:
            params["num_class"] = self.num_classes

        return params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callbacks=None,
    ) -> None:
        import xgboost as xgb

        # Flatten sequence dim if 3D
        if X_train.ndim == 3:
            X_train = X_train.reshape(len(X_train), -1)
        if X_val is not None and X_val.ndim == 3:
            X_val = X_val.reshape(len(X_val), -1)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        evals_result = {}
        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=False,
        )
        self._evals_result = evals_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        import xgboost as xgb
        if X.ndim == 3:
            X = X.reshape(len(X), -1)
        return self.booster.predict(xgb.DMatrix(X))

    def save(self, path: str) -> None:
        if self.booster is not None:
            self.booster.save_model(str(path) + ".xgb")
            logger.debug("XGBoost model saved to %s.xgb", path)

    def load(self, path: str) -> None:
        import xgboost as xgb
        self.booster = xgb.Booster()
        self.booster.load_model(str(path) + ".xgb")
        logger.debug("XGBoost model loaded from %s.xgb", path)

    def get_train_losses(self) -> list:
        """Return train losses from last training run."""
        if not hasattr(self, "_evals_result"):
            return []
        result = self._evals_result.get("train", {})
        metric_vals = next(iter(result.values()), []) if result else []
        return list(metric_vals)

    def get_val_losses(self) -> list:
        if not hasattr(self, "_evals_result"):
            return []
        result = self._evals_result.get("val", {})
        metric_vals = next(iter(result.values()), []) if result else []
        return list(metric_vals)


@register("xgboost")
def build_xgboost(config: Dict[str, Any], rng_key=None, sample_x=None):
    return XGBoostModel(config)
