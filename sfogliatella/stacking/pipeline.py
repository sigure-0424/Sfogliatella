"""Multi-stage stacking pipeline.

Pipeline config (JSON/YAML) structure:
  stages:
    - name: "stage1"
      model: "mlp"
      model_dir: "outputs/models"
      run_id: "20260301_100000"
      inputs: ["features"]          # what to consume: "features" or name of previous stage
      output_key: "s1_pred"
      model_params: {}
    - name: "meta"
      model: "mlp"
      model_dir: "outputs/meta_model"
      run_id: "20260301_110000"
      inputs: ["features", "s1_pred"]
      output_key: "final"
      model_params: {}
  transform_path: null              # optional per-pipeline transform
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StackResult:
    status: str
    predictions: Optional[np.ndarray] = None
    stage_outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "stage_outputs": {k: v.shape for k, v in self.stage_outputs.items()},
            "output_path": self.output_path,
        }


def run_pipeline(
    pipeline_config: Dict[str, Any],
    data_path: Optional[str] = None,
    data_array: Optional[np.ndarray] = None,
    base_config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> StackResult:
    """Execute a multi-stage inference pipeline.

    Each stage runs predict_model with the specified model, then feeds outputs
    to subsequent stages as additional features.
    """
    from sfogliatella.core.data import load_data, make_windows, resolve_feature_cols
    from sfogliatella.core.trainer import _predict_jax, predict_model
    from sfogliatella.registry.registry import _ensure_models_registered, build_model
    from sfogliatella.io.checkpoint import get_latest_checkpoint, load_checkpoint
    import jax
    _ensure_models_registered()

    bc = base_config or {}
    stages = pipeline_config.get("stages", [])
    transform_path = pipeline_config.get("transform_path", None)

    if not stages:
        raise ValueError("Pipeline config must have at least one stage.")

    # Load base data
    if data_array is not None:
        data = np.asarray(data_array, dtype=np.float32)
    elif data_path:
        data = load_data(data_path, bc.get("data_format"))
    else:
        raise ValueError("data_path or data_array required for pipeline.")

    lookback = int(bc.get("lookback", 32))
    horizon  = int(bc.get("horizon", 1))
    target_col = int(bc.get("target_col", 0))
    n_cols = data.shape[1]
    feature_cols = bc.get("feature_cols", [c for c in range(n_cols) if c != target_col])

    X_base, y_base = make_windows(data, lookback, horizon, target_col, feature_cols)
    stage_outputs: Dict[str, np.ndarray] = {"features": X_base}

    for stage in stages:
        stage_name = stage.get("name", "unknown")
        model_name = stage.get("model", bc.get("model", "mlp"))
        stage_inputs = stage.get("inputs", ["features"])
        output_key = stage.get("output_key", stage_name + "_pred")
        stage_run_id = stage.get("run_id", bc.get("run_id"))
        stage_model_dir = Path(stage.get("model_dir", bc.get("model_dir", "outputs/models")))
        seed = int(stage.get("seed", bc.get("seed", 42)))

        logger.info("Running stage: %s (model=%s)", stage_name, model_name)

        # Build input for this stage
        parts = []
        for inp_key in stage_inputs:
            if inp_key not in stage_outputs:
                raise KeyError(f"Stage {stage_name}: input '{inp_key}' not found in previous outputs.")
            arr = stage_outputs[inp_key]
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim == 3:
                # Flatten last dim: (N, lookback, feat) -> pass through
                pass
            parts.append(arr)

        # If all inputs are 3D windows, concatenate features
        X_stage = _combine_stage_inputs(parts)

        stage_config = {
            **bc,
            **stage.get("model_params", {}),
            "model": model_name,
            "lookback": X_stage.shape[1] if X_stage.ndim == 3 else lookback,
            "input_dim": X_stage.shape[2] if X_stage.ndim == 3 else X_stage.shape[1],
            "task": stage.get("task", bc.get("task", "regression")),
        }

        from sfogliatella.models.xgboost_model import XGBoostModel
        if model_name == "xgboost":
            model = XGBoostModel(stage_config)
            model.load(str(stage_model_dir / stage_run_id / "model"))
            preds = model.predict(X_stage)
        else:
            rng = jax.random.PRNGKey(seed)
            model = build_model(model_name, stage_config, rng_key=rng)
            ckpt = get_latest_checkpoint(stage_model_dir, stage_run_id)
            if ckpt is None:
                raise FileNotFoundError(f"No checkpoint for stage {stage_name}: {stage_model_dir}/{stage_run_id}")
            model, _, _ = load_checkpoint(model, ckpt)
            preds = _predict_jax(model, X_stage, int(bc.get("batch_size", 64)))

        stage_outputs[output_key] = preds
        logger.info("Stage %s output: shape=%s", stage_name, preds.shape)

    # Optional transform
    if transform_path:
        transform = _load_transform(transform_path)
        final_preds = transform(stage_outputs.get(stages[-1].get("output_key", "features"), None), stage_outputs)
    else:
        last_key = stages[-1].get("output_key", "features")
        final_preds = stage_outputs.get(last_key)

    # Save output
    if output_path and final_preds is not None:
        from sfogliatella.io.metadata import save_predictions
        save_predictions(final_preds, None, None, Path(output_path), bc.get("task", "regression"))

    return StackResult(
        status="done",
        predictions=final_preds,
        stage_outputs=stage_outputs,
        output_path=output_path,
    )


def _combine_stage_inputs(parts: List[np.ndarray]) -> np.ndarray:
    """Combine input arrays from multiple stages."""
    if len(parts) == 1:
        return parts[0]
    # If all 3D: concatenate along feature axis
    if all(p.ndim == 3 for p in parts):
        return np.concatenate(parts, axis=2)
    # Mix of 2D and 3D: expand 2D and tile along time
    flat_parts = []
    n = parts[0].shape[0]
    time_steps = next((p.shape[1] for p in parts if p.ndim == 3), 1)
    for p in parts:
        if p.ndim == 3:
            flat_parts.append(p.reshape(n, time_steps, -1))
        elif p.ndim == 2:
            # (n, feat) -> (n, time, feat) broadcast
            flat_parts.append(np.broadcast_to(p[:, None, :], (n, time_steps, p.shape[1])).copy())
        elif p.ndim == 1:
            flat_parts.append(np.broadcast_to(p[:, None, None], (n, time_steps, 1)).copy())
    return np.concatenate(flat_parts, axis=2)


def _load_transform(transform_path: str):
    import importlib.util
    path = Path(transform_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")
    spec = importlib.util.spec_from_file_location("_stage_transform", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "transform"):
        return mod.transform
    raise AttributeError(f"Transform file {path} must expose 'transform(preds, context)'")
