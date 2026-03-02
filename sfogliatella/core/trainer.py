"""Training loop for JAX/equinox and XGBoost models."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from sfogliatella.core.data import (
    load_data, make_windows, chronological_split,
    time_series_folds, make_batches,
)
from sfogliatella.core.eval import compute_metrics
from sfogliatella.core.utils import make_run_id, nan_inf_check, ensure_dir
from sfogliatella.io.checkpoint import save_checkpoint, get_latest_checkpoint, load_checkpoint
from sfogliatella.io.metadata import save_run_config, save_metrics, save_loss_curves
from sfogliatella.losses.loader import load_loss_fn
from sfogliatella.models.base import TrainResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JIT-compiled training step (equinox)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _train_step(
    model: Any,
    opt_state: Any,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
) -> Tuple[Any, Any, jnp.ndarray]:
    def compute_loss(m):
        preds = jax.vmap(m)(x_batch)
        return loss_fn(preds, y_batch)

    loss, grads = eqx.filter_value_and_grad(compute_loss)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


@eqx.filter_jit
def _eval_step(
    model: Any,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable,
) -> jnp.ndarray:
    preds = jax.vmap(model)(x_batch)
    return loss_fn(preds, y_batch)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_model(
    config: Dict[str, Any],
    data_path: Optional[str] = None,
    data_array: Optional[np.ndarray] = None,
) -> TrainResult:
    """Train a model according to config.

    Either data_path (CSV/Parquet file) or data_array (numpy) must be provided.
    """
    from sfogliatella.registry.registry import _ensure_models_registered, build_model
    _ensure_models_registered()

    # ---- Parse config ----
    model_name  = config.get("model", "mlp")
    task        = config.get("task", "regression")
    lookback    = int(config.get("lookback", 32))
    horizon     = int(config.get("horizon", 1))
    target_col  = int(config.get("target_col", 0))
    feature_cols = config.get("feature_cols", None)
    val_ratio   = float(config.get("val_ratio", 0.1))
    test_ratio  = float(config.get("test_ratio", 0.1))
    n_folds     = int(config.get("folds", 1))
    batch_size  = int(config.get("batch_size", 32))
    epochs      = int(config.get("epochs", 10))
    lr          = float(config.get("lr", 1e-3))
    seed        = int(config.get("seed", 42))
    model_dir   = Path(config.get("model_dir", "outputs/models"))
    run_id      = make_run_id(config.get("run_id"))
    ckpt_every  = int(config.get("checkpoint_every", 1))  # epochs
    data_format = config.get("data_format", None)
    loss_name   = config.get("loss", None)
    loss_path   = config.get("loss_path", None)

    run_dir = model_dir / run_id
    ensure_dir(run_dir)

    logger.info("run_id=%s  model=%s  task=%s  lookback=%d  horizon=%d",
                run_id, model_name, task, lookback, horizon)

    # ---- Load data ----
    if data_array is not None:
        data = np.asarray(data_array, dtype=np.float32)
    elif data_path:
        data = load_data(data_path, data_format)
    else:
        raise ValueError("Either data_path or data_array must be provided.")

    n_cols = data.shape[1]
    if feature_cols is None:
        feature_cols = [c for c in range(n_cols) if c != target_col]
    input_dim = len(feature_cols)

    # Update config with derived dims
    config = {**config, "input_dim": input_dim, "run_id": run_id, "lookback": lookback, "horizon": horizon}

    logger.info("Data shape=%s  input_dim=%d  target_col=%d", data.shape, input_dim, target_col)

    # ---- Build windows ----
    X, y = make_windows(data, lookback, horizon, target_col, feature_cols)
    logger.info("Windows: X=%s y=%s", X.shape, y.shape)

    # ---- Save config ----
    save_run_config(config, run_dir)

    # ---- XGBoost path ----
    from sfogliatella.models.xgboost_model import XGBoostModel

    if model_name == "xgboost":
        return _train_xgboost(
            config, X, y, run_id, run_dir, model_dir, model_name,
            val_ratio, test_ratio, task,
        )

    # ---- Custom model path ----
    if model_name == "custom":
        return _train_custom(config, run_id, run_dir)

    # ---- JAX path ----
    # Build model
    rng = jax.random.PRNGKey(seed)
    model = build_model(model_name, config, rng_key=rng)

    # Loss function
    loss_fn = load_loss_fn(loss_name, loss_path, task, config)

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Resume from checkpoint?
    resume_from = config.get("resume_from", None)
    start_epoch = 0
    global_step = 0
    ckpt_dir = model_dir

    if resume_from:
        ckpt_path = Path(resume_from)
    else:
        ckpt_path = get_latest_checkpoint(ckpt_dir, run_id)

    if ckpt_path:
        logger.info("Resuming from checkpoint: %s", ckpt_path)
        model, opt_state, meta = load_checkpoint(model, ckpt_path, opt_state)
        start_epoch = int(meta.get("epoch", 0)) + 1
        global_step = int(meta.get("step", 0))
        if opt_state is None:
            opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Split
    if n_folds <= 1:
        (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te) = chronological_split(X, y, val_ratio, test_ratio)
        folds = [((X_tr, y_tr), (X_vl, y_vl))]
    else:
        folds = time_series_folds(X, y, n_folds, val_ratio)
        (_, _), (X_te, y_te) = chronological_split(X, y, val_ratio, test_ratio)

    all_train_losses: List[float] = []
    all_val_losses:   List[float] = []
    np_rng = np.random.default_rng(seed)

    best_val_loss = float("inf")
    stagnation_count = 0

    for fold_idx, ((X_tr, y_tr), (X_vl, y_vl)) in enumerate(folds):
        logger.info("Fold %d/%d  train=%d  val=%d", fold_idx+1, len(folds), len(X_tr), len(X_vl))

        X_tr_jnp = jnp.array(X_tr)
        y_tr_jnp = jnp.array(y_tr)
        X_vl_jnp = jnp.array(X_vl)
        y_vl_jnp = jnp.array(y_vl)

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            batches = make_batches(X_tr, y_tr, batch_size, shuffle=False)
            epoch_losses = []

            for X_b, y_b in batches:
                X_b_jnp = jnp.array(X_b)
                y_b_jnp = jnp.array(y_b)
                model, opt_state, loss = _train_step(model, opt_state, X_b_jnp, y_b_jnp, loss_fn, optimizer)
                step_loss = float(loss)
                epoch_losses.append(step_loss)
                global_step += 1

                if nan_inf_check(step_loss, global_step):
                    logger.error("Training aborted due to NaN/Inf at step %d", global_step)
                    return TrainResult(run_id=run_id, model_dir=str(run_dir), status="failed",
                                       train_losses=all_train_losses, val_losses=all_val_losses)

            train_loss = float(np.mean(epoch_losses))
            all_train_losses.append(train_loss)

            # Validation
            val_batches = make_batches(X_vl, y_vl, batch_size * 2)
            val_losses_ep = []
            for X_b, y_b in val_batches:
                vl = float(_eval_step(model, jnp.array(X_b), jnp.array(y_b), loss_fn))
                val_losses_ep.append(vl)
            val_loss = float(np.mean(val_losses_ep))
            all_val_losses.append(val_loss)

            elapsed = time.time() - epoch_start
            logger.info("Epoch %03d/%d  train=%.5f  val=%.5f  [%.1fs]",
                        epoch+1, epochs, train_loss, val_loss, elapsed)

            # Divergence / stagnation detection
            if len(all_val_losses) > 5:
                recent = all_val_losses[-5:]
                if all(v > all_val_losses[-6] for v in recent):
                    logger.warning("Val loss increasing for 5 epochs (possible divergence)")
                if all_val_losses[-1] > best_val_loss * 1.5:
                    logger.warning("Val loss %.4f is 50%% above best %.4f (loss explosion?)",
                                   all_val_losses[-1], best_val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count >= 20:
                    logger.warning("Val loss stagnant for 20 epochs")

            # Checkpoint
            if (epoch + 1) % ckpt_every == 0:
                save_checkpoint(
                    model, opt_state, global_step, epoch, config,
                    ckpt_dir, run_id, rng=None,
                )

    # Final checkpoint
    save_checkpoint(model, opt_state, global_step, epochs-1, config, ckpt_dir, run_id)

    # Evaluate on test set
    metrics = {}
    if len(X_te) > 0:
        preds = _predict_jax(model, X_te, batch_size)
        metrics = compute_metrics(preds, y_te, task)
        logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    save_metrics({"train_losses": all_train_losses, "val_losses": all_val_losses, **metrics}, run_dir)
    save_loss_curves(all_train_losses, all_val_losses, run_dir)

    return TrainResult(
        run_id=run_id,
        model_dir=str(run_dir),
        train_losses=all_train_losses,
        val_losses=all_val_losses,
        metrics=metrics,
        config=config,
        status="done",
    )


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def _train_xgboost(config, X, y, run_id, run_dir, model_dir, model_name, val_ratio, test_ratio, task):
    from sfogliatella.models.xgboost_model import XGBoostModel
    (X_tr, y_tr), (X_vl, y_vl), (X_te, y_te) = chronological_split(X, y, val_ratio, test_ratio)

    model = XGBoostModel(config)
    model.fit(X_tr, y_tr, X_vl, y_vl)

    # Save
    model.save(str(run_dir / "model"))

    train_losses = model.get_train_losses()
    val_losses   = model.get_val_losses()

    metrics = {}
    if len(X_te) > 0:
        preds = model.predict(X_te)
        metrics = compute_metrics(preds, y_te, task)
        logger.info("XGBoost test metrics: %s", metrics)

    save_metrics({"train_losses": train_losses, "val_losses": val_losses, **metrics}, run_dir)
    save_loss_curves(train_losses, val_losses, run_dir)

    return TrainResult(
        run_id=run_id,
        model_dir=str(run_dir),
        train_losses=train_losses,
        val_losses=val_losses,
        metrics=metrics,
        config=config,
        status="done",
    )


# ---------------------------------------------------------------------------
# Custom model training (subprocess delegation)
# ---------------------------------------------------------------------------

def _train_custom(config: Dict[str, Any], run_id: str, run_dir: Path) -> TrainResult:
    """Delegate training to a user-provided runner via subprocess.

    Requires config keys:
      custom_model_root : path to user model code directory
      custom_model_call : shell command to invoke the runner

    The runner is called as:
      <custom_model_call> --config <json_tempfile> --mode train

    The config JSON includes all standard keys plus model_dir / run_id.
    The runner must write artifacts into model_dir/run_id/ following the
    standard Sfogliatella conventions.
    """
    import subprocess
    import tempfile
    import json as _json

    custom_root = config.get("custom_model_root")
    custom_call = config.get("custom_model_call")

    if not custom_call:
        raise ValueError("--custom_model_call is required when --model custom")

    # Write config to a temp JSON file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(run_dir)
    ) as tf:
        _json.dump({**config, "run_id": run_id, "model_dir": str(run_dir.parent)}, tf)
        config_path = tf.name

    cmd = f"{custom_call} --config {config_path} --mode train"
    env = None
    if custom_root:
        import os
        env = {**os.environ, "PYTHONPATH": custom_root + ":" + os.environ.get("PYTHONPATH", "")}

    logger.info("Custom model train cmd: %s", cmd)
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info("Custom runner output:\n%s", result.stdout[-2000:])
    except subprocess.CalledProcessError as e:
        logger.error("Custom runner failed (exit %d):\n%s", e.returncode, e.output[-2000:])
        return TrainResult(run_id=run_id, model_dir=str(run_dir), status="failed")

    return TrainResult(run_id=run_id, model_dir=str(run_dir), status="done")


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_jax(model: Any, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Run inference on a numpy array using a JAX/equinox model."""
    all_preds = []
    n = len(X)
    for start in range(0, n, batch_size):
        xb = jnp.array(X[start : start + batch_size])
        preds = jax.vmap(model)(xb)
        all_preds.append(np.array(preds))
    return np.concatenate(all_preds, axis=0)


def _predict_custom(
    config: Dict[str, Any],
    X: np.ndarray,
    model_dir: Path,
    run_id: str,
) -> np.ndarray:
    """Delegate prediction to a user-provided custom runner.

    Saves X to a temp numpy file, calls the runner with --mode predict,
    and reads the output predictions from the temp result file.
    """
    import subprocess
    import tempfile
    import json as _json

    custom_call = config.get("custom_model_call")
    custom_root = config.get("custom_model_root")

    if not custom_call:
        raise ValueError("--custom_model_call is required for custom model prediction")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        x_path = str(tmpdir_path / "X.npy")
        pred_path = str(tmpdir_path / "preds.npy")
        cfg_path = str(tmpdir_path / "config.json")

        np.save(x_path, X)
        with open(cfg_path, "w") as f:
            _json.dump({
                **config,
                "run_id": run_id,
                "model_dir": str(model_dir),
                "x_path": x_path,
                "pred_path": pred_path,
            }, f)

        cmd = f"{custom_call} --config {cfg_path} --mode predict"
        env = None
        if custom_root:
            import os
            env = {**os.environ, "PYTHONPATH": custom_root + ":" + os.environ.get("PYTHONPATH", "")}

        logger.info("Custom model predict cmd: %s", cmd)
        try:
            result = subprocess.run(
                cmd, shell=True, check=True, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            logger.info("Custom runner output:\n%s", result.stdout[-2000:])
        except subprocess.CalledProcessError as e:
            logger.error("Custom runner failed (exit %d):\n%s", e.returncode, e.output[-2000:])
            raise RuntimeError("Custom model prediction failed") from e

        if not Path(pred_path).exists():
            raise FileNotFoundError(
                f"Custom runner did not write predictions to expected path: {pred_path}"
            )
        return np.load(pred_path)


def predict_model(
    config: Dict[str, Any],
    model_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    data_array: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> "PredictResult":
    """Run batch prediction (walk-forward, no leakage)."""
    from sfogliatella.registry.registry import _ensure_models_registered, build_model
    from sfogliatella.io.checkpoint import get_latest_checkpoint, load_checkpoint
    from sfogliatella.io.metadata import save_predictions
    from sfogliatella.models.base import PredictResult
    _ensure_models_registered()

    run_id    = make_run_id(config.get("run_id"))
    model_name = config.get("model", "mlp")
    task      = config.get("task", "regression")
    lookback  = int(config.get("lookback", 32))
    horizon   = int(config.get("horizon", 1))
    target_col = int(config.get("target_col", 0))
    feature_cols = config.get("feature_cols", None)
    batch_size = int(config.get("batch_size", 64))
    seed      = int(config.get("seed", 42))
    data_format = config.get("data_format", None)

    _model_dir = Path(model_dir or config.get("model_dir", "outputs/models"))
    _run_id = config.get("run_id", run_id)

    # Load data
    if data_array is not None:
        data = np.asarray(data_array, dtype=np.float32)
    elif data_path:
        data = load_data(data_path, data_format)
    else:
        raise ValueError("Either data_path or data_array must be provided.")

    n_cols = data.shape[1]
    if feature_cols is None:
        feature_cols = [c for c in range(n_cols) if c != target_col]
    input_dim = len(feature_cols)
    config = {**config, "input_dim": input_dim}

    X, y = make_windows(data, lookback, horizon, target_col, feature_cols)

    # Load model
    from sfogliatella.models.xgboost_model import XGBoostModel
    if model_name == "xgboost":
        model = XGBoostModel(config)
        model.load(str(_model_dir / _run_id / "model"))
        preds = model.predict(X)
    elif model_name in ("timesfm", "timesfm_ft"):
        model = build_model(model_name, config, rng_key=None)
        model.load(str(_model_dir / _run_id))
        preds = model.predict(X)
    elif model_name == "custom":
        preds = _predict_custom(config, X, _model_dir, _run_id)
    else:
        rng = jax.random.PRNGKey(seed)
        model = build_model(model_name, config, rng_key=rng)
        ckpt = get_latest_checkpoint(_model_dir, _run_id)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {_model_dir}/{_run_id}")
        model, _, _ = load_checkpoint(model, ckpt)
        preds = _predict_jax(model, X, batch_size)

    indices = list(range(lookback, lookback + len(preds)))
    metrics = compute_metrics(preds, y, task)

    if output_path:
        save_predictions(preds, indices, np.asarray(y), Path(output_path), task)

    return PredictResult(
        run_id=_run_id,
        predictions=preds,
        output_path=output_path,
        metrics=metrics,
        status="done",
    )
