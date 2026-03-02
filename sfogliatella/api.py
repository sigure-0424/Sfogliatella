"""Public Python API for Sfogliatella."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def load_model(model_dir_or_pack: Union[str, Path]):
    """Load a trained model from a directory or .sfog archive."""
    path = Path(model_dir_or_pack)
    if path.suffix == ".sfog":
        from sfogliatella.io.pack import unpack_model
        return unpack_model(path)
    return path  # return dir path; actual loading is deferred to predict()


def train(
    config: Dict[str, Any],
    data: Union[str, np.ndarray, None] = None,
    data_path: Optional[str] = None,
) -> "TrainResult":
    """Train a model.

    Args:
        config: Training configuration dict.
        data: numpy array (rows x cols) or file path string.
        data_path: Explicit file path (overrides data if both given).
    Returns:
        TrainResult
    """
    from sfogliatella.core.trainer import train_model

    if isinstance(data, str):
        data_path = data
        data = None

    return train_model(config, data_path=data_path, data_array=data)


def predict(
    config_or_model: Union[Dict[str, Any], Path],
    data: Union[str, np.ndarray, None] = None,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> "PredictResult":
    """Run batch strict (no-leakage) prediction.

    Args:
        config_or_model: Config dict or path to model_dir.
        data: numpy array or file path.
        data_path: Explicit file path.
        output_path: Where to save predictions CSV.
    Returns:
        PredictResult
    """
    from sfogliatella.core.trainer import predict_model

    if isinstance(config_or_model, (str, Path)):
        config = {"model_dir": str(config_or_model)}
    else:
        config = config_or_model

    if isinstance(data, str):
        data_path = data
        data = None

    return predict_model(config, data_path=data_path, data_array=data, output_path=output_path)


def live_predict(
    config_or_model: Union[Dict[str, Any], Path],
    csv_path: str,
    output_path: Optional[str] = None,
    poll_interval: float = 1.0,
) -> None:
    """Start live prediction loop (blocks until interrupted).

    Args:
        config_or_model: Config dict or path to model_dir.
        csv_path: Path to the CSV file that grows by appending new rows.
        output_path: Where to append prediction results.
        poll_interval: Seconds between file checks.
    """
    from sfogliatella.cli.live_cli import watch_and_predict

    if isinstance(config_or_model, (str, Path)):
        config = {"model_dir": str(config_or_model)}
    else:
        config = config_or_model

    watch_and_predict(config, csv_path=csv_path, output_path=output_path, poll_interval=poll_interval)


def hpo(
    config: Dict[str, Any],
    baseline_measurements=None,
    constraints: Optional[Dict[str, Any]] = None,
    return_why: bool = True,
) -> "HPOResult":
    """Compute HPO prior / baseline hyperparameter configuration.

    Args:
        config: Must include 'model', 'lookback', 'horizon', 'input_dim',
                'num_train_samples', etc.
        baseline_measurements: List of (N, loss) tuples for scaling law fit.
        constraints: Hardware/time constraints dict.
        return_why: Include reasoning payload.
    Returns:
        HPOResult
    """
    from sfogliatella.hpo.solver import SolverInputs, HardwareConfig, solve_hpo_prior, HPOResult

    hw_cfg = constraints or {}
    hw = HardwareConfig(
        gpu_vram_gb=float(hw_cfg.get("gpu_vram_gb", 24.0)),
        gpu_peak_tflops=float(hw_cfg.get("gpu_peak_tflops", 30.0)),
        gpu_utilization=float(hw_cfg.get("gpu_utilization", 0.4)),
        optimizer=hw_cfg.get("optimizer", "adamw"),
    )

    inp = SolverInputs(
        baseline_measurements=baseline_measurements,
        target_loss=float(config.get("target_loss", 0.1)),
        noise_floor=float(config.get("noise_floor", 0.0)),
        num_train_samples=int(config.get("num_train_samples", 10000)),
        num_epochs=int(config.get("epochs", 10)),
        batch_size=int(config.get("batch_size", 32)),
        lookback=int(config.get("lookback", 32)),
        horizon=int(config.get("horizon", 1)),
        input_dim=int(config.get("input_dim", 1)),
        output_dim=int(config.get("output_dim", 1)),
        max_training_hours=float(config.get("max_training_hours", 24.0)),
        lambda_data_cap=float(config.get("lambda_data_cap", 20.0)),
        hardware=hw,
        task=config.get("task", "regression"),
    )

    model_name = config.get("model", "mlp")
    raw = solve_hpo_prior(model_name, inp, return_why=return_why)
    return HPOResult(
        model=model_name,
        status=raw.get("status", "unknown"),
        config=raw.get("config", {}),
        estimates=raw.get("estimates", {}),
        search_space=raw.get("hpo_search_space", {}),
        why=raw.get("why"),
        raw=raw,
    )


def stack(
    pipeline_config: Dict[str, Any],
    data: Union[str, np.ndarray, None] = None,
    data_path: Optional[str] = None,
    base_config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> "StackResult":
    """Run a multi-stage stacking pipeline.

    Args:
        pipeline_config: Pipeline definition dict with 'stages'.
        data: numpy array or file path.
        data_path: Explicit file path.
        base_config: Common config (lookback, horizon, etc.).
        output_path: Output path for final predictions.
    Returns:
        StackResult
    """
    from sfogliatella.stacking.pipeline import run_pipeline

    if isinstance(data, str):
        data_path = data
        data = None

    return run_pipeline(
        pipeline_config,
        data_path=data_path,
        data_array=data,
        base_config=base_config,
        output_path=output_path,
    )
