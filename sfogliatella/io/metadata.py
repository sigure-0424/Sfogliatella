"""Saving/loading run metadata and artifacts."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sfogliatella.core.utils import get_version_info, save_json

logger = logging.getLogger(__name__)


def save_run_config(config: Dict[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json({**config, "_versions": get_version_info()}, run_dir / "config.json")


def save_metrics(metrics: Dict[str, Any], run_dir: Path) -> None:
    save_json(metrics, run_dir / "metrics.json")


def save_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    run_dir: Path,
) -> None:
    """Save loss curves as CSV and PNG."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Numeric (CSV)
    csv_path = run_dir / "loss_curves.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])
        n = max(len(train_losses), len(val_losses))
        for i in range(n):
            tr = train_losses[i] if i < len(train_losses) else ""
            vl = val_losses[i] if i < len(val_losses) else ""
            writer.writerow([i, tr, vl])
    logger.debug("Loss curves (CSV) saved: %s", csv_path)

    # Image (PNG) — optional, skip if matplotlib not available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        if train_losses:
            plt.plot(train_losses, label="train", linewidth=1.5)
        if val_losses:
            plt.plot(val_losses, label="val", linewidth=1.5)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.tight_layout()
        png_path = run_dir / "loss_curves.png"
        plt.savefig(png_path, dpi=100)
        plt.close()
        logger.debug("Loss curves (PNG) saved: %s", png_path)
    except Exception as e:
        logger.debug("Could not save loss curve PNG: %s", e)


def save_predictions(
    predictions: np.ndarray,
    indices: Optional[List[int]],
    y_true: Optional[np.ndarray],
    output_path: Path,
    task: str = "regression",
) -> None:
    """Save predictions to CSV or Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preds = np.asarray(predictions)
    n = len(preds)
    idx = indices if indices is not None else list(range(n))

    if output_path.suffix == ".parquet":
        _save_parquet_predictions(preds, idx, y_true, output_path, task)
    else:
        _save_csv_predictions(preds, idx, y_true, output_path, task)


def _save_csv_predictions(preds, idx, y_true, path, task):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["row_index"]
        if task == "regression":
            header += ["y_pred"]
        elif task == "classification":
            if preds.ndim > 1:
                header += [f"logit_{i}" for i in range(preds.shape[1])]
                header += [f"proba_{i}" for i in range(preds.shape[1])]
            else:
                header += ["logit", "proba"]
        elif task == "ranking":
            header += ["score"]
        elif task == "anomaly_score":
            header += ["anomaly_score"]
        elif task == "embedding":
            header += [f"emb_{i}" for i in range(preds.shape[-1] if preds.ndim > 1 else 1)]
        else:
            header += ["y_pred"]

        if y_true is not None:
            header += ["y_true"]

        writer.writerow(header)

        for i, (row_idx, pred) in enumerate(zip(idx, preds)):
            row = [row_idx]
            if np.isscalar(pred) or (hasattr(pred, 'ndim') and pred.ndim == 0):
                row.append(float(pred))
            else:
                row.extend([float(v) for v in pred.ravel()])
                if task == "classification":
                    # Add probabilities
                    import jax
                    import jax.numpy as jnp
                    lgt = jnp.array(pred)
                    if lgt.ndim == 1 and len(lgt) == 1:
                        proba = float(jax.nn.sigmoid(lgt[0]))
                        row.append(proba)
                    elif lgt.ndim == 1:
                        proba = jax.nn.softmax(lgt)
                        row.extend([float(v) for v in proba])
            if y_true is not None:
                yt = y_true[i]
                if np.isscalar(yt) or (hasattr(yt, 'ndim') and yt.ndim == 0):
                    row.append(float(yt))
                else:
                    row.extend([float(v) for v in np.array(yt).ravel()])
            writer.writerow(row)

    logger.info("Predictions saved: %s (%d rows)", path, len(preds))


def _save_parquet_predictions(preds, idx, y_true, path, task):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Parquet output")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet output")

    data = {"row_index": idx}
    preds_arr = np.asarray(preds)
    if preds_arr.ndim == 1:
        data["y_pred"] = preds_arr
    else:
        for j in range(preds_arr.shape[1]):
            data[f"pred_{j}"] = preds_arr[:, j]

    if y_true is not None:
        yt = np.asarray(y_true)
        if yt.ndim == 1:
            data["y_true"] = yt
        else:
            for j in range(yt.shape[1]):
                data[f"y_true_{j}"] = yt[:, j]

    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    logger.info("Predictions saved (parquet): %s", path)
