"""Live forecasting CLI: watches a CSV file for updates and predicts on each."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def watch_and_predict(
    config: Dict[str, Any],
    csv_path: str,
    output_path: Optional[str] = None,
    poll_interval: float = 1.0,
) -> None:
    """Monitor a CSV file for appended rows and predict on each update.

    The CSV grows by appending new rows at the bottom.
    """
    from sfogliatella.core.data import load_data, make_windows, resolve_feature_cols
    from sfogliatella.core.trainer import _predict_jax
    from sfogliatella.registry.registry import _ensure_models_registered, build_model
    from sfogliatella.io.checkpoint import get_latest_checkpoint, load_checkpoint
    from sfogliatella.io.metadata import save_predictions
    import jax
    import numpy as np
    _ensure_models_registered()

    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"Live CSV not found: {csv_path}")

    model_name   = config.get("model", "mlp")
    lookback     = int(config.get("lookback", 32))
    horizon      = int(config.get("horizon", 1))
    target_col   = int(config.get("target_col", 0))
    feature_cols = config.get("feature_cols", None)
    batch_size   = int(config.get("batch_size", 64))
    model_dir    = Path(config.get("model_dir", "outputs/models"))
    run_id       = config.get("run_id", "")
    seed         = int(config.get("seed", 42))
    _output_path = Path(output_path) if output_path else csv_path_obj.parent / "predictions_live.csv"

    # Load model once
    from sfogliatella.models.xgboost_model import XGBoostModel
    if model_name == "xgboost":
        model = XGBoostModel(config)
        model.load(str(model_dir / run_id / "model"))
        is_xgb = True
    else:
        rng = jax.random.PRNGKey(seed)
        model = build_model(model_name, config, rng_key=rng)
        ckpt = get_latest_checkpoint(model_dir, run_id)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}/{run_id}")
        model, _, _ = load_checkpoint(model, ckpt)
        is_xgb = False

    logger.info("Live mode started. Watching: %s", csv_path)
    logger.info("Output: %s", _output_path)

    last_mtime = 0.0
    last_row_count = 0

    while True:
        try:
            mtime = csv_path_obj.stat().st_mtime
        except FileNotFoundError:
            logger.warning("CSV not found (deleted?), retrying...")
            time.sleep(poll_interval)
            continue

        if mtime == last_mtime:
            time.sleep(poll_interval)
            continue

        # File updated — try to load
        for retry in range(3):
            try:
                data = load_data(csv_path, "csv")
                break
            except Exception as e:
                if retry == 2:
                    logger.warning("Could not read CSV after 3 retries: %s", e)
                    time.sleep(0.5)
                    continue
                time.sleep(0.5)
        else:
            last_mtime = mtime
            continue

        n_rows = len(data)
        if n_rows <= last_row_count:
            last_mtime = mtime
            time.sleep(poll_interval)
            continue

        # New rows available
        n_cols = data.shape[1]
        if feature_cols is None:
            _fcols = [c for c in range(n_cols) if c != target_col]
        else:
            _fcols = feature_cols

        # We need at least lookback + horizon rows
        if n_rows < lookback + horizon:
            logger.debug("Not enough rows yet (%d < %d)", n_rows, lookback + horizon)
            last_mtime = mtime
            last_row_count = n_rows
            time.sleep(poll_interval)
            continue

        # Use the last lookback rows for prediction
        window = data[-lookback:, _fcols]     # (lookback, n_feat)
        x = window[None]                       # (1, lookback, n_feat)

        try:
            if is_xgb:
                pred = model.predict(x)
            else:
                import jax.numpy as jnp
                pred = _predict_jax(model, x, batch_size=1)
            pred_val = float(pred.ravel()[0])
        except Exception as e:
            logger.warning("Prediction failed: %s", e)
            last_mtime = mtime
            last_row_count = n_rows
            time.sleep(poll_interval)
            continue

        # Append to output CSV
        _write_live_prediction(_output_path, n_rows, pred_val)
        logger.info("Row=%d  y_pred=%.6f", n_rows, pred_val)

        last_mtime = mtime
        last_row_count = n_rows
        time.sleep(poll_interval)


def _write_live_prediction(output_path: Path, row_idx: int, pred: float) -> None:
    import csv
    write_header = not output_path.exists()
    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["row_index", "y_pred"])
        writer.writerow([row_idx, round(pred, 8)])


def main(argv=None):
    import sfogliatella
    from sfogliatella.cli.args import build_common_parser, args_to_config
    from sfogliatella.cli.banner import print_banner
    from sfogliatella.core.utils import setup_logging
    from sfogliatella.devices.device import setup_device, configure_precision

    p = build_common_parser("Sfogliatella Live")
    p.add_argument("--poll_interval", type=float, default=1.0, help="Poll interval in seconds")
    args = p.parse_args(argv)

    setup_logging(args.log_level, args.log_items)
    print_banner(sfogliatella.__version__, args.no_banner)
    setup_device(args.device, args.show_tpu_warnings)
    configure_precision(args.precision)

    config = args_to_config(args)

    if args.data_path is None:
        p.error("--data_path (CSV file to watch) is required for live mode")

    watch_and_predict(
        config,
        csv_path=args.data_path,
        output_path=args.output_path,
        poll_interval=args.poll_interval,
    )
