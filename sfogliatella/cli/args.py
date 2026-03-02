"""Common CLI argument parser factory."""

from __future__ import annotations

import argparse
import json
from typing import Optional


def build_common_parser(description: str = "Sfogliatella") -> argparse.ArgumentParser:
    """Return an ArgumentParser with all common arguments pre-registered."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model
    p.add_argument("--model", default="mlp",
                   choices=["mlp", "lstm", "rnn", "transformer", "cnn", "xgboost",
                            "timesfm", "timesfm_ft", "custom"],
                   help="Model name")
    p.add_argument("--model_params", default=None, help="Model-specific params as JSON string")
    p.add_argument("--model_params_path", default=None, help="Model-specific params JSON/YAML file")
    p.add_argument("--custom_model_root", default=None, help="Path to user model code (--model custom)")
    p.add_argument("--custom_model_call", default=None, help="Invocation command for custom model")

    # Task
    p.add_argument("--task", default="regression",
                   choices=["regression","classification","ranking","anomaly_score","probability","clustering","embedding"],
                   help="Task type")
    p.add_argument("--num_classes", type=int, default=2, help="Num classes (classification)")
    p.add_argument("--task_params", default=None, help="Task-specific params as JSON string")
    p.add_argument("--task_params_path", default=None, help="Task-specific params JSON/YAML file")

    # Data
    p.add_argument("--data_path", default=None, help="Input data file (CSV/Parquet)")
    p.add_argument("--data_format", default=None, choices=["csv","parquet"], help="Data format (auto-detect if omitted)")
    p.add_argument("--target_col", type=int, default=0, help="Target column index (0-based)")
    p.add_argument("--feature_cols", default=None, help="Feature column indices as JSON list, e.g. '[1,2,3]'")

    # Windows
    p.add_argument("--lookback", type=int, default=32, help="Lookback window length")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon")

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=1, help="Number of time-series folds (default 1)")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    # Loss
    p.add_argument("--loss", default=None, help="Built-in loss name")
    p.add_argument("--loss_path", default=None, help="Path to custom loss Python file")

    # IO
    p.add_argument("--model_dir", default="outputs/models", help="Model save/load directory")
    p.add_argument("--run_id", default=None, help="Run identifier (default: YYYYMMDD_HHMMSS)")
    p.add_argument("--output_path", default=None, help="Prediction output file path")
    p.add_argument("--return_json", action="store_true", help="Emit structured JSON result to stdout")

    # Device
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda","tpu"])
    p.add_argument("--num_devices", type=int, default=1)
    p.add_argument("--precision", default="auto", choices=["auto","fp32","bf16","fp16"])

    # Logging
    p.add_argument("--log_level", default="info", choices=["error","warn","info","debug"])
    p.add_argument("--log_items", default=None, help="Log item groups (e.g. 'run,device,data,train')")
    p.add_argument("--no_banner", action="store_true", help="Suppress startup banner")
    p.add_argument("--show_tpu_warnings", action="store_true", help="Show TPU init warnings")

    # Checkpointing
    p.add_argument("--checkpoint_every", type=int, default=1, help="Checkpoint every N epochs")
    p.add_argument("--resume_from", default=None, help="Path to checkpoint to resume from")

    return p


def parse_model_params(args: argparse.Namespace) -> dict:
    """Merge model_params_path and model_params into a dict."""
    from sfogliatella.core.utils import load_json_or_yaml
    params = {}
    if args.model_params_path:
        params.update(load_json_or_yaml(args.model_params_path))
    if args.model_params:
        params.update(json.loads(args.model_params))
    return params


def parse_feature_cols(feature_cols_str: Optional[str]) -> Optional[list]:
    if feature_cols_str is None:
        return None
    return json.loads(feature_cols_str)


def args_to_config(args: argparse.Namespace) -> dict:
    """Convert parsed namespace to config dict."""
    config = {
        "model":           args.model,
        "task":            args.task,
        "num_classes":     args.num_classes,
        "lookback":        args.lookback,
        "horizon":         args.horizon,
        "target_col":      args.target_col,
        "feature_cols":    parse_feature_cols(getattr(args, "feature_cols", None)),
        "batch_size":      args.batch_size,
        "epochs":          args.epochs,
        "lr":              args.lr,
        "seed":            args.seed,
        "folds":           args.folds,
        "val_ratio":       args.val_ratio,
        "test_ratio":      args.test_ratio,
        "loss":            args.loss,
        "loss_path":       args.loss_path,
        "model_dir":       args.model_dir,
        "run_id":          args.run_id,
        "output_path":     args.output_path,
        "device":          args.device,
        "precision":       args.precision,
        "checkpoint_every": args.checkpoint_every,
        "resume_from":     args.resume_from,
        "data_format":     args.data_format,
    }
    config.update(parse_model_params(args))
    return config
