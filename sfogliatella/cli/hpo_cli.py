"""HPO CLI implementation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from sfogliatella.cli.banner import print_banner
from sfogliatella.core.utils import setup_logging, save_json


def build_hpo_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sfogliatella HPO — Hyperparameter Prior Solver")
    p.add_argument("--model", default="mlp",
                   choices=["mlp","lstm","rnn","transformer","cnn","xgboost"],
                   help="Model to compute HPO prior for")
    p.add_argument("--num_train_samples", type=int, default=10000)
    p.add_argument("--num_epochs",        type=int, default=10)
    p.add_argument("--batch_size",        type=int, default=32)
    p.add_argument("--lookback",          type=int, default=32)
    p.add_argument("--horizon",           type=int, default=1)
    p.add_argument("--input_dim",         type=int, default=1)
    p.add_argument("--output_dim",        type=int, default=1)
    p.add_argument("--target_loss",       type=float, default=0.1)
    p.add_argument("--noise_floor",       type=float, default=0.0)
    p.add_argument("--max_training_hours",type=float, default=24.0)
    p.add_argument("--lambda_data_cap",   type=float, default=20.0)
    p.add_argument("--gpu_vram_gb",       type=float, default=24.0)
    p.add_argument("--gpu_peak_tflops",   type=float, default=30.0)
    p.add_argument("--gpu_utilization",   type=float, default=0.4)
    p.add_argument("--optimizer",         default="adamw",
                   choices=["adamw","adam","sgd","adafactor"])
    p.add_argument("--baseline_measurements", default=None,
                   help="JSON list of [N, loss] pairs, e.g. '[[1e6,0.5],[4e6,0.35]]'")
    p.add_argument("--return_why",        action="store_true", default=True,
                   help="Include 'why' payload in output")
    p.add_argument("--output_path",       default=None, help="Save result JSON here")
    p.add_argument("--return_json",       action="store_true")
    p.add_argument("--no_banner",         action="store_true")
    p.add_argument("--log_level",         default="info")
    p.add_argument("--task",              default="regression")
    return p


def main(argv=None):
    import sfogliatella
    from sfogliatella.hpo.solver import SolverInputs, HardwareConfig, solve_hpo_prior, HPOResult

    p = build_hpo_parser()
    args = p.parse_args(argv)

    setup_logging(args.log_level)
    print_banner(sfogliatella.__version__, args.no_banner)

    hw = HardwareConfig(
        gpu_vram_gb=args.gpu_vram_gb,
        gpu_peak_tflops=args.gpu_peak_tflops,
        gpu_utilization=args.gpu_utilization,
        optimizer=args.optimizer,
    )

    baseline = None
    if args.baseline_measurements:
        raw = json.loads(args.baseline_measurements)
        baseline = [(float(x[0]), float(x[1])) for x in raw]

    inp = SolverInputs(
        baseline_measurements=baseline,
        target_loss=args.target_loss,
        noise_floor=args.noise_floor,
        num_train_samples=args.num_train_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lookback=args.lookback,
        horizon=args.horizon,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        max_training_hours=args.max_training_hours,
        lambda_data_cap=args.lambda_data_cap,
        hardware=hw,
        task=args.task,
    )

    raw_result = solve_hpo_prior(args.model, inp, return_why=args.return_why)

    result = HPOResult(
        model=args.model,
        status=raw_result.get("status", "unknown"),
        config=raw_result.get("config", {}),
        estimates=raw_result.get("estimates", {}),
        search_space=raw_result.get("hpo_search_space", {}),
        why=raw_result.get("why"),
        raw=raw_result,
    )

    # Print summary
    import logging
    logger = logging.getLogger(__name__)
    logger.info("HPO result for %s: status=%s  active_constraint=%s",
                args.model, result.status, raw_result.get("active_constraint"))
    logger.info("  config: %s", result.config)
    logger.info("  estimates: %s", result.estimates)

    if args.output_path:
        save_json(result.to_dict(), args.output_path)
        logger.info("HPO result saved: %s", args.output_path)

    if args.return_json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        # Human-readable summary
        print(f"\nHPO Prior for {args.model} ({result.status})")
        print(f"  Active constraint: {raw_result.get('active_constraint','?')}")
        print(f"  Config: {result.config}")
        print(f"  Estimates: {result.estimates}")
        if result.why and result.why.get("shrink_steps"):
            print(f"  Shrink steps taken: {len(result.why['shrink_steps'])}")
        print(f"  HPO search space: {result.search_space}")

    return result
