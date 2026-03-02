"""Stacking/pipeline CLI implementation."""

from __future__ import annotations

import argparse
import json
import sys

from sfogliatella.cli.banner import print_banner
from sfogliatella.core.utils import setup_logging, load_json_or_yaml


def build_stack_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sfogliatella Stack — Multi-Stage Pipeline")
    p.add_argument("--pipeline", default=None, help="Pipeline config as JSON string")
    p.add_argument("--pipeline_path", default=None, help="Path to pipeline config JSON/YAML file")
    p.add_argument("--data_path", default=None, help="Input data file")
    p.add_argument("--data_format", default=None, choices=["csv","parquet"])
    p.add_argument("--output_path", default=None, help="Prediction output path")
    p.add_argument("--lookback", type=int, default=32)
    p.add_argument("--horizon",  type=int, default=1)
    p.add_argument("--target_col", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_banner", action="store_true")
    p.add_argument("--log_level", default="info")
    p.add_argument("--return_json", action="store_true")
    return p


def main(argv=None):
    import sfogliatella
    from sfogliatella.stacking.pipeline import run_pipeline

    p = build_stack_parser()
    args = p.parse_args(argv)

    setup_logging(args.log_level)
    print_banner(sfogliatella.__version__, args.no_banner)

    if args.pipeline_path:
        pipeline_config = load_json_or_yaml(args.pipeline_path)
    elif args.pipeline:
        pipeline_config = json.loads(args.pipeline)
    else:
        p.error("Either --pipeline or --pipeline_path is required")

    base_config = {
        "lookback": args.lookback,
        "horizon":  args.horizon,
        "target_col": args.target_col,
        "batch_size": args.batch_size,
        "seed":     args.seed,
        "data_format": args.data_format,
    }

    result = run_pipeline(
        pipeline_config,
        data_path=args.data_path,
        base_config=base_config,
        output_path=args.output_path,
    )

    if args.return_json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"Pipeline done. Status: {result.status}")
        if result.output_path:
            print(f"Output: {result.output_path}")

    return result
