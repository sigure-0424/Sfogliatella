"""Batch prediction CLI implementation."""

from __future__ import annotations

import json
import sys

from sfogliatella.cli.args import build_common_parser, args_to_config
from sfogliatella.cli.banner import print_banner
from sfogliatella.core.utils import setup_logging
from sfogliatella.devices.device import setup_device, configure_precision


def main(argv=None):
    import sfogliatella
    p = build_common_parser("Sfogliatella Predict")
    args = p.parse_args(argv)

    setup_logging(args.log_level, args.log_items)
    print_banner(sfogliatella.__version__, args.no_banner)

    setup_device(args.device, args.show_tpu_warnings)
    configure_precision(args.precision)

    config = args_to_config(args)

    if args.data_path is None:
        p.error("--data_path is required for prediction")

    from sfogliatella.core.trainer import predict_model
    result = predict_model(
        config,
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_path=args.output_path,
    )

    if args.return_json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")

    return result
