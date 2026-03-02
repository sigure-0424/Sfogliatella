"""Entry point: python -m pack"""
import argparse
import sys
from pathlib import Path
from sfogliatella.cli.banner import print_banner
from sfogliatella.core.utils import setup_logging

def main(argv=None):
    import sfogliatella
    p = argparse.ArgumentParser(description="Sfogliatella Pack — Single-file model packaging")
    p.add_argument("model_dir", help="Model directory")
    p.add_argument("run_id", help="Run ID to package")
    p.add_argument("--output", default=None, help="Output .sfog file path")
    p.add_argument("--no_banner", action="store_true")
    p.add_argument("--log_level", default="info")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    print_banner(sfogliatella.__version__, args.no_banner)
    from sfogliatella.io.pack import pack_model
    out = pack_model(Path(args.model_dir), args.run_id, Path(args.output) if args.output else None)
    print(f"Packed: {out}")

if __name__ == "__main__":
    main()
