"""Entry point: python -m unpack"""
import argparse
from pathlib import Path
from sfogliatella.cli.banner import print_banner
from sfogliatella.core.utils import setup_logging

def main(argv=None):
    import sfogliatella
    p = argparse.ArgumentParser(description="Sfogliatella Unpack — Unpack .sfog model archive")
    p.add_argument("pack_path", help="Path to .sfog archive")
    p.add_argument("--output_dir", default=None, help="Output directory")
    p.add_argument("--no_banner", action="store_true")
    p.add_argument("--log_level", default="info")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    print_banner(sfogliatella.__version__, args.no_banner)
    from sfogliatella.io.pack import unpack_model
    out = unpack_model(Path(args.pack_path), Path(args.output_dir) if args.output_dir else None)
    print(f"Unpacked to: {out}")

if __name__ == "__main__":
    main()
