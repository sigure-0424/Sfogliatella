"""Entry point: python -m eval"""
import argparse
import json
from sfogliatella.cli.banner import print_banner
from sfogliatella.core.utils import setup_logging

def main(argv=None):
    import sfogliatella
    p = argparse.ArgumentParser(description="Sfogliatella Eval — Walk-forward evaluation & metrics")
    p.add_argument("--predictions_path", required=True, help="Predictions CSV")
    p.add_argument("--task", default="regression")
    p.add_argument("--no_banner", action="store_true")
    p.add_argument("--log_level", default="info")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    print_banner(sfogliatella.__version__, args.no_banner)

    import csv, numpy as np
    from sfogliatella.core.eval import compute_metrics

    rows = []
    with open(args.predictions_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows or "y_pred" not in rows[0]:
        print("No y_pred column found in predictions CSV.")
        return

    y_pred = np.array([float(r["y_pred"]) for r in rows])
    if "y_true" in rows[0]:
        y_true = np.array([float(r["y_true"]) for r in rows])
        metrics = compute_metrics(y_pred, y_true, args.task)
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
    else:
        print(f"Predictions: {len(y_pred)} rows (no ground truth for metrics)")

if __name__ == "__main__":
    main()
