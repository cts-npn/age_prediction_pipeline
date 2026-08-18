"""
Step 6 (optional) — after training/evaluating more than one backbone into
separate --out-dir folders, combine their summary_metrics.csv files into one
table so you can compare backbones (and classification vs regression within
each) side by side.

Usage (after training+evaluating resnet18 and resnet50 into separate dirs):
    python compare_backbones.py --run-dirs runs\resnet18 runs\resnet50 runs\efficientnet_b0 --out backbone_comparison.csv

By default each run is labeled with its folder name; pass --labels to
override (must match --run-dirs in count and order).
"""
import argparse
from pathlib import Path

import pandas as pd


def main(args):
    if args.labels and len(args.labels) != len(args.run_dirs):
        raise SystemExit("--labels must have the same number of entries as --run-dirs")
    labels = args.labels or [d.name for d in args.run_dirs]

    frames = []
    for run_dir, label in zip(args.run_dirs, labels):
        summary_path = run_dir / "summary_metrics.csv"
        if not summary_path.exists():
            print(f"Skipping {run_dir} — no summary_metrics.csv found (run evaluate.py on it first).")
            continue
        df = pd.read_csv(summary_path, index_col=0)
        df.insert(0, "backbone", label)
        df.index.name = "approach"
        frames.append(df.reset_index())

    if not frames:
        raise SystemExit("No summary_metrics.csv files found in the given --run-dirs.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.out, index=False)

    print("\n=== Backbone comparison (classification vs regression, per backbone) ===")
    print(combined.round(3).to_string(index=False))
    print(f"\nWrote {args.out}")

    # Quick "best" call-outs.
    for approach in combined["approach"].unique():
        sub = combined[combined["approach"] == approach]
        best = sub.loc[sub["MAE"].idxmin()]
        print(f"Lowest MAE for {approach}: {best['backbone']} (MAE {best['MAE']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                     help="Two or more run directories, each already processed by train.py + evaluate.py.")
    ap.add_argument("--labels", nargs="+", default=None,
                     help="Optional display names for each run dir, same order as --run-dirs.")
    ap.add_argument("--out", type=Path, default=Path("backbone_comparison.csv"))
    args = ap.parse_args()
    main(args)
