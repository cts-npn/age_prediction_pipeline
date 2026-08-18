"""
Step 5 (optional) — evaluate every trained model in one pass.

Auto-discovers run directories (each produced by train.py, containing
best_model.pt + test_split.csv), runs the same classification-vs-regression
evaluation as evaluate.py on each, writes each model's usual
comparison_report.csv / summary_metrics.csv / predicted_vs_actual.png into
its own run dir, and combines all summaries into a single side-by-side table.

Usage:
    python evaluate_all.py --runs-root runs
    python evaluate_all.py --run-dirs runs/resnet18 runs/resnet50 runs/efficientnet_b0
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from evaluate import evaluate_checkpoint


def discover_run_dirs(runs_root: Path):
    return sorted(
        p.parent for p in runs_root.glob("*/best_model.pt")
        if (p.parent / "test_split.csv").exists()
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_dirs = args.run_dirs or discover_run_dirs(args.runs_root)
    if not run_dirs:
        raise SystemExit(f"No run directories with best_model.pt + test_split.csv found under {args.runs_root}")

    labels = args.labels or [d.name for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit("--labels must have the same number of entries as the run directories being evaluated")

    rows = []
    for run_dir, label in zip(run_dirs, labels):
        print(f"\n=== Evaluating {label} ({run_dir}) ===")
        checkpoint = run_dir / "best_model.pt"
        test_split = run_dir / "test_split.csv"

        results, summary, min_age, max_age = evaluate_checkpoint(
            checkpoint, test_split, args.batch_size, device
        )

        results.to_csv(run_dir / "comparison_report.csv", index=False)
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(run_dir / "summary_metrics.csv")
        print(summary_df.round(3).to_string())

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
        for ax, col, title in zip(
            axes, ["classification_pred", "regression_pred"], ["Classification", "Regression"]
        ):
            ax.scatter(results["true_age"], results[col], alpha=0.3, s=8)
            lims = [min_age, max_age]
            ax.plot(lims, lims, "r--", linewidth=1)
            ax.set_xlabel("True age")
            ax.set_ylabel("Predicted age")
            ax.set_title(title)
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(run_dir / "predicted_vs_actual.png", dpi=150)
        plt.close(fig)

        model_summary = summary_df.reset_index().rename(columns={"index": "approach"})
        model_summary.insert(0, "backbone", label)
        rows.append(model_summary)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(args.out, index=False)

    print("\n=== All models — classification vs regression, side by side ===")
    print(combined.round(3).to_string(index=False))
    print(f"\nCombined comparison written to {args.out}")

    for approach in combined["approach"].unique():
        sub = combined[combined["approach"] == approach]
        best = sub.loc[sub["MAE"].idxmin()]
        print(f"Lowest MAE for {approach}: {best['backbone']} (MAE {best['MAE']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("runs"),
                     help="Parent directory to auto-discover run subfolders in (used when --run-dirs is omitted).")
    ap.add_argument("--run-dirs", type=Path, nargs="+", default=None,
                     help="Explicit run directories to evaluate, each with best_model.pt + test_split.csv.")
    ap.add_argument("--labels", nargs="+", default=None,
                     help="Optional display names for each run dir, same order as the run dirs being evaluated.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("all_models_comparison.csv"))
    args = ap.parse_args()
    main(args)
