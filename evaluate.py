"""
Step 4 — evaluate the trained model on the held-out test split and produce a
side-by-side comparison of the classification approach vs the regression
approach (the two results shown together, as requested).

Usage:
    python evaluate.py --checkpoint runs/exp1/best_model.pt --test-split runs/exp1/test_split.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AgeDataset
from model import DualHeadAgeModel


def within_k_accuracy(preds, true, k):
    return float((np.abs(np.array(preds) - np.array(true)) <= k).mean())


def evaluate_checkpoint(checkpoint: Path, test_split: Path, batch_size: int, device):
    """
    Run one trained model against its held-out test split.
    Returns (per-image results df, {'classification': {...}, 'regression': {...}} summary, min_age, max_age).
    """
    ckpt = torch.load(checkpoint, map_location=device)
    min_age, max_age = ckpt["min_age"], ckpt["max_age"]

    model = DualHeadAgeModel(num_classes=ckpt["num_classes"], backbone=ckpt["backbone"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_df = pd.read_csv(test_split)
    ds = AgeDataset(test_df, min_age, max_age, train=False, img_size=ckpt["img_size"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    true_ages, class_preds, reg_preds = [], [], []
    with torch.no_grad():
        for imgs, class_labels, ages in tqdm(loader, leave=False):
            imgs = imgs.to(device)
            class_logits, reg_out = model(imgs)
            class_pred_age = class_logits.argmax(1).cpu().numpy() + min_age
            reg_pred_age = reg_out.cpu().numpy()

            true_ages.extend(ages.numpy().tolist())
            class_preds.extend(class_pred_age.tolist())
            reg_preds.extend(reg_pred_age.tolist())

    results = pd.DataFrame(
        {
            "true_age": true_ages,
            "classification_pred": class_preds,
            "regression_pred": reg_preds,
        }
    )
    results["classification_error"] = (results["classification_pred"] - results["true_age"]).abs()
    results["regression_error"] = (results["regression_pred"] - results["true_age"]).abs()
    results["agreement_diff"] = (results["classification_pred"] - results["regression_pred"]).abs()

    summary = {
        "classification": {
            "exact_accuracy": float((results["classification_pred"] == results["true_age"]).mean()),
            "within_3_years": within_k_accuracy(results["classification_pred"], results["true_age"], 3),
            "within_5_years": within_k_accuracy(results["classification_pred"], results["true_age"], 5),
            "MAE": mean_absolute_error(results["true_age"], results["classification_pred"]),
            "RMSE": mean_squared_error(results["true_age"], results["classification_pred"]) ** 0.5,
        },
        "regression": {
            "within_3_years": within_k_accuracy(results["regression_pred"], results["true_age"], 3),
            "within_5_years": within_k_accuracy(results["regression_pred"], results["true_age"], 5),
            "MAE": mean_absolute_error(results["true_age"], results["regression_pred"]),
            "RMSE": mean_squared_error(results["true_age"], results["regression_pred"]) ** 0.5,
        },
    }
    return results, summary, min_age, max_age


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, summary, min_age, max_age = evaluate_checkpoint(
        args.checkpoint, args.test_split, args.batch_size, device
    )

    out_dir = args.checkpoint.parent
    results.to_csv(out_dir / "comparison_report.csv", index=False)

    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(out_dir / "summary_metrics.csv")

    print("\n=== Classification vs Regression — side by side ===")
    print(summary_df.round(3).to_string())
    print(f"\nFull per-image comparison written to {out_dir / 'comparison_report.csv'}")
    print(f"Summary metrics written to {out_dir / 'summary_metrics.csv'}")

    # Scatter plot: predicted vs actual, both methods on one figure.
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
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    print(f"Scatter plot written to {out_dir / 'predicted_vs_actual.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/resnet50/best_model.pt"))
    ap.add_argument("--test-split", type=Path, default=Path("runs/resnet50/test_split.csv"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    main(args)
