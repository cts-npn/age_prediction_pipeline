"""
Step 3 — train the dual-head (classification + regression) model.

Usage:
    python train.py --manifest manifest_aligned.csv --epochs 30 --out-dir runs/exp1

Key options:
    --dataset-scope {full,narrow,combined}   which images to train on (default: combined)
    --backbone {resnet18,efficientnet_b0}
    --class-loss-weight                      weight on the classification loss (0-1); regression gets (1 - weight)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AgeDataset, load_manifest
from model import DualHeadAgeModel


def run_epoch(model, loader, device, optimizer, class_loss_fn, reg_loss_fn, class_weight, train: bool):
    model.train(train)
    total_loss = total_class_correct = total_reg_abs_err = n = 0
    torch.set_grad_enabled(train)
    for imgs, class_labels, ages in tqdm(loader, leave=False):
        imgs, class_labels, ages = imgs.to(device), class_labels.to(device), ages.to(device)
        if train:
            optimizer.zero_grad()
        class_logits, reg_out = model(imgs)
        loss_c = class_loss_fn(class_logits, class_labels)
        loss_r = reg_loss_fn(reg_out, ages)
        loss = class_weight * loss_c + (1 - class_weight) * loss_r
        if train:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_class_correct += (class_logits.argmax(1) == class_labels).sum().item()
        total_reg_abs_err += (reg_out - ages).abs().sum().item()
        n += bs

    return total_loss / n, total_class_correct / n, total_reg_abs_err / n


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_manifest(args.manifest, args.dataset_scope, drop_failed_alignment=args.drop_failed_alignment)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    num_classes = max_age - min_age + 1
    print(f"Age range {min_age}-{max_age} -> {num_classes} classes. "
          f"Train rows: {len(train_df)}, test rows: {len(test_df)}")

    # Stratified train/val split (fall back to unstratified if any age has <2 samples)
    try:
        train_sub, val_sub = train_test_split(
            train_df, test_size=0.1, stratify=train_df["age"], random_state=42
        )
    except ValueError:
        train_sub, val_sub = train_test_split(train_df, test_size=0.1, random_state=42)

    train_ds = AgeDataset(train_sub, min_age, max_age, train=True, img_size=args.img_size)
    val_ds = AgeDataset(val_sub, min_age, max_age, train=False, img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = DualHeadAgeModel(num_classes=num_classes, backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    class_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    reg_loss_fn = nn.SmoothL1Loss()  # Huber loss

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_epoch = 1

    last_ckpt_path = out_dir / "last_checkpoint.pt"
    if args.resume:
        if not last_ckpt_path.exists():
            raise FileNotFoundError(f"--resume given but no checkpoint found at {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_loss = ckpt["best_val_loss"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        start_epoch = ckpt["epoch"] + 1
        history_path = out_dir / "training_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"Resumed from epoch {ckpt['epoch']} (best val loss so far: {best_val_loss:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, train_mae = run_epoch(
            model, train_loader, device, optimizer, class_loss_fn, reg_loss_fn, args.class_loss_weight, train=True
        )
        val_loss, val_acc, val_mae = run_epoch(
            model, val_loader, device, optimizer, class_loss_fn, reg_loss_fn, args.class_loss_weight, train=False
        )
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} MAE {train_mae:.2f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f} MAE {val_mae:.2f}")
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, train_mae=train_mae,
                             val_loss=val_loss, val_acc=val_acc, val_mae=val_mae))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "min_age": min_age, "max_age": max_age,
                 "num_classes": num_classes, "backbone": args.backbone, "img_size": args.img_size},
                out_dir / "best_model.pt",
            )
            print(f"  -> saved new best checkpoint (val loss {val_loss:.4f})")
        else:
            epochs_no_improve += 1

        torch.save(
            {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
             "scheduler_state": scheduler.state_dict(), "epoch": epoch, "best_val_loss": best_val_loss,
             "epochs_no_improve": epochs_no_improve, "min_age": min_age, "max_age": max_age,
             "num_classes": num_classes, "backbone": args.backbone, "img_size": args.img_size},
            last_ckpt_path,
        )
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        if epochs_no_improve >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping early.")
            break

    # Save the resolved test split too, so evaluate.py uses exactly what train.py held out.
    test_df.to_csv(out_dir / "test_split.csv", index=False)
    print(f"\nDone. Best checkpoint and logs saved under {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("manifest_aligned.csv"))
    ap.add_argument("--dataset-scope", choices=["full", "narrow", "combined"], default="combined")
    ap.add_argument("--drop-failed-alignment", action="store_true",
                     help="Exclude images where face alignment failed (default: keep them, resized only).")
    ap.add_argument("--backbone", choices=["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2"], default="resnet18")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--class-loss-weight", type=float, default=0.5,
                     help="Weight on classification loss; regression loss gets (1 - this).")
    ap.add_argument("--patience", type=int, default=5, help="Early-stopping patience in epochs.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/exp1"))
    ap.add_argument("--resume", action="store_true",
                     help="Resume from <out-dir>/last_checkpoint.pt (model, optimizer, scheduler, epoch count).")
    args = ap.parse_args()
    main(args)
