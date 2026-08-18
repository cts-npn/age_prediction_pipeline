"""PyTorch Dataset for the age-prediction manifest, shared by train/evaluate."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, img_size: int = 128):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class AgeDataset(Dataset):
    """
    Expects a manifest CSV with at least: filepath (or aligned_path), age, split.
    `min_age`/`max_age` fix the classification label space so class indices are
    consistent between train/val/test and between separate runs.
    """

    def __init__(self, manifest_df: pd.DataFrame, min_age: int, max_age: int, train: bool, img_size: int = 128):
        self.df = manifest_df.reset_index(drop=True)
        self.min_age = min_age
        self.max_age = max_age
        self.transform = build_transforms(train, img_size)
        path_col = "aligned_path" if "aligned_path" in self.df.columns else "filepath"
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[self.path_col]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        age = int(row["age"])
        class_label = age - self.min_age  # zero-indexed class
        return img, torch.tensor(class_label, dtype=torch.long), torch.tensor(float(age), dtype=torch.float32)


def load_manifest(manifest_path: Path, dataset_scope: str, drop_failed_alignment: bool) -> pd.DataFrame:
    """
    dataset_scope: 'full' (age_prediction_up only), 'narrow' (20-50 only),
    or 'combined' (both, de-duplicated using the manifest's keep/is_duplicate columns).
    """
    df = pd.read_csv(manifest_path)

    if "keep" in df.columns:
        df = df[df["keep"]].copy()

    if dataset_scope == "full":
        df = df[df["source_dataset"] == "full"]
    elif dataset_scope == "narrow":
        df = df[df["source_dataset"] == "narrow"]
    # 'combined' keeps everything already filtered by `keep`

    if drop_failed_alignment and "alignment_ok" in df.columns:
        df = df[df["alignment_ok"]]

    path_col = "aligned_path" if "aligned_path" in df.columns else "filepath"
    exists = df[path_col].apply(lambda p: Path(p).exists())
    n_missing = (~exists).sum()
    if n_missing:
        print(f"Warning: {n_missing} of {len(df)} rows point to missing files on disk "
              f"({n_missing / len(df):.2%}) — dropping them from this run.")
        df = df[exists]

    return df.reset_index(drop=True)
