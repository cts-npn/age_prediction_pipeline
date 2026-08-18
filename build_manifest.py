"""
Step 1 — build a CSV manifest of (filepath, age, split, source_dataset) from the
two dataset folders. Doesn't move or modify any images.

Usage:
    python build_manifest.py ^
        --narrow-root "D:\\NPN\\20-50\\20-50" ^
        --full-root "D:\\NPN\\age_prediction_up\\age_prediction" ^
        --out manifest.csv

Run this first. Re-run any time to refresh the manifest (e.g. after adding images).
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from age_utils import iter_age_images


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def build(narrow_root: Path, full_root: Path, out_path: Path, hash_for_dedup: bool):
    rows = []
    sources = []
    if full_root and full_root.is_dir():
        sources.append(("full", full_root))
    if narrow_root and narrow_root.is_dir():
        sources.append(("narrow", narrow_root))

    if not sources:
        raise SystemExit("Neither dataset root exists — check --narrow-root / --full-root paths.")

    for source_name, root in sources:
        for split in ("train", "test"):
            for filepath, age in tqdm(
                list(iter_age_images(root, split)),
                desc=f"{source_name}/{split}",
            ):
                rows.append(
                    {
                        "filepath": str(filepath),
                        "age": age,
                        "split": split,
                        "source_dataset": source_name,
                        "filename": filepath.name,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No images found under the given roots.")

    # Flag likely duplicates between the two datasets (same filename, or same
    # content if --hash-for-dedup is passed — slower but exact).
    if hash_for_dedup:
        print("Hashing files to find exact duplicates (this can take a while)...")
        df["content_hash"] = [file_hash(Path(p)) for p in tqdm(df["filepath"])]
        dup_key = "content_hash"
    else:
        dup_key = "filename"

    df["is_duplicate"] = df.duplicated(subset=[dup_key], keep=False) & (
        df["source_dataset"].nunique() > 1
    )
    # Prefer the 'full' dataset copy when the same image exists in both.
    df["keep"] = ~(df["is_duplicate"] & (df["source_dataset"] == "narrow"))

    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} rows to {out_path}")
    print("\nPer-source / per-split counts:")
    print(df.groupby(["source_dataset", "split"]).size())
    print("\nAge coverage (min/max) per source:")
    print(df.groupby("source_dataset")["age"].agg(["min", "max", "count"]))
    dup_count = int(df["is_duplicate"].sum())
    if dup_count:
        print(f"\n{dup_count} rows flagged as likely duplicates across datasets (see 'is_duplicate' / 'keep' columns).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrow-root", type=Path, default=Path(r"D:\NPN\20-50\20-50"),
                     help="Root of the ages-20-50 dataset (contains train/ and test/).")
    ap.add_argument("--full-root", type=Path, default=Path(r"D:\NPN\age_prediction_up\age_prediction"),
                     help="Root of the ages-1-100 dataset (contains train/ and test/).")
    ap.add_argument("--out", type=Path, default=Path("manifest.csv"))
    ap.add_argument("--hash-for-dedup", action="store_true",
                     help="Use file content hashing instead of filename matching to find duplicates (slower, exact).")
    args = ap.parse_args()
    build(args.narrow_root, args.full_root, args.out, args.hash_for_dedup)
