"""
Step 2 — detect faces and align them to a canonical pose, for every image in
the manifest. Aligned copies are written to a parallel folder tree; originals
are never modified. The manifest is updated with 'aligned_path' and
'alignment_ok' columns.

This is resumable: re-running skips images that already have an aligned
output on disk, so it's safe to stop and restart (Ctrl+C is fine).

Usage:
    python align_faces.py --manifest manifest.csv --aligned-dir aligned --out manifest_aligned.csv

If you only want to test on a subset first:
    python align_faces.py --manifest manifest.csv --aligned-dir aligned --out manifest_aligned.csv --limit 500
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform as sktransform
from tqdm import tqdm

from face_detector import detect_landmarks

# Canonical 5-point landmark positions for a 128x128 output (roughly matches
# the ArcFace/FaceNet alignment convention, scaled down from 112 to 128).
REFERENCE_LANDMARKS = np.array(
    [
        [45.0, 52.0],   # left eye
        [83.0, 52.0],   # right eye
        [64.0, 72.0],   # nose tip
        [50.0, 96.0],   # left mouth corner
        [79.0, 96.0],   # right mouth corner
    ],
    dtype=np.float32,
)
OUTPUT_SIZE = 128


def align_image(img: Image.Image, landmarks: np.ndarray, size: int = OUTPUT_SIZE) -> Image.Image:
    tform = sktransform.SimilarityTransform()
    tform.estimate(landmarks.astype(np.float64), REFERENCE_LANDMARKS)
    arr = np.array(img)
    warped = sktransform.warp(arr, tform.inverse, output_shape=(size, size), preserve_range=True)
    return Image.fromarray(warped.astype(np.uint8))


def main(manifest_path: Path, aligned_dir: Path, out_path: Path, limit: int, force_realign: bool = False):
    df = pd.read_csv(manifest_path)
    if limit:
        df = df.head(limit).copy()

    aligned_paths = []
    ok_flags = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="aligning"):
        src = Path(row["filepath"])
        rel = Path(row["source_dataset"]) / row["split"] / str(row["age"]) / src.name
        dst = aligned_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not force_realign:
            aligned_paths.append(str(dst))
            ok_flags.append(True)  # assume a previous successful run; re-run with --force-realign to redo
            continue

        try:
            img = Image.open(src).convert("RGB")
            landmarks = detect_landmarks(img)
            if landmarks is None:
                raise ValueError("no face detected")
            aligned = align_image(img, landmarks)
            aligned.save(dst)
            aligned_paths.append(str(dst))
            ok_flags.append(True)
        except Exception:
            # Detection/alignment failed: fall back to the original image,
            # resized to the target size, and flag it so it can be filtered
            # or reviewed later.
            try:
                img = Image.open(src).convert("RGB").resize((OUTPUT_SIZE, OUTPUT_SIZE))
                img.save(dst)
            except Exception:
                dst = src  # last resort: point at the original file
            aligned_paths.append(str(dst))
            ok_flags.append(False)

    df["aligned_path"] = aligned_paths
    df["alignment_ok"] = ok_flags
    df.to_csv(out_path, index=False)

    n_fail = (~df["alignment_ok"]).sum()
    print(f"\nDone. {len(df)} images processed, {n_fail} failed alignment ({n_fail / len(df):.1%}).")
    print(f"Wrote {out_path}")
    print("Failed rows kept a resized (unaligned) copy and alignment_ok=False, so you can filter or inspect them later.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    ap.add_argument("--aligned-dir", type=Path, default=Path("aligned"))
    ap.add_argument("--out", type=Path, default=Path("manifest_aligned.csv"))
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N rows (for a quick test run).")
    ap.add_argument(
        "--force-realign",
        action="store_true",
        help="Reprocess images even if a file already exists at the aligned destination path.",
    )
    args = ap.parse_args()
    main(args.manifest, args.aligned_dir, args.out, args.limit, args.force_realign)
