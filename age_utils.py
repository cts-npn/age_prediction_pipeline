"""Shared helpers used by every script in this pipeline."""
import re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def age_from_folder_name(name: str) -> int:
    """Folder names are the age label, sometimes zero-padded ('001', '045', '20')."""
    m = re.match(r"^0*(\d+)$", name.strip())
    if not m:
        raise ValueError(f"Folder name '{name}' does not look like an age label")
    return int(m.group(1))


def iter_age_images(dataset_root: Path, split: str):
    """Yield (filepath, age) for every image under dataset_root/<split>/<age>/*.jpg"""
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        return
    for age_dir in sorted(split_dir.iterdir()):
        if not age_dir.is_dir():
            continue
        try:
            age = age_from_folder_name(age_dir.name)
        except ValueError:
            continue
        for f in age_dir.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                yield f, age
