# src/data/split.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir, project_root, save_json


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_class_folders(data_root: Path) -> List[Path]:
    """
    Returns a sorted list of class directories under data_root.
    A "class directory" is a folder that contains at least one image file.
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    class_dirs = []
    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        # Check if this directory has at least 1 image file
        has_image = any((f.suffix.lower() in IMG_EXTS) for f in p.iterdir() if f.is_file())
        if has_image:
            class_dirs.append(p)

    class_dirs = sorted(class_dirs, key=lambda x: x.name)
    if len(class_dirs) == 0:
        raise RuntimeError(
            f"No class folders with images were found under: {data_root}\n"
            f"Expected: data/raw/PlantVillage/<class_folder>/*.jpg"
        )
    return class_dirs


def collect_images(data_root: Path) -> pd.DataFrame:
    """
    Collect all image file paths and labels from a PlantVillage-style folder:
      data_root/
        ClassA/
          img1.jpg
        ClassB/
          img2.jpg

    Returns a DataFrame with columns:
      - filepath (relative to project root)
      - label (class folder name)
    """
    class_dirs = find_class_folders(data_root)

    rows: List[Tuple[str, str]] = []
    root = project_root()

    for class_dir in class_dirs:
        label = class_dir.name
        for f in class_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in IMG_EXTS:
                continue
            rel_path = f.resolve().relative_to(root.resolve())
            rows.append((str(rel_path), label))

    df = pd.DataFrame(rows, columns=["filepath", "label"])
    if df.empty:
        raise RuntimeError(f"No images found under: {data_root}")
    return df


def make_label_map(labels: List[str]) -> Dict[str, int]:
    """
    Deterministic mapping from class name -> integer id.
    We sort labels so mapping is stable across machines/runs.
    """
    unique = sorted(set(labels))
    return {name: idx for idx, name in enumerate(unique)}


def stratified_split(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    """
    Create stratified train/val/test splits.

    Output df columns:
      filepath,label,label_idx,split
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Create deterministic label map
    label_map = make_label_map(df["label"].tolist())
    df = df.copy()
    df["label_idx"] = df["label"].map(label_map)

    # First split: train vs temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["label_idx"],
    )

    # Second split: val vs test from temp
    # val_ratio and test_ratio are out of the full dataset; we need proportion within temp.
    temp_total = val_ratio + test_ratio
    val_size_within_temp = val_ratio / temp_total

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_size_within_temp),
        random_state=seed,
        stratify=temp_df["label_idx"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], axis=0).sample(frac=1.0, random_state=seed)
    out = out.reset_index(drop=True)

    return out, label_map


def print_summary(df: pd.DataFrame) -> None:
    """
    Print:
      - total counts per split
      - counts per class per split
    """
    print("\n===== Split Summary =====")
    print("Total images:", len(df))
    print("\nImages per split:")
    print(df["split"].value_counts())

    print("\nClasses:", df["label"].nunique())
    print("\nPer-class counts by split:")
    table = pd.crosstab(df["label"], df["split"])
    # Show all rows but in a consistent order
    table = table.sort_index()
    print(table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create train/val/test split CSV for PlantVillage-style dataset.")
    p.add_argument(
        "--data_root",
        type=str,
        default=str(project_root() / "data" / "raw" / "PlantVillage"),
        help="Path to dataset root containing class folders.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(project_root() / "data" / "splits"),
        help="Output directory for split.csv and label_map.json",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())

    df = collect_images(data_root)

    split_df, label_map = stratified_split(
        df=df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Save outputs
    split_csv_path = out_dir / "split.csv"
    label_map_path = out_dir / "label_map.json"

    split_df.to_csv(split_csv_path, index=False)
    save_json(label_map_path, label_map)

    print(f"\nSaved split CSV to: {split_csv_path}")
    print(f"Saved label map to: {label_map_path}")

    print_summary(split_df)


if __name__ == "__main__":
    main()
