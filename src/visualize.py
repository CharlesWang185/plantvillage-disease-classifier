# src/visualize.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.utils import get_device, project_root, ensure_dir, load_json
from src.data.dataset import PlantVillageDataset
from src.data.transforms import build_eval_transforms
from src.models.cnn_baseline import CNNBaseline
from src.models.transfer import build_model


# -----------------------------
# Utilities
# -----------------------------
def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_label_names() -> Optional[List[str]]:
    """
    Try to load label_map.json saved by split.py.
    We expect label_map.json is usually: { "ClassName": idx, ... }
    Return list where index -> class name.
    """
    root = project_root()
    p = root / "data" / "splits" / "label_map.json"
    if not p.exists():
        return None
    label_map = load_json(p)  # name -> idx
    # invert
    inv = [None] * (max(label_map.values()) + 1)
    for name, idx in label_map.items():
        inv[idx] = name
    # replace any None (just in case)
    inv = [x if x is not None else f"class_{i}" for i, x in enumerate(inv)]
    return inv


def _ensure_reports_figures() -> Path:
    root = project_root()
    reports = ensure_dir(root / "reports")
    figs = ensure_dir(reports / "figures")
    return figs


# -----------------------------
# 1) Training curves
# -----------------------------
def plot_training_curves(history_path: Path, out_dir: Path) -> Path:
    """
    Plots:
      - train_loss vs epoch (by step order)
      - val_loss vs epoch
      - val accuracy vs epoch
      - val macro_f1 vs epoch
    Saves a single PNG.
    """
    hist = _read_json(history_path)
    if not isinstance(hist, list) or len(hist) == 0:
        raise ValueError(f"history.json is empty or invalid: {history_path}")

    # Make a simple "step" x-axis so head+finetune epochs appear sequentially
    steps = list(range(1, len(hist) + 1))
    train_loss = [row.get("train_loss", None) for row in hist]
    val_loss = [row.get("val_loss", None) for row in hist]
    val_acc = [row.get("val_summary", {}).get("accuracy", None) for row in hist]
    val_f1 = [row.get("val_summary", {}).get("macro_f1", None) for row in hist]
    phases = [row.get("phase", "") for row in hist]

    fig_path = out_dir / "training_curves.png"

    plt.figure(figsize=(10, 8))

    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(steps, train_loss, label="train_loss")
    plt.plot(steps, val_loss, label="val_loss")
    plt.xlabel("epoch step (head+finetune)")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Metrics plot
    plt.subplot(2, 1, 2)
    plt.plot(steps, val_acc, label="val_accuracy")
    plt.plot(steps, val_f1, label="val_macro_f1")
    plt.xlabel("epoch step (head+finetune)")
    plt.ylabel("score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add simple phase boundary line (where phase switches)
    # Find first index where phase == "finetune"
    if "finetune" in phases and "head" in phases:
        switch_idx = phases.index("finetune") + 1  # steps are 1-based
        plt.subplot(2, 1, 1)
        plt.axvline(switch_idx, linestyle="--", linewidth=1)
        plt.subplot(2, 1, 2)
        plt.axvline(switch_idx, linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


# -----------------------------
# 2) Confusion matrix plot
# -----------------------------
def plot_confusion_matrix(
    cm_path: Path,
    out_dir: Path,
    normalize: bool = False,
    max_labels: int = 30,
) -> Path:
    """
    Plots confusion matrix from .npy.
    If too many labels, it will still plot but may be unreadable; for your case (15) it's fine.
    """
    cm = np.load(cm_path)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got {cm.shape}")

    labels = _load_label_names()
    n = cm.shape[0]
    if labels is None or len(labels) != n:
        labels = [str(i) for i in range(n)]

    cm_to_plot = cm.astype(float)
    if normalize:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_to_plot = cm_to_plot / row_sums

    fig_path = out_dir / ("confusion_matrix_norm.png" if normalize else "confusion_matrix.png")

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    # ticks
    if n <= max_labels:
        tick_marks = np.arange(n)
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels)
    else:
        # Too many labels -> just show indices
        tick_marks = np.arange(n)
        plt.xticks(tick_marks, [str(i) for i in range(n)], rotation=90)
        plt.yticks(tick_marks, [str(i) for i in range(n)])

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


# -----------------------------
# 3) Misclassified gallery (optional)
# -----------------------------
def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_num_classes() -> int:
    labels = _load_label_names()
    if labels is None:
        raise FileNotFoundError("data/splits/label_map.json not found; cannot infer num_classes.")
    return len(labels)


def _build_model_from_cfg(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    name = cfg["model"]["name"].lower().strip()
    pretrained = bool(cfg["model"].get("pretrained", True))
    dropout = float(cfg["model"].get("dropout", 0.0))

    if name in {"baseline_cnn", "cnn_baseline"}:
        return CNNBaseline(num_classes=num_classes, dropout=dropout)
    return build_model(name=name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def _get_image_path_from_dataset(ds: Any, idx: int) -> str:
    """
    Tries multiple ways to get original filepath for a given sample.
    Your PlantVillageDataset likely stores a table of samples.
    """
    # common patterns
    if hasattr(ds, "get_filepath"):
        return str(ds.get_filepath(idx))
    if hasattr(ds, "samples"):
        s = ds.samples[idx]
        if isinstance(s, dict) and "filepath" in s:
            return str(s["filepath"])
        if isinstance(s, (tuple, list)) and len(s) >= 1:
            return str(s[0])
    if hasattr(ds, "df") and "filepath" in getattr(ds, "df").columns:
        return str(ds.df.iloc[idx]["filepath"])

    raise AttributeError("Cannot find filepath in dataset. Add ds.get_filepath(i) or ds.samples[i]['filepath'].")


@torch.no_grad()
def save_misclassified_gallery(
    config_path: str,
    checkpoint_path: str,
    split: str,
    out_dir: Path,
    max_images: int = 25,
    cols: int = 5,
) -> Path:
    """
    Runs model on the given split and saves a grid of misclassified samples.
    """
    cfg = _load_config(config_path)
    num_classes = _infer_num_classes()
    labels = _load_label_names() or [str(i) for i in range(num_classes)]

    prefer_mps = bool(cfg.get("device", {}).get("prefer_mps", True))
    device = get_device(prefer_mps=prefer_mps)

    # dataset/dataloader
    img_size = int(cfg["data"].get("img_size", 224))
    batch_size = int(cfg["training"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 0))
    split_csv = cfg["data"]["split_csv"]

    tfm = build_eval_transforms(img_size=img_size)
    ds = PlantVillageDataset(split=split, csv_path=split_csv, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # model
    model = _build_model_from_cfg(cfg, num_classes=num_classes).to(device)
    ckpt = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # collect misclassified indices with (true, pred)
    mis: List[Tuple[int, int, int]] = []  # (global_idx, true, pred)
    seen = 0
    for images, y in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_cpu = y.numpy()

        for i in range(len(y_cpu)):
            if preds[i] != y_cpu[i]:
                mis.append((seen + i, int(y_cpu[i]), int(preds[i])))
                if len(mis) >= max_images:
                    break
        if len(mis) >= max_images:
            break
        seen += len(y_cpu)

    if len(mis) == 0:
        raise ValueError(f"No misclassified samples found in split='{split}'. (Model may be perfect on this split.)")

    # plot grid
    rows = int(np.ceil(len(mis) / cols))
    fig_w = cols * 3
    fig_h = rows * 3
    out_path = out_dir / f"misclassified_{split}.png"

    plt.figure(figsize=(fig_w, fig_h))
    for j, (idx, t, p) in enumerate(mis, start=1):
        img_path = _get_image_path_from_dataset(ds, idx)
        img = Image.open(img_path).convert("RGB")

        plt.subplot(rows, cols, j)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"T:{labels[t]}\nP:{labels[p]}", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate plots for PlantVillage project.")
    p.add_argument("--history", type=str, default="reports/history.json", help="Path to history.json")
    p.add_argument("--cm", type=str, default="reports/test_confusion_matrix.npy", help="Path to confusion matrix .npy")
    p.add_argument("--normalize_cm", action="store_true", help="Normalize confusion matrix rows")
    p.add_argument("--make_gallery", action="store_true", help="Also generate misclassified gallery (needs config+ckpt).")

    # Only needed if --make_gallery
    p.add_argument("--config", type=str, default="configs/resnet18.yaml", help="Config YAML for gallery inference")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for gallery inference")
    p.add_argument("--split", type=str, default="test", help="Split for gallery: test/val/train")
    p.add_argument("--max_images", type=int, default=25, help="Max images in gallery")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    out_dir = _ensure_reports_figures()

    # 1) curves
    history_path = (root / args.history).resolve() if not Path(args.history).is_absolute() else Path(args.history)
    if history_path.exists():
        p1 = plot_training_curves(history_path, out_dir)
        print(f"Saved: {p1}")
    else:
        print(f"Skip curves: history not found at {history_path}")

    # 2) confusion matrix
    cm_path = (root / args.cm).resolve() if not Path(args.cm).is_absolute() else Path(args.cm)
    if cm_path.exists():
        p2 = plot_confusion_matrix(cm_path, out_dir, normalize=args.normalize_cm)
        print(f"Saved: {p2}")
    else:
        print(f"Skip CM: file not found at {cm_path}")

    # 3) misclassified gallery (optional)
    if args.make_gallery:
        exp_ckpt = args.checkpoint
        if exp_ckpt is None:
            # default to reports/checkpoints/<experiment>_best.pt
            cfg = _load_config(args.config)
            exp_name = cfg["experiment"]["name"]
            exp_ckpt = str(root / "reports" / "checkpoints" / f"{exp_name}_best.pt")

        p3 = save_misclassified_gallery(
            config_path=args.config,
            checkpoint_path=exp_ckpt,
            split=args.split,
            out_dir=out_dir,
            max_images=args.max_images,
        )
        print(f"Saved: {p3}")


if __name__ == "__main__":
    main()
