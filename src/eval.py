# src/eval.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.utils import get_device, load_json, project_root, save_json, ensure_dir
from src.data.dataset import PlantVillageDataset
from src.data.transforms import build_eval_transforms
from src.metrics import evaluate_predictions

from src.models.cnn_baseline import CNNBaseline
from src.models.transfer import build_model


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_num_classes(cfg: Dict[str, Any]) -> int:
    root = project_root()
    label_map_path = root / "data" / "splits" / "label_map.json"
    if label_map_path.exists():
        label_map = load_json(label_map_path)
        return len(label_map)

    split_csv = root / cfg["data"]["split_csv"]
    if not split_csv.exists():
        raise FileNotFoundError(f"split.csv not found: {split_csv}")
    import pandas as pd

    df = pd.read_csv(split_csv)
    return int(df["label_idx"].nunique())


def build_from_config(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    name = cfg["model"]["name"].lower().strip()
    pretrained = bool(cfg["model"].get("pretrained", True))
    dropout = float(cfg["model"].get("dropout", 0.0))

    if name in {"baseline_cnn", "cnn_baseline"}:
        return CNNBaseline(num_classes=num_classes, dropout=dropout)

    return build_model(name=name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)


@torch.no_grad()
def run_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
    topk: int = 3,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n = 0

    y_true: List[int] = []
    y_pred: List[int] = []
    y_logits_list: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_logits_list.append(logits.detach().cpu().numpy())

    avg_loss = total_loss / max(n, 1)
    y_logits = np.concatenate(y_logits_list, axis=0) if y_logits_list else None

    metrics_out = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_logits=y_logits,
        class_names=None,
        topk=topk,
        num_classes=num_classes,
    )

    return {
        "test_loss": avg_loss,
        "summary": metrics_out["summary"],
        "topk_accuracy": metrics_out.get("topk_accuracy", None),
        "per_class_df": metrics_out["per_class_df"],
        "confusion_matrix": metrics_out["confusion_matrix"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained PlantVillage model on test split.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (same one used in training).")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to checkpoint .pt. If not set, uses reports/checkpoints/<exp>_best.pt",
    )
    p.add_argument("--split", type=str, default="test", help="Which split to evaluate: test (default), val, train.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    prefer_mps = bool(cfg.get("device", {}).get("prefer_mps", True))
    device = get_device(prefer_mps=prefer_mps)
    print(f"Device: {device}")

    num_classes = infer_num_classes(cfg)
    print(f"Num classes inferred: {num_classes}")

    # Build model
    model = build_from_config(cfg, num_classes=num_classes)
    model.to(device)

    # Choose checkpoint path
    root = project_root()
    exp_name = cfg["experiment"]["name"]
    default_ckpt = root / "reports" / "checkpoints" / f"{exp_name}_best.pt"
    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else default_ckpt

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError("Checkpoint missing key 'model_state'.")
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")

    # Data
    img_size = int(cfg["data"].get("img_size", 224))
    batch_size = int(cfg["training"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 0))
    split_csv = cfg["data"]["split_csv"]

    eval_tfms = build_eval_transforms(img_size=img_size)
    ds = PlantVillageDataset(split=args.split, csv_path=split_csv, transform=eval_tfms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # Loss (match training)
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Run evaluation
    out = run_test(model, loader, device, criterion, num_classes=num_classes, topk=3)

    print(
        f"\nTest results ({args.split}): "
        f"loss={out['test_loss']:.4f} "
        f"acc={out['summary']['accuracy']:.4f} "
        f"macro_f1={out['summary']['macro_f1']:.4f} "
        f"top3={out['topk_accuracy'] if out['topk_accuracy'] is not None else 'n/a'}"
    )

    # Save artifacts
    reports_dir = ensure_dir(root / "reports")
    figures_dir = ensure_dir(reports_dir / "figures")
    _ = figures_dir  # reserved for later visualize.py outputs

    # Save JSON summary
    save_json(reports_dir / "test_results.json", {
        "config": cfg,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "test_loss": out["test_loss"],
        "summary": out["summary"],
        "topk_accuracy": out["topk_accuracy"],
    })

    # Save per-class CSV
    per_class_df = out["per_class_df"]
    per_class_df.to_csv(reports_dir / "test_per_class.csv", index=True)

    # Save confusion matrix (npy)
    cm = out["confusion_matrix"]
    np.save(reports_dir / "test_confusion_matrix.npy", cm)

    print(f"Saved: {reports_dir / 'test_results.json'}")
    print(f"Saved: {reports_dir / 'test_per_class.csv'}")
    print(f"Saved: {reports_dir / 'test_confusion_matrix.npy'}")


if __name__ == "__main__":
    main()
