# src/train.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.utils import (
    ensure_dir,
    get_device,
    load_json,
    project_root,
    save_checkpoint,
    save_json,
    seed_everything,
)
from src.data.dataset import PlantVillageDataset
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.metrics import evaluate_predictions

from src.models.cnn_baseline import CNNBaseline
from src.models.transfer import build_model, freeze_backbone, unfreeze_all, unfreeze_last_n_children


# -----------------------------
# Config
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def infer_num_classes(cfg: Dict[str, Any]) -> int:
    """
    Prefer label_map.json if available; else infer from split.csv.
    """
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


# -----------------------------
# DataLoaders
# -----------------------------
def make_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    img_size = int(cfg["data"].get("img_size", 224))
    batch_size = int(cfg["training"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 0))  # macOS often safest at 0 or small
    split_csv = cfg["data"]["split_csv"]

    train_tfms = build_train_transforms(img_size=img_size)
    eval_tfms = build_eval_transforms(img_size=img_size)

    train_ds = PlantVillageDataset(split="train", csv_path=split_csv, transform=train_tfms)
    val_ds = PlantVillageDataset(split="val", csv_path=split_csv, transform=eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't benefit like CUDA
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


# -----------------------------
# Model
# -----------------------------
def build_from_config(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    name = cfg["model"]["name"].lower().strip()
    pretrained = bool(cfg["model"].get("pretrained", True))
    dropout = float(cfg["model"].get("dropout", 0.0))

    if name in {"baseline_cnn", "cnn_baseline"}:
        return CNNBaseline(num_classes=num_classes, dropout=dropout)

    # transfer models
    return build_model(name=name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def apply_freeze_strategy(cfg: Dict[str, Any], model: nn.Module, phase: str) -> None:
    """
    phase: "head" or "finetune"
    """
    strategy = cfg.get("freeze", {}).get("strategy", "none")
    last_n = int(cfg.get("freeze", {}).get("unfreeze_last_n_children", 2))

    if strategy == "none":
        unfreeze_all(model)
        return

    if phase == "head":
        # Freeze everything except classifier head
        freeze_backbone(model, keep_classifier_trainable=True)
        return

    # phase == "finetune"
    if strategy == "all":
        unfreeze_all(model)
    elif strategy == "last_n_children":
        unfreeze_last_n_children(model, n=last_n)
    elif strategy == "head_then_last_n":
        # common approach: freeze all then unfreeze last N children (includes head too)
        unfreeze_last_n_children(model, n=last_n)
    else:
        # fallback: unfreeze all
        unfreeze_all(model)


# -----------------------------
# Training & evaluation
# -----------------------------
def make_optimizer(cfg: Dict[str, Any], model: nn.Module, phase: str) -> torch.optim.Optimizer:
    opt_name = cfg["training"].get("optimizer", "adamw").lower().strip()
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))

    if phase == "head":
        lr = float(cfg["training"].get("lr_head", 1e-3))
    else:
        lr = float(cfg["training"].get("lr_finetune", 1e-4))

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {opt_name}")


def make_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer, total_epochs: int):
    sched_cfg = cfg.get("scheduler", {}) or {}
    stype = (sched_cfg.get("type") or "none").lower().strip()

    if stype == "none":
        return None

    if stype == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    raise ValueError(f"Unknown scheduler type: {stype}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
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
        topk=3,
        num_classes=num_classes,
    )

    return {
        "val_loss": avg_loss,
        "summary": metrics_out["summary"],
        "topk_accuracy": metrics_out.get("topk_accuracy", None),
    }


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PlantVillage classifier (baseline + transfer learning).")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config, e.g. configs/resnet18.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["experiment"].get("seed", 42))
    seed_everything(seed)

    prefer_mps = bool(cfg.get("device", {}).get("prefer_mps", True))
    device = get_device(prefer_mps=prefer_mps)
    print(f"Device: {device}")

    # infer num classes (don’t trust hard-coded value)
    num_classes = infer_num_classes(cfg)
    print(f"Num classes inferred: {num_classes}")

    # data
    train_loader, val_loader = make_loaders(cfg)

    # model
    model = build_from_config(cfg, num_classes=num_classes)
    model.to(device)

    # loss
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # outputs
    out_dir = project_root() / (cfg.get("logging", {}).get("output_dir", "reports"))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    exp_name = cfg["experiment"]["name"]

    best_ckpt_path = ckpt_dir / f"{exp_name}_best.pt"
    last_ckpt_path = ckpt_dir / f"{exp_name}_last.pt"
    history_path = out_dir / "history.json"
    results_path = out_dir / "results.json"

    history: List[Dict[str, Any]] = []

    # training plan
    epochs_head = int(cfg["training"].get("epochs_head", 0))
    epochs_finetune = int(cfg["training"].get("epochs_finetune", 0))

    early_metric = (cfg["training"].get("early_stopping_metric") or "macro_f1").lower().strip()
    patience = int(cfg["training"].get("early_stopping_patience", 5))

    best_score = -1e9
    bad_epochs = 0

    def score_from_val(val_out: Dict[str, Any]) -> float:
        # We primarily early-stop on macro_f1 (recommended).
        if early_metric == "macro_f1":
            return float(val_out["summary"]["macro_f1"])
        if early_metric == "accuracy":
            return float(val_out["summary"]["accuracy"])
        # fallback to macro_f1
        return float(val_out["summary"]["macro_f1"])

    # ---- Phase 1: head training ----
    if epochs_head > 0:
        print(f"\n=== Phase 1: Train head only ({epochs_head} epochs) ===")
        apply_freeze_strategy(cfg, model, phase="head")
        optimizer = make_optimizer(cfg, model, phase="head")
        scheduler = make_scheduler(cfg, optimizer, total_epochs=epochs_head)

        for epoch in range(1, epochs_head + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
            val_out = evaluate(model, val_loader, device, criterion, num_classes=num_classes)
            if scheduler is not None:
                scheduler.step()

            score = score_from_val(val_out)
            lr_now = float(optimizer.param_groups[0]["lr"])
            dt = time.time() - t0

            row = {
                "phase": "head",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_out["val_loss"],
                "val_summary": val_out["summary"],
                "val_topk": val_out.get("topk_accuracy", None),
                "lr": lr_now,
                "seconds": dt,
            }
            history.append(row)

            print(
                f"[head][{epoch}/{epochs_head}] "
                f"train_loss={train_loss:.4f} val_loss={val_out['val_loss']:.4f} "
                f"acc={val_out['summary']['accuracy']:.4f} macro_f1={val_out['summary']['macro_f1']:.4f} "
                f"lr={lr_now:.2e}"
            )

            # Save last checkpoint each epoch
            save_checkpoint(
                last_ckpt_path,
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "phase": "head",
                    "best_score": best_score,
                    "config": cfg,
                    "num_classes": num_classes,
                },
            )

            # Save best
            if score > best_score:
                best_score = score
                bad_epochs = 0
                save_checkpoint(
                    best_ckpt_path,
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "phase": "head",
                        "best_score": best_score,
                        "config": cfg,
                        "num_classes": num_classes,
                    },
                )
            else:
                bad_epochs += 1

            # Early stop during head phase (optional but ok)
            if bad_epochs >= patience:
                print(f"Early stopping triggered in head phase (patience={patience}).")
                break

    # reset early-stopping counter for finetune phase
    bad_epochs = 0

    # ---- Phase 2: fine-tuning ----
    if epochs_finetune > 0:
        print(f"\n=== Phase 2: Fine-tune ({epochs_finetune} epochs) ===")
        apply_freeze_strategy(cfg, model, phase="finetune")
        optimizer = make_optimizer(cfg, model, phase="finetune")
        scheduler = make_scheduler(cfg, optimizer, total_epochs=epochs_finetune)

        for epoch in range(1, epochs_finetune + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
            val_out = evaluate(model, val_loader, device, criterion, num_classes=num_classes)
            if scheduler is not None:
                scheduler.step()

            score = score_from_val(val_out)
            lr_now = float(optimizer.param_groups[0]["lr"])
            dt = time.time() - t0

            row = {
                "phase": "finetune",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_out["val_loss"],
                "val_summary": val_out["summary"],
                "val_topk": val_out.get("topk_accuracy", None),
                "lr": lr_now,
                "seconds": dt,
            }
            history.append(row)

            print(
                f"[finetune][{epoch}/{epochs_finetune}] "
                f"train_loss={train_loss:.4f} val_loss={val_out['val_loss']:.4f} "
                f"acc={val_out['summary']['accuracy']:.4f} macro_f1={val_out['summary']['macro_f1']:.4f} "
                f"lr={lr_now:.2e}"
            )

            save_checkpoint(
                last_ckpt_path,
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "phase": "finetune",
                    "best_score": best_score,
                    "config": cfg,
                    "num_classes": num_classes,
                },
            )

            if score > best_score:
                best_score = score
                bad_epochs = 0
                save_checkpoint(
                    best_ckpt_path,
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "phase": "finetune",
                        "best_score": best_score,
                        "config": cfg,
                        "num_classes": num_classes,
                    },
                )
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                print(f"Early stopping triggered in finetune phase (patience={patience}).")
                break

    # Save history & final results
    save_json(history_path, history)

    # Find best row in history by metric
    best_row = None
    best_row_score = -1e9
    for row in history:
        s = row["val_summary"]["macro_f1"] if early_metric == "macro_f1" else row["val_summary"]["accuracy"]
        if s > best_row_score:
            best_row_score = s
            best_row = row

    results = {
        "experiment": cfg["experiment"]["name"],
        "device": str(device),
        "num_classes": num_classes,
        "best_metric": early_metric,
        "best_score": float(best_score),
        "best_row": best_row,
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
    }
    save_json(results_path, results)

    print("\nTraining complete.")
    print(f"Saved history: {history_path}")
    print(f"Saved results: {results_path}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
