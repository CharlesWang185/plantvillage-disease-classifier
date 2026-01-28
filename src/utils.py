# src/utils.py
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


PathLike = Union[str, Path]


# -----------------------------
# Paths / filesystem utilities
# -----------------------------
def project_root() -> Path:
    """
    Returns the project root assuming this file is at: <root>/src/utils.py
    """
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: PathLike) -> Path:
    """
    Create a directory if it doesn't exist. Returns Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: PathLike) -> Path:
    """
    Convert a path-like input to an absolute Path (without checking existence).
    """
    return Path(path).expanduser().resolve()


# -----------------------------
# Reproducibility (seeding)
# -----------------------------
def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    deterministic=True can slow things down and may restrict certain ops.
    It's optional; for most coursework/projects deterministic=False is fine.

    On Apple Silicon (MPS), full determinism is not always guaranteed,
    but seeding still improves repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if CUDA not available

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# -----------------------------
# Device selection (Mac MPS)
# -----------------------------
def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Returns the best available device.
    Priority:
      1) CUDA (if available, usually not on Apple Silicon)
      2) MPS (Apple Silicon GPU)
      3) CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def device_str(device: Optional[torch.device] = None) -> str:
    """
    Human-friendly device name for logging.
    """
    d = device or get_device()
    return str(d)


# -----------------------------
# JSON helpers (configs/results)
# -----------------------------
def _to_jsonable(obj: Any) -> Any:
    """
    Convert common Python objects to JSON-serializable forms.
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json(path: PathLike, data: Any, indent: int = 2) -> None:
    """
    Save JSON with safe conversion for numpy, torch, Path, dataclasses.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=_to_jsonable)


def load_json(path: PathLike) -> Any:
    """
    Load JSON file.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Model checkpoint helpers
# -----------------------------
def save_checkpoint(path: PathLike, payload: Dict[str, Any]) -> None:
    """
    Save a PyTorch checkpoint dict.
    Typical payload keys:
      - 'model_state'
      - 'optimizer_state'
      - 'epoch'
      - 'best_metric'
      - 'config'
      - 'label_map'
    """
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(payload, p)


def load_checkpoint(path: PathLike, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint dict.
    """
    p = Path(path)
    if map_location is None:
        map_location = get_device()
    return torch.load(p, map_location=map_location)


# -----------------------------
# Small logging helper
# -----------------------------
def print_once(msg: str) -> None:
    """
    Simple wrapper so later you can swap logging without editing many files.
    """
    print(msg)
