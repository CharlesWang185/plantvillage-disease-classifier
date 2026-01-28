# src/explain/gradcam.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import yaml
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.utils import get_device, project_root, ensure_dir, load_json
from src.data.dataset import PlantVillageDataset
from src.models.cnn_baseline import CNNBaseline
from src.models.transfer import build_model


# -----------------------------
# Helpers: label names
# -----------------------------
def load_label_names() -> Optional[List[str]]:
    """
    Loads data/splits/label_map.json if it exists.
    Expected format: { "ClassName": idx, ... }
    Returns list where index -> class name.
    """
    root = project_root()
    p = root / "data" / "splits" / "label_map.json"
    if not p.exists():
        return None
    label_map = load_json(p)  # name -> idx
    inv = [None] * (max(label_map.values()) + 1)
    for name, idx in label_map.items():
        inv[idx] = name
    return [x if x is not None else f"class_{i}" for i, x in enumerate(inv)]


# -----------------------------
# Model building/loading
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_num_classes() -> int:
    labels = load_label_names()
    if labels is None:
        raise FileNotFoundError("data/splits/label_map.json not found; cannot infer num_classes.")
    return len(labels)


def build_from_config(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    name = cfg["model"]["name"].lower().strip()
    pretrained = bool(cfg["model"].get("pretrained", True))
    dropout = float(cfg["model"].get("dropout", 0.0))

    if name in {"baseline_cnn", "cnn_baseline"}:
        return CNNBaseline(num_classes=num_classes, dropout=dropout)

    # transfer model
    return build_model(name=name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def load_checkpoint_into(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError("Checkpoint missing key 'model_state'.")
    model.load_state_dict(ckpt["model_state"], strict=True)


# -----------------------------
# Target layer selection
# -----------------------------
def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """
    Get a submodule via dotted path, e.g.:
      - "layer4"
      - "layer4.1.conv2"
      - "features"
      - "features.7"
    """
    cur: nn.Module = model
    if not name:
        raise ValueError("target_layer name is empty.")
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]  # type: ignore[index]
        else:
            cur = getattr(cur, part)
    return cur


def default_target_layer_name(model_name: str) -> str:
    """
    Sensible defaults:
      - ResNet*: layer4
      - EfficientNet/MobileNet: features
      - Baseline CNN: features (we'll define it in CNNBaseline if exists; else try 'backbone')
    """
    mn = model_name.lower()
    if "resnet" in mn:
        return "layer4"
    if "efficientnet" in mn or "mobilenet" in mn:
        return "features"
    # fallback
    return "layer4"


# -----------------------------
# Preprocess for Grad-CAM (match eval geometry)
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_eval_geometry(img_size: int = 224, resize_shorter: int = 256):
    """
    Resize shorter side to 256, then center crop 224 (standard ImageNet eval).
    Returns a transform that outputs PIL image (not tensor).
    """
    return transforms.Compose(
        [
            transforms.Resize(resize_shorter, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
        ]
    )


def make_tensor_normalize():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# -----------------------------
# Grad-CAM core
# -----------------------------
@dataclass
class GradCAMResult:
    heatmap: np.ndarray  # (H, W), 0..1
    pred_idx: int
    pred_prob: float
    target_idx: int
    target_prob: float


class GradCAM:
    """
    Vanilla Grad-CAM for CNN backbones.
    Works for ResNet/EfficientNet/MobileNet and most conv nets.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        self._hook_a = target_layer.register_forward_hook(self._forward_hook)
        self._hook_g = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inp, out):
        # out: (B, C, H, W)
        self._activations = out.detach()

    def _backward_hook(self, module: nn.Module, grad_input, grad_output):
        # grad_output[0]: (B, C, H, W)
        self._grads = grad_output[0].detach()

    def close(self):
        self._hook_a.remove()
        self._hook_g.remove()

    def __call__(self, x: torch.Tensor, target_idx: Optional[int] = None) -> GradCAMResult:
        """
        x: (1, 3, H, W) normalized tensor
        target_idx:
          - None -> use predicted class
          - int  -> compute Grad-CAM for that class
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)

        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_idx].item())

        if target_idx is None:
            target_idx = pred_idx

        target_prob = float(probs[0, target_idx].item())

        # Backprop score of target class
        score = logits[0, target_idx]
        score.backward(retain_graph=False)

        if self._activations is None or self._grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # activations/grads: (1, C, h, w)
        A = self._activations
        G = self._grads

        # weights: global-average-pool gradients over spatial dims -> (1, C)
        weights = G.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * A).sum(dim=1, keepdim=False)  # (1, h, w)
        cam = torch.relu(cam)

        # Normalize to 0..1
        cam_np = cam[0].cpu().numpy()
        cam_np -= cam_np.min()
        denom = cam_np.max() + 1e-8
        cam_np = cam_np / denom

        return GradCAMResult(
            heatmap=cam_np,
            pred_idx=pred_idx,
            pred_prob=pred_prob,
            target_idx=int(target_idx),
            target_prob=target_prob,
        )


# -----------------------------
# Visualization: overlay heatmap on image
# -----------------------------
def heatmap_to_rgba(heatmap: np.ndarray) -> np.ndarray:
    """
    Convert (H,W) heatmap 0..1 to RGBA using matplotlib colormap.
    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap("jet")
    rgba = cmap(heatmap)  # (H,W,4) float 0..1
    return (rgba * 255).astype(np.uint8)


def overlay_heatmap_on_image(
    base_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """
    base_pil: PIL RGB, size (W,H)
    heatmap: (H,W) 0..1 at same spatial size as base_pil
    """
    if heatmap.shape[0] != base_pil.size[1] or heatmap.shape[1] != base_pil.size[0]:
        # resize heatmap to image size
        heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(base_pil.size, resample=Image.BILINEAR)
        heatmap = np.array(heat).astype(np.float32) / 255.0

    hm_rgba = Image.fromarray(heatmap_to_rgba(heatmap)).convert("RGBA")
    base_rgba = base_pil.convert("RGBA")

    blended = Image.blend(base_rgba, hm_rgba, alpha=alpha).convert("RGB")
    return blended


# -----------------------------
# Run Grad-CAM on samples
# -----------------------------
@torch.no_grad()
def predict_label(model: nn.Module, x: torch.Tensor) -> Tuple[int, float]:
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    idx = int(torch.argmax(probs, dim=1).item())
    prob = float(probs[0, idx].item())
    return idx, prob


def run_gradcam_on_indices(
    cfg: Dict[str, Any],
    model: nn.Module,
    ckpt_path: Path,
    split: str,
    indices: List[int],
    out_dir: Path,
    target_layer_name: str,
    alpha: float = 0.45,
) -> None:
    device = get_device(prefer_mps=bool(cfg.get("device", {}).get("prefer_mps", True)))
    model.to(device)
    model.eval()
    load_checkpoint_into(model, ckpt_path)

    labels = load_label_names()
    num_classes = infer_num_classes()
    if labels is None or len(labels) != num_classes:
        labels = [str(i) for i in range(num_classes)]

    img_size = int(cfg["data"].get("img_size", 224))

    # geometry for aligned overlay and tensor input
    geom = make_eval_geometry(img_size=img_size)
    to_tensor_norm = make_tensor_normalize()

    # dataset WITHOUT transforms (we will handle it ourselves to keep original filepath + aligned PIL)
    ds = PlantVillageDataset(split=split, csv_path=cfg["data"]["split_csv"], transform=None)

    # pick target layer
    target_layer = get_module_by_name(model, target_layer_name)
    cam = GradCAM(model, target_layer)

    for idx in indices:
        img_path = Path(ds.get_filepath(idx))  # you added this method (using samples[idx].filepath)
        true_name = ds.get_class_name(idx)
        true_idx = int(ds.samples[idx].label_idx)  # type: ignore[attr-defined]

        pil = Image.open(img_path).convert("RGB")
        pil_aligned = geom(pil)  # resized+center-cropped PIL

        x = to_tensor_norm(pil_aligned).unsqueeze(0).to(device)

        # We want gradients -> temporarily enable grad
        for p in model.parameters():
            p.requires_grad_(True)

        # Grad-CAM uses backward, so we can't be under torch.no_grad() for that part.
        # We'll do it manually:
        with torch.enable_grad():
            result = cam(x, target_idx=None)  # explain predicted class

        pred_name = labels[result.pred_idx]
        tgt_name = labels[result.target_idx]
        overlay = overlay_heatmap_on_image(pil_aligned, result.heatmap, alpha=alpha)

        # filename
        safe_true = true_name.replace("/", "_")
        safe_pred = pred_name.replace("/", "_")
        out_path = out_dir / f"idx{idx:05d}_T-{safe_true}_P-{safe_pred}_{result.pred_prob:.3f}.png"

        overlay.save(out_path)

    cam.close()


def choose_indices_auto(
    cfg: Dict[str, Any],
    model: nn.Module,
    ckpt_path: Path,
    split: str,
    num_images: int,
    seed: int = 42,
) -> List[int]:
    """
    Automatically choose a mixture of correct + incorrect examples:
      - try to collect ~50% wrong, ~50% correct (if possible)
    """
    rng = np.random.default_rng(seed)
    device = get_device(prefer_mps=bool(cfg.get("device", {}).get("prefer_mps", True)))

    model.to(device)
    model.eval()
    load_checkpoint_into(model, ckpt_path)

    num_classes = infer_num_classes()
    img_size = int(cfg["data"].get("img_size", 224))
    geom = make_eval_geometry(img_size=img_size)
    to_tensor_norm = make_tensor_normalize()

    ds = PlantVillageDataset(split=split, csv_path=cfg["data"]["split_csv"], transform=None)

    # sample a pool to search
    pool = list(range(len(ds)))
    rng.shuffle(pool)
    pool = pool[: min(len(pool), max(500, num_images * 50))]

    correct: List[int] = []
    wrong: List[int] = []

    with torch.no_grad():
        for idx in pool:
            img_path = Path(ds.get_filepath(idx))
            pil = Image.open(img_path).convert("RGB")
            pil_aligned = geom(pil)
            x = to_tensor_norm(pil_aligned).unsqueeze(0).to(device)

            pred_idx, _ = predict_label(model, x)
            true_idx = int(ds.samples[idx].label_idx)  # type: ignore[attr-defined]

            if pred_idx == true_idx:
                if len(correct) < num_images:
                    correct.append(idx)
            else:
                if len(wrong) < num_images:
                    wrong.append(idx)

            if len(correct) >= num_images and len(wrong) >= num_images:
                break

    # target mix
    want_wrong = num_images // 2
    want_correct = num_images - want_wrong

    chosen = wrong[:want_wrong] + correct[:want_correct]
    if len(chosen) < num_images:
        # fill from whatever we have
        rest = (wrong + correct)
        for i in rest:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= num_images:
                break

    return chosen[:num_images]


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Grad-CAM overlays for PlantVillage model.")
    p.add_argument("--config", type=str, required=True, help="Path to config YAML (e.g., configs/resnet18.yaml)")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt (default: reports/checkpoints/<exp>_best.pt)",
    )
    p.add_argument("--split", type=str, default="test", help="Split: test/val/train")
    p.add_argument("--out_dir", type=str, default="reports/figures/gradcam", help="Output directory for overlays")
    p.add_argument("--num_images", type=int, default=12, help="How many images to generate")
    p.add_argument(
        "--target_layer",
        type=str,
        default="",
        help="Target conv layer name (e.g. layer4). If empty, uses a default based on model name.",
    )
    p.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay strength (0..1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for choosing samples")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    root = project_root()
    out_dir = ensure_dir(root / args.out_dir)

    num_classes = infer_num_classes()
    model = build_from_config(cfg, num_classes=num_classes)

    exp_name = cfg["experiment"]["name"]
    default_ckpt = root / "reports" / "checkpoints" / f"{exp_name}_best.pt"
    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else default_ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_name = cfg["model"]["name"]
    target_layer_name = args.target_layer.strip() or default_target_layer_name(model_name)

    # Choose indices automatically (mix of correct + wrong if possible)
    indices = choose_indices_auto(cfg, model, ckpt_path, split=args.split, num_images=args.num_images, seed=args.seed)

    run_gradcam_on_indices(
        cfg=cfg,
        model=model,
        ckpt_path=ckpt_path,
        split=args.split,
        indices=indices,
        out_dir=out_dir,
        target_layer_name=target_layer_name,
        alpha=args.alpha,
    )

    print(f"Saved Grad-CAM overlays to: {out_dir}")
    print(f"Target layer: {target_layer_name}")
    print(f"Num images: {len(indices)}")


if __name__ == "__main__":
    main()
