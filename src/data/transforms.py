# src/data/transforms.py
from __future__ import annotations

from typing import Optional, Tuple

from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ImageNet normalization (required for pretrained ResNet/EfficientNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(
    img_size: int = 224,
    crop_scale: Tuple[float, float] = (0.8, 1.0),
    hflip_p: float = 0.5,
    rotation_deg: int = 15,
    color_jitter: Optional[Tuple[float, float, float, float]] = (0.2, 0.2, 0.2, 0.05),
) -> transforms.Compose:
    """
    Training transforms (with augmentation).

    Function:
    1) Randomly crop & resize to img_size (helps generalization)
    2) Random horizontal flip
    3) Small rotation
    4) Light color jitter (optional)
    5) Convert to tensor in [0, 1]
    6) Normalize with ImageNet mean/std (important for pretrained models)

    Parameters:
    - img_size: output size, e.g. 224
    - crop_scale: how much area to keep when cropping (0.8 to 1.0 keeps most of leaf)
    - hflip_p: probability of horizontal flip
    - rotation_deg: max rotation in degrees (+/- rotation_deg)
    - color_jitter: (brightness, contrast, saturation, hue) or None to disable
    """
    t = [
        transforms.RandomResizedCrop(
            size=img_size,
            scale=crop_scale,
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomRotation(degrees=rotation_deg, interpolation=InterpolationMode.BILINEAR),
    ]

    if color_jitter is not None:
        b, c, s, h = color_jitter
        t.append(transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))

    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(t)


def build_eval_transforms(
    img_size: int = 224,
    resize_shorter: int = 256,
) -> transforms.Compose:
    """
    Validation/Test transforms (NO randomness).

    Common standard:
      Resize the shorter side to 256, then center crop 224.

    This makes evaluation fair and repeatable.
    """
    return transforms.Compose(
        [
            transforms.Resize(resize_shorter, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
