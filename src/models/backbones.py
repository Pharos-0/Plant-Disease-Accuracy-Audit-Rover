"""Backbone factory.

All four architectures come from timm with ImageNet weights, exposing a common
interface so the caching stage can pull logits and penultimate features without
special-casing the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "vit_small_patch16": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}

# 224 for every backbone; ViT-S/16 requires exactly this.
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelBundle:
    model: nn.Module
    name: str
    timm_id: str
    n_features: int
    n_classes: int


def create_model(name: str, n_classes: int, *, pretrained: bool = True) -> ModelBundle:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; expected one of {sorted(BACKBONES)}")
    timm_id = BACKBONES[name]
    model = timm.create_model(timm_id, pretrained=pretrained, num_classes=n_classes)
    n_features = model.num_features
    return ModelBundle(
        model=model, name=name, timm_id=timm_id, n_features=n_features, n_classes=n_classes
    )


@torch.no_grad()
def forward_features_and_logits(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (penultimate features, logits) in one pass.

    ``forward_head(..., pre_logits=True)`` is timm's uniform way to reach the
    pooled penultimate representation across CNNs and ViTs alike.
    """
    feats = model.forward_features(x)
    pooled = model.forward_head(feats, pre_logits=True)
    logits = model.get_classifier()(pooled)
    return pooled, logits
