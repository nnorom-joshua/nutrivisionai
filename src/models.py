"""
src/models.py
Model factory for food recognition.
Supports EfficientNet-B0 (primary), ResNet-50, MobileNetV3 as candidates.
Each model gets a custom classification head with dropout.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import numpy as np
import torch

torch.serialization.add_safe_globals([np.core.multiarray.scalar])
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    NUM_CLASSES, DROPOUT_RATE, HIDDEN_DIM,
    SELECTED_MODEL, MODEL_CANDIDATES, FREEZE_EPOCHS
)

log = logging.getLogger(__name__)


# ─── Custom Classification Head ───────────────────────────────────────────────
class ClassificationHead(nn.Module):
    """
    Two-layer MLP head: GlobalPool → Dropout → FC(hidden) → BN → ReLU → Dropout → FC(n_classes)
    Replaces the default head of any backbone.
    """
    def __init__(self, in_features: int, n_classes: int,
                 hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT_RATE):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(hidden_dim, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─── Model Wrappers ───────────────────────────────────────────────────────────
class FoodClassifier(nn.Module):
    """
    Unified food classifier wrapping any supported backbone.
    Call freeze_backbone() / unfreeze_backbone() for two-phase training.
    """
    SUPPORTED = {
        "efficientnet_b0":        ("efficientnet_b0",        "classifier.1"),
        "resnet50":                ("resnet50",                "fc"),
        "mobilenetv3_large_100":   ("mobilenet_v3_large",      "classifier.3"),
    }

    def __init__(
        self,
        arch: str          = SELECTED_MODEL,
        n_classes: int     = NUM_CLASSES,
        hidden_dim: int    = HIDDEN_DIM,
        dropout: float     = DROPOUT_RATE,
        pretrained: bool   = True,
    ):
        super().__init__()
        assert arch in self.SUPPORTED, f"Unknown arch: {arch}. Choose from {list(self.SUPPORTED)}"

        self.arch      = arch
        self.n_classes = n_classes

        # ── Load backbone ─────────────────────────────────────────────────
        weights_arg = "DEFAULT" if pretrained else None
        backbone_fn, head_attr = self.SUPPORTED[arch]

        if arch == "efficientnet_b0":
            self.backbone = tv_models.efficientnet_b0(weights=weights_arg)
            in_features   = self.backbone.classifier[1].in_features
            self.backbone.classifier = ClassificationHead(in_features, n_classes, hidden_dim, dropout)

        elif arch == "resnet50":
            self.backbone = tv_models.resnet50(weights=weights_arg)
            in_features   = self.backbone.fc.in_features
            self.backbone.fc = ClassificationHead(in_features, n_classes, hidden_dim, dropout)

        elif arch == "mobilenetv3_large_100":
            self.backbone = tv_models.mobilenet_v3_large(weights=weights_arg)
            in_features   = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = ClassificationHead(in_features, n_classes, hidden_dim, dropout)

        log.info(f"[Model] {arch} | in_features={in_features} | classes={n_classes} | dropout={dropout}")

    def freeze_backbone(self):
        """Freeze all backbone layers except the custom head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False
        n_frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        log.info(f"[Model] Backbone frozen ({n_frozen} params frozen).")

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        log.info("[Model] Backbone unfrozen for fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self(x), dim=1)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (top1_class_idx, top1_confidence)."""
        probs = self.predict_proba(x)
        conf, cls = probs.max(dim=1)
        return cls, conf

    def predict_top_k(self, x: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return top-k class indices and their confidence scores."""
        probs = self.predict_proba(x)
        return probs.topk(k, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ─── Label Smoothing Loss ─────────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    Reduces overconfidence — especially useful for Food-101 inter-class ambiguity.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        # Hard label component
        nll       = F.nll_loss(log_probs, targets, reduction="mean")
        # Smooth component
        smooth    = -log_probs.mean(dim=1).mean()
        loss      = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss


# ─── Model Factory ────────────────────────────────────────────────────────────
def build_model(
    arch: str       = SELECTED_MODEL,
    n_classes: int  = NUM_CLASSES,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float  = DROPOUT_RATE,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> FoodClassifier:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FoodClassifier(
        arch=arch, n_classes=n_classes,
        hidden_dim=hidden_dim, dropout=dropout,
        pretrained=pretrained,
    ).to(device)

    params = model.count_parameters()
    log.info(f"[Model] Built {arch}: "
             f"total={params['total']:,} | trainable={params['trainable']:,}")
    return model
import os
print("=== CHECKPOINT DEBUG ===", flush=True)
print("Path:", checkpoint_path, flush=True)
print("Exists:", os.path.exists(checkpoint_path), flush=True)
print("Size (bytes):", os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else "N/A", flush=True)
try:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
except Exception as e:
    print("=== TORCH.LOAD FAILED ===", flush=True)
    print("Exception type:", type(e).__name__, flush=True)
    print("Exception message:", repr(e), flush=True)
    raise

def load_model(
    checkpoint_path: Path,
    arch: str       = SELECTED_MODEL,
    n_classes: int  = NUM_CLASSES,
    hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> FoodClassifier:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    resolved_hidden_dim = hidden_dim if hidden_dim is not None else saved_cfg.get("hidden_dim", HIDDEN_DIM)
    resolved_dropout    = dropout    if dropout    is not None else saved_cfg.get("dropout", DROPOUT_RATE)

    model = build_model(arch=arch, n_classes=n_classes,
                         hidden_dim=resolved_hidden_dim, dropout=resolved_dropout,
                         pretrained=False, device=device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    log.info(f"[Model] Loaded checkpoint from {checkpoint_path} "
             f"(hidden_dim={resolved_hidden_dim}, dropout={resolved_dropout})")
    return model


if __name__ == "__main__":
    # Smoke test
    device = torch.device("cpu")
    for arch in MODEL_CANDIDATES:
        m = build_model(arch=arch, device=device)
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        print(f"{arch}: output shape = {out.shape} ✓")
        params = m.count_parameters()
        print(f"  Params: {params}")
