"""
src/trainer.py
Complete training engine with:
- Two-phase training (freeze backbone → unfreeze for fine-tuning)
- Cosine annealing + ReduceLROnPlateau schedulers
- Early stopping with best-model checkpointing
- Mixed precision (AMP) when CUDA available
- TensorBoard-style JSON metrics logging
- Per-epoch confusion matrix and classification report
"""

import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, top_k_accuracy_score, classification_report, confusion_matrix
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    NUM_EPOCHS, EARLY_STOP_PAT, FREEZE_EPOCHS,
    CHECKPOINT_DIR, BEST_MODEL_PATH, LOGS_DIR, SEED
)
from src.models import FoodClassifier, LabelSmoothingCrossEntropy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PAT, min_delta: float = 1e-4, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = None
        self.counter    = 0
        self.triggered  = False

    def __call__(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best + self.min_delta) if self.mode == "max" \
                   else (value < self.best - self.min_delta)

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            log.info(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.triggered = True

        return self.triggered


# ─── Metrics Helper ────────────────────────────────────────────────────────────
def compute_metrics(all_labels: np.ndarray, all_preds: np.ndarray,
                    all_probs: np.ndarray, n_classes: int) -> Dict:
    """Compute full evaluation metrics."""
    acc   = accuracy_score(all_labels, all_preds)
    prec  = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec   = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1    = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        top5 = top_k_accuracy_score(all_labels, all_probs, k=5, labels=list(range(n_classes)))
    except Exception:
        top5 = 0.0

    return {
        "accuracy": round(acc * 100, 4),
        "top5_accuracy": round(top5 * 100, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
    }


# ─── One Epoch ────────────────────────────────────────────────────────────────
def run_epoch(
    model: FoodClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    scaler: Optional[GradScaler],
    training: bool = True,
) -> Tuple[float, Dict]:
    model.train() if training else model.eval()

    total_loss  = 0.0
    all_labels  = []
    all_preds   = []
    all_probs   = []
    n_batches   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            # Handle both 2-tuple and 3-tuple returns
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch

            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training and scaler is not None:
                with autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            probs = torch.softmax(logits.detach(), dim=1)
            preds = probs.argmax(dim=1)

            total_loss  += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            n_batches   += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics  = compute_metrics(
        np.array(all_labels), np.array(all_preds),
        np.array(all_probs), model.n_classes
    )
    metrics["loss"] = round(avg_loss, 6)
    return avg_loss, metrics


# ─── Trainer ──────────────────────────────────────────────────────────────────
class Trainer:
    """
    Full training pipeline for FoodClassifier.
    Two-phase: freeze (feature extraction) → unfreeze (fine-tuning).
    """

    def __init__(
        self,
        model: FoodClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        run_name: str = "run",
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.config       = config
        self.run_name     = run_name

        # ── Criterion ──────────────────────────────────────────────────────
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # ── History ────────────────────────────────────────────────────────
        self.history: List[Dict] = []
        self.best_val_acc = 0.0
        self.best_epoch   = 0

        # ── AMP Scaler ─────────────────────────────────────────────────────
        self.scaler = GradScaler() if device.type == "cuda" else None

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self, lr: float, wd: float, opt_name: str) -> optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))
        elif opt_name == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        else:
            return optim.AdamW(params, lr=lr, weight_decay=wd)

    def _build_scheduler(self, optimizer, sched_name: str, n_epochs: int):
        if sched_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
        elif sched_name == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, n_epochs // 3), gamma=0.3)
        elif sched_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.3, patience=3
            )
        return None

    def train(self) -> List[Dict]:
        cfg         = self.config
        n_epochs    = cfg.get("epochs",       NUM_EPOCHS)
        freeze_ep   = cfg.get("freeze_epochs", FREEZE_EPOCHS)
        lr_freeze   = cfg.get("lr",            1e-3)
        lr_finetune = cfg.get("lr",            1e-3) * 0.1
        wd          = cfg.get("weight_decay",  1e-4)
        opt_name    = cfg.get("optimizer",     "adamw")
        sched_name  = cfg.get("scheduler",    "cosine")
        patience    = cfg.get("patience",     EARLY_STOP_PAT)

        early_stop = EarlyStopping(patience=patience, mode="max")

        # ── Phase 1: Feature Extraction (frozen backbone) ──────────────────
        log.info(f"[Trainer] ── Phase 1: Feature Extraction ({freeze_ep} epochs) ──")
        self.model.freeze_backbone()
        optimizer = self._build_optimizer(lr_freeze, wd, opt_name)
        scheduler = self._build_scheduler(optimizer, sched_name, freeze_ep)

        for epoch in range(1, freeze_ep + 1):
            self._run_one_epoch(epoch, optimizer, scheduler, sched_name, early_stop)
            if early_stop.triggered:
                log.info("[Trainer] Early stopping triggered in Phase 1.")
                break

        # ── Phase 2: Fine-Tuning (all layers unfrozen) ────────────────────
        remaining = n_epochs - freeze_ep
        log.info(f"[Trainer] ── Phase 2: Fine-Tuning ({remaining} epochs) ──")
        self.model.unfreeze_backbone()
        early_stop = EarlyStopping(patience=patience, mode="max")  # reset
        optimizer  = self._build_optimizer(lr_finetune, wd, opt_name)
        scheduler  = self._build_scheduler(optimizer, sched_name, remaining)

        for epoch in range(freeze_ep + 1, n_epochs + 1):
            self._run_one_epoch(epoch, optimizer, scheduler, sched_name, early_stop)
            if early_stop.triggered:
                log.info("[Trainer] Early stopping triggered in Phase 2.")
                break

        # ── Save metrics log ──────────────────────────────────────────────
        log_path = LOGS_DIR / f"{self.run_name}_history.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        log.info(f"[Trainer] Training history saved to {log_path}")
        log.info(f"[Trainer] Best val accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")

        return self.history

    def _run_one_epoch(self, epoch, optimizer, scheduler, sched_name, early_stop):
        t0 = time.time()

        _, train_m = run_epoch(
            self.model, self.train_loader, self.criterion,
            optimizer, self.device, self.scaler, training=True
        )
        _, val_m   = run_epoch(
            self.model, self.val_loader, self.criterion,
            None, self.device, None, training=False
        )

        # ── LR step ─────────────────────────────────────────────────────
        if scheduler is not None:
            if sched_name == "plateau":
                scheduler.step(val_m["accuracy"])
            else:
                scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = {
            "epoch":          epoch,
            "lr":             lr_now,
            "train_loss":     train_m["loss"],
            "train_acc":      train_m["accuracy"],
            "train_top5":     train_m["top5_accuracy"],
            "val_loss":       val_m["loss"],
            "val_acc":        val_m["accuracy"],
            "val_top5":       val_m["top5_accuracy"],
            "val_precision":  val_m["precision_macro"],
            "val_recall":     val_m["recall_macro"],
            "val_f1":         val_m["f1_macro"],
            "elapsed_s":      round(elapsed, 1),
        }
        self.history.append(row)

        log.info(
            f"Ep {epoch:03d} | "
            f"Train Loss={row['train_loss']:.4f} Acc={row['train_acc']:.2f}% | "
            f"Val Loss={row['val_loss']:.4f} Acc={row['val_acc']:.2f}% "
            f"F1={row['val_f1']:.4f} | LR={lr_now:.2e} | {elapsed:.0f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_m["accuracy"] > self.best_val_acc:
            self.best_val_acc = val_m["accuracy"]
            self.best_epoch   = epoch
            self._save_checkpoint(epoch, val_m)

        early_stop(val_m["accuracy"])

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        ckpt = {
            "epoch":            epoch,
            "arch":             self.model.arch,
            "model_state_dict": self.model.state_dict(),
            "metrics":          metrics,
            "config":           self.config,
        }
        torch.save(ckpt, BEST_MODEL_PATH)
        ep_path = CHECKPOINT_DIR / f"ckpt_ep{epoch:03d}.pth"
        torch.save(ckpt, ep_path)
        log.info(f"[Trainer] Checkpoint saved → {BEST_MODEL_PATH} (val_acc={metrics['accuracy']:.2f}%)")


# ─── Full Evaluation on Test Set ──────────────────────────────────────────────
def evaluate_on_test(
    model: FoodClassifier,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict:
    """Run inference on the test set and return full evaluation report."""
    criterion  = LabelSmoothingCrossEntropy()
    _, metrics = run_epoch(model, test_loader, criterion,
                           None, device, None, training=False)

    # Collect predictions for classification report
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch
            imgs   = imgs.to(device)
            preds  = model(imgs).argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    report     = classification_report(
        all_labels, all_preds,
        target_names=class_names[:len(set(all_labels))],
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    log.info(f"\n[Evaluation] Test Accuracy: {metrics['accuracy']:.2f}%")
    log.info(f"[Evaluation] Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    log.info(f"[Evaluation] Macro F1: {metrics['f1_macro']:.4f}")

    return {
        "metrics":              metrics,
        "classification_report": report,
        "confusion_matrix":     cm.tolist(),
    }
