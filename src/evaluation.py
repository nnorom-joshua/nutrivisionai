"""
src/evaluation.py
Comprehensive model evaluation utilities:
- Per-class precision / recall / F1
- Confusion matrix (full + top-N)
- Top-K accuracy curves
- Calibration analysis (ECE)
- Per-class worst/best performer analysis
- Export to JSON / CSV
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    top_k_accuracy_score, classification_report, confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import LOGS_DIR, NUM_CLASSES
from src.models import FoodClassifier
from src.trainer import LabelSmoothingCrossEntropy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─── Inference collection ─────────────────────────────────────────────────────
def collect_predictions(
    model: FoodClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference over loader.
    Returns: (all_labels, all_preds, all_probs)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch

            imgs   = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ─── Core Metrics ─────────────────────────────────────────────────────────────
def compute_full_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    n_classes: int = NUM_CLASSES,
) -> Dict:
    """
    Compute all evaluation metrics and return as a structured dict.
    """
    acc    = accuracy_score(labels, preds)
    prec   = precision_score(labels, preds, average="macro", zero_division=0)
    rec    = recall_score(labels, preds, average="macro", zero_division=0)
    f1     = f1_score(labels, preds, average="macro", zero_division=0)
    f1_w   = f1_score(labels, preds, average="weighted", zero_division=0)

    try:
        top3 = top_k_accuracy_score(labels, probs, k=3, labels=list(range(n_classes)))
        top5 = top_k_accuracy_score(labels, probs, k=5, labels=list(range(n_classes)))
    except Exception:
        top3, top5 = 0.0, 0.0

    # Per-class report — align target_names with unique labels present
    unique_labels = sorted(set(labels))
    target_names  = [class_names[i] for i in unique_labels if i < len(class_names)] \
                    if class_names else None
    report = classification_report(labels, preds,
                                    labels=unique_labels,
                                    target_names=target_names,
                                    output_dict=True, zero_division=0)

    # ECE (Expected Calibration Error)
    ece = _compute_ece(labels, probs, n_bins=15)

    return {
        "accuracy":         round(acc * 100, 4),
        "top3_accuracy":    round(top3 * 100, 4),
        "top5_accuracy":    round(top5 * 100, 4),
        "precision_macro":  round(prec, 4),
        "recall_macro":     round(rec, 4),
        "f1_macro":         round(f1, 4),
        "f1_weighted":      round(f1_w, 4),
        "ece":              round(ece, 4),
        "n_samples":        int(len(labels)),
        "n_classes":        int(n_classes),
        "per_class_report": report,
    }


def _compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error across all classes (macro-averaged)."""
    max_probs = probs.max(axis=1)
    correct   = (probs.argmax(axis=1) == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (max_probs >= lo) & (max_probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = max_probs[mask].mean()
        ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ─── Per-class analysis ───────────────────────────────────────────────────────
def per_class_analysis(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """Return sorted DataFrame of per-class metrics."""
    rows = []
    for cls_idx, cls_name in enumerate(class_names):
        mask = (labels == cls_idx)
        if mask.sum() == 0:
            continue
        cls_labels = (labels == cls_idx).astype(int)
        cls_preds  = (preds  == cls_idx).astype(int)
        tp  = ((cls_preds == 1) & (cls_labels == 1)).sum()
        fp  = ((cls_preds == 1) & (cls_labels == 0)).sum()
        fn  = ((cls_preds == 0) & (cls_labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1c  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        avg_conf = probs[mask, cls_idx].mean()
        rows.append({
            "class_idx":    cls_idx,
            "class_name":   cls_name.replace("_", " ").title(),
            "support":      int(mask.sum()),
            "precision":    round(prec, 4),
            "recall":       round(rec, 4),
            "f1":           round(f1c, 4),
            "avg_confidence": round(float(avg_conf), 4),
        })

    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return df


# ─── Confusion Matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(
    labels: np.ndarray,
    preds:  np.ndarray,
    class_names: List[str],
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot normalised confusion matrix for the top_n most frequent classes.
    """
    # Select top_n classes by support
    top_classes = (
        pd.Series(labels)
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    mask     = np.isin(labels, top_classes)
    sub_lab  = labels[mask]
    sub_pred = preds[mask]

    cm    = confusion_matrix(sub_lab, sub_pred, labels=top_classes, normalize="true")
    names = [class_names[i].replace("_", " ").title() if i < len(class_names)
             else str(i) for i in top_classes]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names,
                linewidths=0.3, linecolor="#e2e8f0",
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True",      fontsize=12, labelpad=10)
    ax.set_title(f"Normalised Confusion Matrix (Top {top_n} Classes)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[Eval] Confusion matrix saved → {save_path}")

    return fig


# ─── Top-K Accuracy Curve ─────────────────────────────────────────────────────
def plot_topk_curve(
    labels: np.ndarray,
    probs:  np.ndarray,
    n_classes: int = NUM_CLASSES,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Top-K accuracy for K = 1..10."""
    ks   = list(range(1, min(11, n_classes + 1)))
    accs = []
    for k in ks:
        try:
            a = top_k_accuracy_score(labels, probs, k=k, labels=list(range(n_classes)))
        except Exception:
            a = 0.0
        accs.append(round(a * 100, 2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, accs, marker="o", color="#2C6FB2", linewidth=2, markersize=7)
    ax.fill_between(ks, accs, alpha=0.12, color="#2C6FB2")
    for k, a in zip(ks, accs):
        ax.annotate(f"{a:.1f}%", (k, a), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)
    ax.axhline(85, color="#ef4444", linestyle="--", linewidth=1, label="85% target")
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Top-K Accuracy (%)", fontsize=12)
    ax.set_title("Top-K Accuracy Curve", fontsize=14)
    ax.set_xticks(ks)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[Eval] Top-K curve saved → {save_path}")

    return fig


# ─── Calibration Curve ────────────────────────────────────────────────────────
def plot_calibration_curve(
    labels: np.ndarray,
    probs:  np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Reliability diagram for model calibration."""
    max_probs   = probs.max(axis=1)
    correct     = (probs.argmax(axis=1) == labels).astype(float)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        correct, max_probs, n_bins=10, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.5)
    ax.plot(mean_predicted_value, fraction_of_positives,
            "s-", color="#2C6FB2", linewidth=2, markersize=8, label="Model")
    ax.fill_between(mean_predicted_value, fraction_of_positives,
                    mean_predicted_value, alpha=0.12, color="#ef4444")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title("Model Calibration (Reliability Diagram)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[Eval] Calibration curve saved → {save_path}")

    return fig


# ─── Confidence Distribution ──────────────────────────────────────────────────
def plot_confidence_distribution(
    labels: np.ndarray,
    probs:  np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram of max confidence for correct vs incorrect predictions."""
    max_probs = probs.max(axis=1)
    correct   = probs.argmax(axis=1) == labels

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 26)
    ax.hist(max_probs[correct],  bins=bins, alpha=0.65, color="#22c55e",
            label=f"Correct (n={correct.sum():,})", density=True)
    ax.hist(max_probs[~correct], bins=bins, alpha=0.65, color="#ef4444",
            label=f"Incorrect (n={(~correct).sum():,})", density=True)
    ax.axvline(0.60, color="#f59e0b", linestyle="--", linewidth=1.5, label="Threshold (0.60)")
    ax.set_xlabel("Prediction Confidence", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Confidence Distribution: Correct vs Incorrect", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ─── F1 Bar Chart ─────────────────────────────────────────────────────────────
def plot_per_class_f1(
    per_class_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of per-class F1 for top and bottom N classes."""
    top_df  = per_class_df.head(top_n)
    bot_df  = per_class_df.tail(top_n).sort_values("f1")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, df, title, color in [
        (axes[0], top_df,  f"Top {top_n} Classes by F1",    "#22c55e"),
        (axes[1], bot_df,  f"Bottom {top_n} Classes by F1", "#ef4444"),
    ]:
        bars = ax.barh(df["class_name"], df["f1"], color=color, alpha=0.8, edgecolor="white")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("F1 Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axvline(0.85, color="#64748b", linestyle="--", linewidth=1, alpha=0.7, label="Target 0.85")
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.25)
        for bar, val in zip(bars, df["f1"]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout(pad=3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[Eval] F1 chart saved → {save_path}")

    return fig


# ─── Full Evaluation Pipeline ─────────────────────────────────────────────────
def run_full_evaluation(
    model: FoodClassifier,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    output_dir: Path = LOGS_DIR,
) -> Dict:
    """
    Complete evaluation: metrics + all charts + CSV/JSON export.
    Returns the full results dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("[Eval] Collecting predictions …")
    labels, preds, probs = collect_predictions(model, test_loader, device)

    # ── Metrics ────────────────────────────────────────────────────────────
    log.info("[Eval] Computing metrics …")
    metrics = compute_full_metrics(labels, preds, probs, class_names)

    log.info(f"  Top-1 Accuracy  : {metrics['accuracy']:.2f}%")
    log.info(f"  Top-3 Accuracy  : {metrics['top3_accuracy']:.2f}%")
    log.info(f"  Top-5 Accuracy  : {metrics['top5_accuracy']:.2f}%")
    log.info(f"  Macro Precision : {metrics['precision_macro']:.4f}")
    log.info(f"  Macro Recall    : {metrics['recall_macro']:.4f}")
    log.info(f"  Macro F1        : {metrics['f1_macro']:.4f}")
    log.info(f"  Weighted F1     : {metrics['f1_weighted']:.4f}")
    log.info(f"  ECE             : {metrics['ece']:.4f}")

    # ── Per-class ──────────────────────────────────────────────────────────
    per_class_df = per_class_analysis(labels, preds, probs, class_names)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)

    # ── Plots ──────────────────────────────────────────────────────────────
    log.info("[Eval] Generating plots …")
    plot_confusion_matrix(labels, preds, class_names, top_n=20,
                          save_path=output_dir / "confusion_matrix.png")
    plot_topk_curve(labels, probs, save_path=output_dir / "topk_curve.png")
    plot_calibration_curve(labels, probs, save_path=output_dir / "calibration_curve.png")
    plot_confidence_distribution(labels, probs,
                                  save_path=output_dir / "confidence_dist.png")
    plot_per_class_f1(per_class_df, top_n=20,
                       save_path=output_dir / "per_class_f1.png")
    plt.close("all")

    # ── Save JSON ──────────────────────────────────────────────────────────
    save_metrics = {k: v for k, v in metrics.items() if k != "per_class_report"}
    with open(output_dir / "test_evaluation.json", "w") as f:
        json.dump({
            "metrics":               save_metrics,
            "classification_report": metrics["per_class_report"],
        }, f, indent=2)

    log.info(f"[Eval] All outputs saved to {output_dir}")
    return {
        "metrics":       metrics,
        "per_class_df":  per_class_df,
        "labels":        labels,
        "preds":         preds,
        "probs":         probs,
    }


if __name__ == "__main__":
    # Smoke-test with random data
    import torch
    n, c = 500, 101
    labels = np.random.randint(0, c, n)
    probs  = np.random.dirichlet(np.ones(c), size=n)
    preds  = probs.argmax(axis=1)
    names  = [f"food_{i}" for i in range(c)]

    m = compute_full_metrics(labels, preds, probs, names, n_classes=c)
    print(f"Accuracy: {m['accuracy']:.2f}%  |  Macro F1: {m['f1_macro']:.4f}  |  ECE: {m['ece']:.4f}")
    print("Smoke test passed ✓")
