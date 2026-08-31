"""
src/hyperparameter_tuning.py
Optuna-based hyperparameter search + model selection.
Runs multi-arch model selection first, then fine-grained HP search on winner.
"""

import logging
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    NUM_CLASSES, MODEL_CANDIDATES, SELECTED_MODEL,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, LOGS_DIR,
    HP_LR_MIN, HP_LR_MAX, HP_WD_MIN, HP_WD_MAX,
    HP_DROPOUT, HP_HIDDEN, HP_BATCH, HP_OPTIMIZERS, HP_SCHEDULERS,
    FREEZE_EPOCHS, SEED
)
from src.models import build_model, LabelSmoothingCrossEntropy
from src.trainer import run_epoch

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

optuna.logging.set_verbosity(optuna.logging.WARNING)



def get_num_classes(loader):
    dataset = loader.dataset

    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    return len(dataset.classes)

    
# ─── Quick Eval (few epochs on a subset) ─────────────────────────────────────
def quick_eval(
    arch: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_classes: int,
    lr: float,
    wd: float,
    dropout: float,
    hidden_dim: int,
    optimizer_name: str,
    scheduler_name: str,
    n_epochs: int,
    device: torch.device,
    trial: Optional[optuna.Trial] = None,
) -> float:
    """
    Train for n_epochs on the given loaders and return best val accuracy.
    Used as the Optuna objective inner loop.
    """
    model     = build_model(arch=arch, n_classes=n_classes,
                            hidden_dim=hidden_dim, dropout=dropout, device=device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        optimizer = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, n_epochs // 2), gamma=0.3)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    # Freeze backbone for first 2 epochs, then unfreeze
    model.freeze_backbone()
    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        if epoch == 3:
            model.unfreeze_backbone()
            # Lower LR for fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = lr * 0.1

        _, train_m = run_epoch(model, train_loader, criterion, optimizer,
                               device, None, training=True)
        _, val_m   = run_epoch(model, val_loader, criterion,
                               None, device, None, training=False)

        if isinstance(scheduler, optuna.trial._trial.Trial):
            pass
        if scheduler_name == "plateau":
            scheduler.step(val_m["accuracy"])
        else:
            scheduler.step()

        val_acc = val_m["accuracy"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Optuna pruning: cut unpromising trials early
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        log.debug(f"  [QuickEval] Ep {epoch}: val_acc={val_acc:.2f}%")

    return best_val_acc


# ─── Model Selection ─────────────────────────────────────────────────────────
def run_model_selection(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 3,
    architectures: List[str] = MODEL_CANDIDATES,
) -> Tuple[str, Dict]:
    """
    Evaluate each candidate architecture with default hyperparameters for n_epochs.
    Returns the best architecture name and results dict.
    """
    log.info("[ModelSelection] Comparing candidate architectures ...")
    results = {}

    for arch in architectures:
        log.info(f"  → Testing {arch} ...")
        try:
            val_acc = quick_eval(
                arch=arch,
                train_loader=train_loader,
                val_loader=val_loader,
                n_classes=get_num_classes(train_loader),
                lr=1e-3,
                wd=1e-4,
                dropout=0.4,
                hidden_dim=512,
                optimizer_name="adamw",
                scheduler_name="cosine",
                n_epochs=n_epochs,
                device=device,
            )
            results[arch] = {"val_acc": val_acc, "status": "ok"}
            log.info(f"  {arch}: val_acc = {val_acc:.2f}%")
        except Exception as e:
            log.warning(f"  {arch} failed: {e}")
            results[arch] = {"val_acc": 0.0, "status": str(e)}

    best_arch = max(results, key=lambda k: results[k]["val_acc"])
    log.info(f"[ModelSelection] Winner: {best_arch} "
             f"(val_acc={results[best_arch]['val_acc']:.2f}%)")

    # Save results
    out = LOGS_DIR / "model_selection.json"
    with open(out, "w") as f:
        json.dump({"results": results, "winner": best_arch}, f, indent=2)

    return best_arch, results


# ─── Optuna Objective ─────────────────────────────────────────────────────────
def build_objective(
    arch: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 5,
):
    """
    Returns an Optuna objective function for the given arch and loaders.
    """
    def objective(trial: optuna.Trial) -> float:
        lr          = trial.suggest_float("lr",          HP_LR_MIN, HP_LR_MAX, log=True)
        wd          = trial.suggest_float("weight_decay", HP_WD_MIN, HP_WD_MAX, log=True)
        dropout     = trial.suggest_categorical("dropout",   HP_DROPOUT)
        hidden_dim  = trial.suggest_categorical("hidden_dim", HP_HIDDEN)
        opt_name    = trial.suggest_categorical("optimizer",  HP_OPTIMIZERS)
        sched_name  = trial.suggest_categorical("scheduler",  HP_SCHEDULERS)

        try:
            val_acc = quick_eval(
                arch=arch,
                train_loader=train_loader,
                val_loader=val_loader,
                n_classes=get_num_classes(train_loader),
                lr=lr,
                wd=wd,
                dropout=dropout,
                hidden_dim=hidden_dim,
                optimizer_name=opt_name,
                scheduler_name=sched_name,
                n_epochs=n_epochs,
                device=device,
                trial=trial,
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            log.warning(f"Trial failed: {e}")
            return 0.0

        return val_acc

    return objective


def run_hyperparameter_search(
    arch: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int  = OPTUNA_TIMEOUT,
    n_epochs: int = 5,
    study_name: str = "food_hp_search",
) -> Dict:
    """
    Run Optuna TPE search over the hyperparameter space.
    Returns best hyperparameter dict and study object.
    """
    log.info(f"[HPSearch] Starting Optuna search on {arch} "
             f"({n_trials} trials, {timeout}s timeout) ...")

    storage = f"sqlite:///{LOGS_DIR / 'optuna_study.db'}"

    try:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=SEED, n_startup_trials=5),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            storage=storage,
            load_if_exists=True,
        )
    except Exception:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )

    objective = build_objective(arch, train_loader, val_loader, device, n_epochs)
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   n_jobs=1, show_progress_bar=False)

    best = study.best_trial
    best_params = best.params
    best_value  = best.value

    log.info(f"[HPSearch] Best trial #{best.number}: val_acc={best_value:.2f}%")
    log.info(f"[HPSearch] Best params: {best_params}")

    # Save results
    results = {
        "arch":        arch,
        "best_val_acc": best_value,
        "best_params":  best_params,
        "n_trials":     len(study.trials),
        "completed":    len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE]),
        "pruned":       len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.PRUNED]),
    }
    out = LOGS_DIR / f"hp_search_{arch}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[HPSearch] Results saved to {out}")

    return results, study


# ─── Full Pipeline: Selection → HP Search → Return best config ────────────────
def full_tuning_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    skip_selection: bool = False,
    skip_hp_search: bool = False,
) -> Dict:
    """
    1. Model selection (compare archs)
    2. Hyperparameter search on winner
    3. Return final best config ready for full training
    """
    # ── Step 1: Model selection ───────────────────────────────────────────
    if skip_selection:
        best_arch = SELECTED_MODEL
        log.info(f"[Pipeline] Skipping model selection, using {best_arch}")
    else:
        best_arch, _ = run_model_selection(train_loader, val_loader, device, n_epochs=3)

    # ── Step 2: HP search ─────────────────────────────────────────────────
    if skip_hp_search:
        best_params = {
            "lr": 5e-4, "weight_decay": 1e-4,
            "dropout": 0.4, "hidden_dim": 512,
            "optimizer": "adamw", "scheduler": "cosine",
        }
        log.info("[Pipeline] Skipping HP search, using default params")
    else:
        hp_results, _ = run_hyperparameter_search(
            arch=best_arch,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_trials=OPTUNA_N_TRIALS,
        )
        best_params = hp_results["best_params"]

    final_config = {
        "arch":          best_arch,
        "lr":            best_params.get("lr",            5e-4),
        "weight_decay":  best_params.get("weight_decay",  1e-4),
        "dropout":       best_params.get("dropout",       0.4),
        "hidden_dim":    best_params.get("hidden_dim",    512),
        "optimizer":     best_params.get("optimizer",    "adamw"),
        "scheduler":     best_params.get("scheduler",    "cosine"),
        "freeze_epochs": FREEZE_EPOCHS,
    }

    out = LOGS_DIR / "final_config.json"
    with open(out, "w") as f:
        json.dump(final_config, f, indent=2)
    log.info(f"[Pipeline] Final config saved → {out}")
    log.info(f"[Pipeline] Final config: {final_config}")

    return final_config
