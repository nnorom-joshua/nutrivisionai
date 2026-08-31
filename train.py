"""
train.py
Master training script.
Usage:
    python train.py --mode full          # model selection + HP search + full train
    python train.py --mode train_only    # skip selection/HP, use saved config
    python train.py --mode quick         # 5-epoch smoke test (no HP search)
    python train.py --mode evaluate      # load best model and run test evaluation
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import (
    NUM_CLASSES, SELECTED_MODEL, BEST_MODEL_PATH, LOGS_DIR,
    BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, SEED, CHECKPOINT_DIR,
    DATA_DIR, RAW_DATA_DIR, FREEZE_EPOCHS
)
from src.database import build_database, FOOD101_CLASSES
from src.data_pipeline import get_dataloaders, DataCleaner
from src.models import build_model, load_model
from src.trainer import Trainer, evaluate_on_test
from src.hyperparameter_tuning import full_tuning_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "training.log"),
    ]
)
log = logging.getLogger(__name__)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Device] Using: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def check_dataset(data_root: Path) -> bool:
    """Check official Food-101 dataset."""

    images_dir = data_root / "images"
    meta_dir = data_root / "meta"

    train_txt = meta_dir / "train.txt"
    test_txt = meta_dir / "test.txt"

    if not images_dir.exists():
        log.error(f"Missing: {images_dir}")
        return False

    if not train_txt.exists():
        log.error(f"Missing: {train_txt}")
        return False

    if not test_txt.exists():
        log.error(f"Missing: {test_txt}")
        return False

    log.info("✓ Official Food-101 dataset found.")

    return True


def step_build_database():
    log.info("[Step 1] Building nutritional SQLite database ...")
    build_database()
    log.info("[Step 1] ✓ Database ready.")


def step_clean_data(data_root: Path):
    log.info("[Step 2] Running data cleaning pipeline ...")
    cleaner = DataCleaner(data_root)
    report  = cleaner.run(splits=["train", "test"])
    log.info(f"[Step 2] ✓ Cleaning report: {report}")
    eda_df = cleaner.generate_eda_report(split="train")
    eda_path = LOGS_DIR / "eda_class_counts.csv"
    eda_df.to_csv(eda_path, index=False)
    log.info(f"[Step 2] ✓ EDA saved to {eda_path}")
    return report


def step_build_loaders(data_root: Path, batch_size: int = BATCH_SIZE):
    log.info("[Step 3] Building DataLoaders ...")
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )
    # Save class mapping
    mapping_path = LOGS_DIR / "class_to_idx.json"
    with open(mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    log.info(f"[Step 3] ✓ Class mapping saved to {mapping_path}")
    return train_loader, val_loader, test_loader, class_names, class_to_idx


def step_model_selection_and_hp(
    train_loader, val_loader, device,
    skip_selection=False, skip_hp=False
) -> dict:
    log.info("[Step 4] Running model selection + hyperparameter search ...")
    final_config = full_tuning_pipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        skip_selection=skip_selection,
        skip_hp_search=skip_hp,
    )
    log.info(f"[Step 4] ✓ Final config: {final_config}")
    return final_config


def step_full_train(
    train_loader, val_loader, class_names, device, config: dict
) -> list:
    arch        = config.get("arch",       SELECTED_MODEL)
    dropout     = config.get("dropout",    0.4)
    hidden_dim  = config.get("hidden_dim", 512)

    log.info(f"[Step 5] Building final model: {arch} ...")
    model = build_model(
        arch=arch,
        n_classes=len(class_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device
    )

    run_name = f"{arch}_{int(time.time())}"
    trainer  = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        run_name=run_name,
    )

    log.info(f"[Step 5] Starting full training ({config.get('epochs', NUM_EPOCHS)} epochs) ...")
    history = trainer.train()
    log.info("[Step 5] ✓ Training complete.")
    return history


def step_evaluate(test_loader, class_names, device):
    log.info("[Step 6] Evaluating best model on test set ...")
    if not BEST_MODEL_PATH.exists():
        log.error(f"No checkpoint at {BEST_MODEL_PATH}. Run training first.")
        return {}

    model   = load_model(BEST_MODEL_PATH, arch=SELECTED_MODEL,
                         n_classes=len(class_names), device=device)
    results = evaluate_on_test(model, test_loader, class_names, device)

    metrics = results["metrics"]
    log.info(f"[Step 6] ✓ Test Accuracy:    {metrics['accuracy']:.2f}%")
    log.info(f"[Step 6] ✓ Top-5 Accuracy:   {metrics['top5_accuracy']:.2f}%")
    log.info(f"[Step 6] ✓ Macro Precision:  {metrics['precision_macro']:.4f}")
    log.info(f"[Step 6] ✓ Macro Recall:     {metrics['recall_macro']:.4f}")
    log.info(f"[Step 6] ✓ Macro F1-Score:   {metrics['f1_macro']:.4f}")

    # Save evaluation
    eval_path = LOGS_DIR / "test_evaluation.json"
    with open(eval_path, "w") as f:
        # confusion matrix can be large; skip for JSON if needed
        out = {
            "metrics":              results["metrics"],
            "classification_report": results["classification_report"],
        }
        json.dump(out, f, indent=2)
    log.info(f"[Step 6] ✓ Evaluation report saved to {eval_path}")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NutriVision AI Training Pipeline")
    parser.add_argument("--mode", choices=["full", "train_only", "quick", "evaluate"],
                        default="quick", help="Pipeline mode")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(RAW_DATA_DIR / "food-101"),
        help="Path to the Food-101 dataset"
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    device    = get_device()

    # ── Step 1: Database ───────────────────────────────────────────────────
    step_build_database()

    # ── Dataset check ──────────────────────────────────────────────────────
    if not check_dataset(data_root):
        log.error("Food-101 dataset not found.")
        sys.exit(1)

    # ── Step 2: Clean ──────────────────────────────────────────────────────
    if args.mode in ("full", "train_only", "quick") and data_root.exists():
        step_clean_data(data_root)

    # ── Step 3: DataLoaders ────────────────────────────────────────────────
    if args.mode != "evaluate":
        if not data_root.exists():
            log.error("No data available. Exiting.")
            sys.exit(1)
        train_loader, val_loader, test_loader, class_names, class_to_idx = \
            step_build_loaders(data_root, args.batch_size)
    else:
        # Load class names from saved mapping
        mapping_path = LOGS_DIR / "class_to_idx.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                class_to_idx = json.load(f)
            class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
        else:
            class_names = FOOD101_CLASSES
        _, _, test_loader, _, _ = step_build_loaders(data_root, args.batch_size)

    # ── Mode branching ─────────────────────────────────────────────────────
    if args.mode == "full":
        config = step_model_selection_and_hp(
            train_loader, val_loader, device,
            skip_selection=False, skip_hp=False
        )
        config["epochs"] = args.epochs
        step_full_train(train_loader, val_loader, class_names, device, config)
        step_evaluate(test_loader, class_names, device)

    elif args.mode == "train_only":
        # Try loading saved config
        cfg_path = LOGS_DIR / "final_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                config = json.load(f)
            log.info(f"[train_only] Loaded config from {cfg_path}")
        else:
            config = {
                "arch": SELECTED_MODEL, "lr": 5e-4, "weight_decay": 1e-4,
                "dropout": 0.4, "hidden_dim": 512,
                "optimizer": "adamw", "scheduler": "cosine",
                "freeze_epochs": FREEZE_EPOCHS,
            }
        config["epochs"] = args.epochs
        step_full_train(train_loader, val_loader, class_names, device, config)
        step_evaluate(test_loader, class_names, device)

    elif args.mode == "quick":
        log.info("[Quick Mode] 5-epoch smoke test, no HP search.")
        config = {
            "arch": SELECTED_MODEL, "lr": 1e-3, "weight_decay": 1e-4,
            "dropout": 0.4, "hidden_dim": 512,
            "optimizer": "adamw", "scheduler": "cosine",
            "freeze_epochs": 2, "epochs": 5, "patience": 3,
        }
        step_full_train(train_loader, val_loader, class_names, device, config)
        step_evaluate(test_loader, class_names, device)

    elif args.mode == "evaluate":
        step_evaluate(test_loader, class_names, device)

    log.info("[Pipeline] ✓ All steps complete.")


def _create_demo_dataset(data_root: Path, n_classes: int = 10, n_per_class: int = 5):
    """Create a tiny synthetic Food-101-like dataset for smoke-testing."""
    from PIL import Image
    import random
    classes = FOOD101_CLASSES[:n_classes]
    for split in ["train", "test"]:
        for cls in classes:
            cls_dir = data_root / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                img = Image.new("RGB", (224, 224),
                                color=(random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255)))
                img.save(cls_dir / f"{i:04d}.jpg")
    log.info(f"[Demo] Created synthetic dataset at {data_root} "
             f"({n_classes} classes × {n_per_class} images × 2 splits)")


if __name__ == "__main__":
    main()
