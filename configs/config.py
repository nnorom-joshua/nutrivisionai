"""
configs/config.py
Central configuration for the Real-Time AI Nutritional Analysis System.
All paths, hyperparameters, and constants live here.
"""

import os
from pathlib import Path

# ─── Project Paths ─────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
MODELS_DIR      = ROOT_DIR / "models"
LOGS_DIR        = ROOT_DIR / "logs"
SRC_DIR         = ROOT_DIR / "src"
STREAMLIT_DIR   = ROOT_DIR / "streamlit_app"

RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
DB_PATH         = DATA_DIR / "nutritional_db.sqlite"
NUTRITION_CSV   = DATA_DIR / "nutrition_data.csv"
CHECKPOINT_DIR  = MODELS_DIR / "checkpoints"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
OPTUNA_DB       = LOGS_DIR / "optuna_study.db"

# Create dirs if missing
for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, PROCESSED_DIR, RAW_DATA_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME    = "food101"
NUM_CLASSES     = 20
IMAGE_SIZE      = 224
DATASET_MEAN    = [0.485, 0.456, 0.406]   # ImageNet stats
DATASET_STD     = [0.229, 0.224, 0.225]

# Train / Val / Test split ratios (Food-101 has predefined splits)
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
TEST_RATIO      = 0.10

# ─── Training Defaults ────────────────────────────────────────────────────────
SEED            = 42
DEVICE          = "cuda"          # falls back to cpu in code
BATCH_SIZE      = 32
NUM_EPOCHS      = 30
EARLY_STOP_PAT  = 7               # patience epochs
FREEZE_EPOCHS   = 5               # backbone frozen for first N epochs
NUM_WORKERS     = 4

# ─── Model Selection ──────────────────────────────────────────────────────────
# Candidates evaluated during model selection phase
MODEL_CANDIDATES = [
    "efficientnet_b0",
    "resnet50",
    "mobilenetv3_large_100",
]
SELECTED_MODEL  = "efficientnet_b0"   # winner after selection
DROPOUT_RATE    = 0.40
HIDDEN_DIM      = 512

# ─── Hyperparameter Search Space (Optuna) ─────────────────────────────────────
OPTUNA_N_TRIALS = 20               # number of HP search trials
OPTUNA_TIMEOUT  = 3600             # seconds (1 hour max)

HP_LR_MIN       = 1e-5
HP_LR_MAX       = 1e-2
HP_WD_MIN       = 1e-6
HP_WD_MAX       = 1e-3
HP_DROPOUT      = [0.2, 0.3, 0.4, 0.5]
HP_HIDDEN       = [256, 512, 1024]
HP_BATCH        = [16, 32, 64]
HP_OPTIMIZERS   = ["adam", "adamw", "sgd"]
HP_SCHEDULERS   = ["cosine", "step", "plateau"]

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUG_HFLIP_P     = 0.5
AUG_ROTATION    = 15              # degrees
AUG_BRIGHTNESS  = 0.2
AUG_CONTRAST    = 0.2
AUG_SATURATION  = 0.2

# ─── Confidence Threshold ─────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.60            # min softmax confidence for a "normal" record

# ─── BMI Ranges (WHO) ─────────────────────────────────────────────────────────
BMI_CATEGORIES = [
    (0,    18.5, "Underweight",   "#3B82F6"),
    (18.5, 25.0, "Normal Weight", "#22C55E"),
    (25.0, 30.0, "Overweight",    "#F59E0B"),
    (30.0, 35.0, "Obese Class I", "#EF4444"),
    (35.0, 999,  "Obese Class II/III", "#7C3AED"),
]

# ─── RDA Targets (average adult) ─────────────────────────────────────────────
RDA = {
    "calories":      2000,   # kcal
    "protein":         50,   # g
    "carbohydrates":  275,   # g
    "fat":             78,   # g
    "fiber":           28,   # g
    "sugar":           50,   # g
    "sodium":        2300,   # mg
}

# ─── YOLO ─────────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"    # nano — fast, good for demo
YOLO_CONF       = 0.25
YOLO_IOU        = 0.45

# ─── Streamlit ────────────────────────────────────────────────────────────────
APP_TITLE       = "NutriVision AI"
APP_SUBTITLE    = "Real-Time AI-Powered Nutritional Analysis System"
APP_ICON        = "🥗"
PRIMARY_COLOR   = "#2C6FB2"
ACCENT_COLOR    = "#22C55E"
