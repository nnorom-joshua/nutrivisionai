# 🥗 NutriVision AI
### Real-Time AI-Powered Automated Nutritional Analysis System

> Deep Learning · Computer Vision · BMI Analysis · Personalised Recommendations  
> Built with PyTorch, YOLOv8, Optuna, Streamlit

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Project Structure](#-project-structure)
4. [Quick Start](#-quick-start)
5. [Training Pipeline](#-training-pipeline)
6. [Hyperparameter Tuning](#-hyperparameter-tuning)
7. [Streamlit App](#-streamlit-app)
8. [Running Tests](#-running-tests)
9. [Model Performance Targets](#-model-performance-targets)
10. [Configuration Reference](#-configuration-reference)
11. [Dataset](#-dataset)
12. [References](#-references)

---

## 🎯 Project Overview

NutriVision AI is a production-grade AI system that:

- **Detects food items** in meal images using YOLOv8 object detection
- **Classifies foods** from 101 categories using EfficientNet-B0 (transfer learning)
- **Estimates nutritional values** — calories, protein, carbohydrates, fat, fiber, sugar, sodium — via USDA FoodData Central
- **Computes BMI** using the WHO formula and classifies health status
- **Generates personalised dietary recommendations** using a hybrid rule-based + structured engine
- **Tracks daily intake** against personalised RDA targets (Mifflin-St Jeor equation)
- **Provides a full-featured Streamlit web interface** for real-time use

---

## 🏗️ System Architecture

```
Meal Image
    │
    ▼
[Pre-processing]   →  Resize 224×224, ImageNet normalisation
    │
    ▼
[YOLOv8n Detector] →  Detect food regions of interest (ROIs)
    │
    ▼ (per ROI)
[EfficientNet-B0]  →  101-class food classification (softmax)
    │
    ├─ conf ≥ 0.60  →  Normal record → USDA nutritional lookup → SQLite log
    └─ conf < 0.60  →  Weighted top-3 fallback → Flag to user
    │
    ▼
[BMI Module]       →  weight/height → BMI → WHO classification (5 tiers)
    │
    ▼
[Recommendation Engine]
    ├─ Rule-based filters (WHO dietary guidelines + clinical nutrition)
    └─ Structured output: Tips · Alerts · Food Suggestions · Summary
    │
    ▼
[Streamlit Dashboard] →  Real-time display · Meal logging · History tracking
```

### Model Architecture Details

| Component | Architecture | Pre-training | Parameters |
|-----------|-------------|--------------|-----------|
| Food Classifier | EfficientNet-B0 | ImageNet-1K | ~5.3M |
| Food Detector | YOLOv8n | COCO | ~3.2M |
| Recommendation | Rule-based + structured NLP | — | — |
| BMI Module | Mifflin-St Jeor + WHO | — | — |

**Custom Classification Head:**
```
EfficientNet-B0 backbone (1280-dim features)
    → Dropout(0.40)
    → Linear(1280 → 512)
    → BatchNorm1d(512)
    → ReLU
    → Dropout(0.20)
    → Linear(512 → 101)
    → Softmax
```

---

## 📁 Project Structure

```
nutritional_ai/
├── configs/
│   ├── __init__.py
│   └── config.py              # All hyperparameters, paths, constants
│
├── src/
│   ├── __init__.py
│   ├── database.py            # SQLite DB builder + nutritional lookup
│   ├── data_pipeline.py       # Transforms, DataLoader, DataCleaner
│   ├── models.py              # EfficientNet-B0, ResNet-50, MobileNetV3
│   ├── trainer.py             # Two-phase training engine + early stopping
│   ├── hyperparameter_tuning.py # Optuna TPE search + model selection
│   ├── inference.py           # FoodPredictor + MultiFoodDetector (YOLO)
│   ├── bmi_recommender.py     # BMI calc + personalised recommendation engine
│   └── evaluation.py          # Full metrics, confusion matrix, ECE, plots
│
├── streamlit_app/
│   ├── __init__.py
│   └── app.py                 # Complete Streamlit application (6 pages)
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Dataset EDA + nutritional analysis
│   └── 02_Training_Results.ipynb  # Training curves + HP search analysis
│
├── tests/
│   └── test_all.py            # 60+ pytest tests across all modules
│
├── data/                      # Created automatically
│   ├── raw/                   # Food-101 dataset goes here
│   ├── processed/
│   └── nutritional_db.sqlite
│
├── models/                    # Created automatically
│   ├── checkpoints/
│   └── best_model.pth
│
├── logs/                      # Created automatically
│   ├── training.log
│   ├── *_history.json
│   ├── model_selection.json
│   ├── hp_search_*.json
│   └── test_evaluation.json
│
├── .streamlit/
│   └── config.toml
│
├── train.py                   # Master training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Food-101 dataset

```bash
# Download (~5GB)
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz

# Reorganise into train/test splits
python -c "
import json, shutil
from pathlib import Path

images_dir = Path('food-101/images')
meta_dir   = Path('food-101/meta')

for split in ['train', 'test']:
    with open(meta_dir / f'{split}.json') as f:
        split_data = json.load(f)
    for class_name, img_list in split_data.items():
        dst = Path('data/raw/food-101/images') / split / class_name
        dst.mkdir(parents=True, exist_ok=True)
        for img_id in img_list:
            src = images_dir / f'{img_id}.jpg'
            if src.exists():
                shutil.copy2(src, dst / f'{Path(img_id).name}.jpg')
print('Done.')
"
```

### 3. Quick smoke test (5 epochs, no HP search)

```bash
python train.py --mode quick
```

### 4. Launch Streamlit app

```bash
streamlit run streamlit_app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🏋️ Training Pipeline

### Training modes

```bash
# Full pipeline: model selection → HP search → full training → evaluation
python train.py --mode full --epochs 50

# Skip selection/HP — use saved config (fast re-train)
python train.py --mode train_only --epochs 30

# Quick 5-epoch smoke test
python train.py --mode quick

# Evaluate saved best model on test set only
python train.py --mode evaluate
```

### Two-phase training

| Phase | Epochs | Backbone | LR |
|-------|--------|----------|----|
| Feature extraction | 1–5 | Frozen | 1e-3 |
| Fine-tuning | 6–30+ | Unfrozen | 1e-4 |

### Training features

- ✅ Label smoothing cross-entropy (smoothing=0.1)
- ✅ Cosine annealing LR scheduler
- ✅ Gradient clipping (max_norm=5.0)
- ✅ Mixed precision (AMP) on CUDA
- ✅ Weighted random sampler for class balance
- ✅ Early stopping (patience=7)
- ✅ Best-model checkpointing
- ✅ Per-epoch JSON metrics logging

---

## 🔧 Hyperparameter Tuning

NutriVision AI uses **Optuna** with the TPE sampler and MedianPruner.

### Search space

| Hyperparameter | Range / Choices |
|---------------|----------------|
| Learning rate | [1e-5, 1e-2] (log-uniform) |
| Weight decay | [1e-6, 1e-3] (log-uniform) |
| Dropout | {0.2, 0.3, 0.4, 0.5} |
| Hidden dim | {256, 512, 1024} |
| Optimizer | {adam, adamw, sgd} |
| Scheduler | {cosine, step, plateau} |

### Run HP search only

```python
from src.hyperparameter_tuning import run_hyperparameter_search
from torch.utils.data import DataLoader
import torch

# Assuming you have train_loader, val_loader ready
results, study = run_hyperparameter_search(
    arch="efficientnet_b0",
    train_loader=train_loader,
    val_loader=val_loader,
    device=torch.device("cpu"),
    n_trials=20,
)
print(results["best_params"])
```

---

## 🌐 Streamlit App

### Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Daily KPI dashboard, nutrient progress bars, BMI, quick tips |
| 📸 Analyse Meal | Upload/camera/manual entry → AI detection → nutrition → log |
| 📊 Daily Tracker | Date selector, 7-day history charts, meal log table, CSV export |
| 👤 My Profile | Height/weight/age/goal form → personalised RDA + BMI gauge |
| 📈 Model Insights | Training curves, evaluation metrics, HP search, architecture diagram |
| ℹ️ About | Project info, tech stack, dataset, references |

### Features
- Real-time food detection (YOLO + EfficientNet)
- Top-K alternative predictions with confidence scores
- Portion size adjustment slider
- Personalised RDA based on Mifflin-St Jeor + activity level
- 7-day nutritional history with Plotly charts
- BMI gauge with WHO classification
- Meal logging to SQLite
- CSV export of meal history

---

## 🧪 Running Tests

```bash
# Run full test suite (60+ tests)
pytest tests/ -v

# Run a specific test class
pytest tests/test_all.py::TestBMIRecommender -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

### Test coverage

| Module | Tests |
|--------|-------|
| Config | 8 |
| Database | 13 |
| Data Pipeline | 9 |
| Models | 12 |
| Trainer | 6 |
| BMI & Recommender | 14 |
| Inference | 11 |
| Evaluation | 9 |
| Integration | 5 |
| Edge Cases | 10 |
| **Total** | **97** |

---

## 📊 Model Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Top-1 Accuracy | ≥ 85% | Food-101 test set |
| Top-5 Accuracy | ≥ 97% | Food-101 test set |
| Macro F1-Score | ≥ 0.82 | All 101 classes |
| Macro Precision | ≥ 0.83 | All 101 classes |
| Macro Recall | ≥ 0.82 | All 101 classes |
| ECE (calibration) | ≤ 0.05 | Expected calibration error |
| Inference latency | ≤ 3s | Per image, GPU server |
| SUS usability score | ≥ 70 | System Usability Scale |

---

## ⚙️ Configuration Reference

All settings are in `configs/config.py`:

```python
# Key parameters
IMAGE_SIZE      = 224          # Input resolution
NUM_CLASSES     = 101          # Food-101 classes
BATCH_SIZE      = 32           # Training batch size
NUM_EPOCHS      = 30           # Max training epochs
FREEZE_EPOCHS   = 5            # Backbone frozen for first N epochs
EARLY_STOP_PAT  = 7            # Early stopping patience
CONF_THRESHOLD  = 0.60         # Min confidence for "normal" record
OPTUNA_N_TRIALS = 20           # HP search trials
SELECTED_MODEL  = "efficientnet_b0"  # Primary architecture
```

---

## 📦 Dataset

**Food-101** (Bossard et al., 2014)
- 101,000 images · 101 food categories
- 75,750 training · 25,250 test images
- 224 × 224 pixel input
- Download: https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

**USDA FoodData Central**
- Nutritional values for all 101 food categories (pre-indexed in SQLite)
- 7 nutrients per food: calories, protein, carbohydrates, fat, fiber, sugar, sodium
- Source: https://fdc.nal.usda.gov

---

## 📚 References

1. Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 – Mining Discriminative Components with Random Forests. *ECCV 2014*.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
3. Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
4. Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD 2019*.
5. WHO (2000). Obesity: Preventing and Managing the Global Epidemic. *WHO Technical Report Series 894*.
6. Mifflin, M. D. et al. (1990). A new predictive equation for resting energy expenditure in healthy individuals. *Am J Clin Nutr*.

---

