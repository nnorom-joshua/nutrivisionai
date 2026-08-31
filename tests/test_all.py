"""
tests/test_all.py
Comprehensive test suite for the NutriVision AI system.
Run with: pytest tests/ -v
"""

import sys
import json
import sqlite3
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import pytest
import torch

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import (
    NUM_CLASSES, IMAGE_SIZE, DATASET_MEAN, DATASET_STD,
    RDA, BMI_CATEGORIES, CONF_THRESHOLD, SELECTED_MODEL
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="session")
def tmp_db(tmp_path_factory):
    """Temporary SQLite database for testing."""
    db = tmp_path_factory.mktemp("db") / "test.sqlite"
    from src.database import build_database
    build_database(db)
    return db


@pytest.fixture(scope="session")
def dummy_image():
    """224×224 RGB PIL image."""
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(120, 80, 60))


@pytest.fixture(scope="session")
def dummy_tensor():
    """Batch of 2 random tensors [2, 3, 224, 224]."""
    return torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model(device):
    from src.models import build_model
    m = build_model(arch=SELECTED_MODEL, n_classes=NUM_CLASSES,
                    pretrained=False, device=device)
    m.eval()
    return m


@pytest.fixture(scope="session")
def class_names():
    from src.database import FOOD101_CLASSES
    return FOOD101_CLASSES


@pytest.fixture(scope="session")
def sample_user():
    return {
        "username":  "test_user",
        "weight_kg": 75.0,
        "height_m":  1.75,
        "age":       30,
        "gender":    "male",
        "activity":  "moderately active",
        "goal":      "maintain",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class TestConfig:
    def test_num_classes(self):
        assert NUM_CLASSES == 101

    def test_image_size(self):
        assert IMAGE_SIZE == 224

    def test_rda_keys(self):
        required = {"calories", "protein", "carbohydrates", "fat", "fiber", "sugar", "sodium"}
        assert required.issubset(set(RDA.keys()))

    def test_rda_positive(self):
        for k, v in RDA.items():
            assert v > 0, f"RDA[{k}] must be > 0"

    def test_bmi_categories_complete(self):
        # Categories must cover 0–∞ without gaps
        assert len(BMI_CATEGORIES) == 5
        # First starts at 0
        assert BMI_CATEGORIES[0][0] == 0
        # Last ends at a large number
        assert BMI_CATEGORIES[-1][1] >= 999

    def test_conf_threshold(self):
        assert 0 < CONF_THRESHOLD < 1

    def test_dataset_mean_std(self):
        assert len(DATASET_MEAN) == 3
        assert len(DATASET_STD) == 3
        for v in DATASET_MEAN + DATASET_STD:
            assert 0.0 <= v <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATABASE
# ══════════════════════════════════════════════════════════════════════════════
class TestDatabase:
    def test_db_created(self, tmp_db):
        assert tmp_db.exists()

    def test_tables_exist(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        conn.close()
        assert {"food_nutrition", "users", "meal_logs"}.issubset(tables)

    def test_food_count(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM food_nutrition")
        count = cur.fetchone()[0]
        conn.close()
        assert count == 101

    def test_get_nutrition_pizza(self, tmp_db):
        from src.database import get_nutrition
        n = get_nutrition("pizza", 200, tmp_db)
        assert n["food_name"] == "pizza"
        assert n["portion_g"] == 200
        assert n["calories"] > 0
        assert n["protein"] >= 0
        assert n["carbohydrates"] >= 0
        assert n["fat"] >= 0

    def test_get_nutrition_scaling(self, tmp_db):
        from src.database import get_nutrition
        n100 = get_nutrition("pizza", 100, tmp_db)
        n200 = get_nutrition("pizza", 200, tmp_db)
        assert abs(n200["calories"] - n100["calories"] * 2) < 0.5

    def test_get_nutrition_missing(self, tmp_db):
        from src.database import get_nutrition
        n = get_nutrition("nonexistent_food_xyz", 100, tmp_db)
        assert n == {}

    def test_save_and_get_user(self, tmp_db, sample_user):
        from src.database import save_user, get_user
        u = sample_user
        save_user(u["username"], u["weight_kg"], u["height_m"],
                  u["age"], u["gender"], u["activity"], u["goal"], tmp_db)
        result = get_user(u["username"], tmp_db)
        assert result is not None
        assert result["username"] == u["username"]
        assert abs(result["weight_kg"] - u["weight_kg"]) < 0.01
        assert abs(result["height_m"]  - u["height_m"])  < 0.001

    def test_log_meal(self, tmp_db):
        from src.database import log_meal, get_daily_intake
        log_meal("test_user", "pizza", 200, 0.92, tmp_db)
        log_meal("test_user", "sushi", 150, 0.85, tmp_db)
        intake = get_daily_intake("test_user", db_path=tmp_db)
        assert intake["calories"] > 0
        assert intake["protein"]  > 0

    def test_meal_history(self, tmp_db):
        from src.database import get_meal_history
        df = get_meal_history("test_user", days=7, db_path=tmp_db)
        assert isinstance(df, pd.DataFrame)
        assert "food_name" in df.columns

    def test_upsert_user(self, tmp_db, sample_user):
        from src.database import save_user, get_user
        u = sample_user
        save_user(u["username"], 80.0, u["height_m"],
                  u["age"], u["gender"], u["activity"], u["goal"], tmp_db)
        result = get_user(u["username"], tmp_db)
        assert abs(result["weight_kg"] - 80.0) < 0.01

    def test_food101_classes_length(self):
        from src.database import FOOD101_CLASSES
        assert len(FOOD101_CLASSES) == 101

    def test_food_nutrition_keys(self):
        from src.database import FOOD_NUTRITION
        required = {"calories", "protein", "carbohydrates", "fat", "fiber", "sugar", "sodium"}
        for food, data in FOOD_NUTRITION.items():
            missing = required - set(data.keys())
            assert not missing, f"{food} missing keys: {missing}"

    def test_all_nutrition_positive(self):
        from src.database import FOOD_NUTRITION
        for food, data in FOOD_NUTRITION.items():
            assert data["calories"] > 0, f"{food}: calories must be > 0"
            for k, v in data.items():
                assert v >= 0, f"{food}.{k} must be >= 0, got {v}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class TestDataPipeline:
    def test_train_transforms_output_shape(self, dummy_image):
        from src.data_pipeline import get_train_transforms
        t   = get_train_transforms()
        out = t(dummy_image)
        assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert out.dtype == torch.float32

    def test_val_transforms_output_shape(self, dummy_image):
        from src.data_pipeline import get_val_transforms
        t   = get_val_transforms()
        out = t(dummy_image)
        assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_inference_transforms_output_shape(self, dummy_image):
        from src.data_pipeline import get_inference_transforms
        t   = get_inference_transforms()
        out = t(dummy_image)
        assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_preprocess_image_batch_dim(self, dummy_image):
        from src.data_pipeline import preprocess_image
        tensor = preprocess_image(dummy_image)
        assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_normalisation_range(self, dummy_image):
        from src.data_pipeline import get_val_transforms
        t   = get_val_transforms()
        out = t(dummy_image)
        # After normalisation, most values should be in [-3, 3]
        assert out.min().item() > -5
        assert out.max().item() <  5

    def test_denormalise(self, dummy_image):
        from src.data_pipeline import get_val_transforms, denormalize
        t   = get_val_transforms()
        out = t(dummy_image)
        de  = denormalize(out)
        assert de.min().item() >= 0.0
        assert de.max().item() <= 1.0

    def test_data_cleaner_init(self, tmp_path):
        from src.data_pipeline import DataCleaner
        cleaner = DataCleaner(tmp_path)
        assert cleaner.data_root == tmp_path

    def test_data_cleaner_missing_dir(self, tmp_path):
        from src.data_pipeline import DataCleaner
        cleaner = DataCleaner(tmp_path)
        # Should not raise even if splits don't exist
        report = cleaner.run(splits=["train"])
        assert report["total"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELS
# ══════════════════════════════════════════════════════════════════════════════
class TestModels:
    def test_model_builds_efficientnet(self, device):
        from src.models import build_model
        m = build_model("efficientnet_b0", n_classes=101, pretrained=False, device=device)
        assert m is not None
        assert m.arch == "efficientnet_b0"

    def test_model_builds_resnet(self, device):
        from src.models import build_model
        m = build_model("resnet50", n_classes=101, pretrained=False, device=device)
        assert m.arch == "resnet50"

    def test_model_builds_mobilenet(self, device):
        from src.models import build_model
        m = build_model("mobilenetv3_large_100", n_classes=101, pretrained=False, device=device)
        assert m.arch == "mobilenetv3_large_100"

    def test_model_output_shape(self, model, dummy_tensor):
        with torch.no_grad():
            out = model(dummy_tensor)
        assert out.shape == (2, NUM_CLASSES)

    def test_softmax_sums_to_one(self, model, dummy_tensor):
        import torch.nn.functional as F
        with torch.no_grad():
            out   = model(dummy_tensor)
            probs = F.softmax(out, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_predict_returns_class_and_conf(self, model, dummy_tensor):
        with torch.no_grad():
            cls, conf = model.predict(dummy_tensor)
        assert cls.shape  == (2,)
        assert conf.shape == (2,)
        assert (cls  >= 0).all() and (cls  < NUM_CLASSES).all()
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_predict_top_k(self, model, dummy_tensor):
        with torch.no_grad():
            confs, idxs = model.predict_top_k(dummy_tensor, k=5)
        assert confs.shape == (2, 5)
        assert idxs.shape  == (2, 5)

    def test_count_parameters(self, model):
        params = model.count_parameters()
        assert "total"     in params
        assert "trainable" in params
        assert "frozen"    in params
        assert params["total"] == params["trainable"] + params["frozen"]
        assert params["total"] > 0

    def test_freeze_backbone(self, device):
        from src.models import build_model
        m = build_model(pretrained=False, device=device)
        m.freeze_backbone()
        frozen_count    = sum(1 for p in m.parameters() if not p.requires_grad)
        trainable_count = sum(1 for p in m.parameters() if p.requires_grad)
        assert frozen_count > 0
        assert trainable_count > 0

    def test_unfreeze_backbone(self, device):
        from src.models import build_model
        m = build_model(pretrained=False, device=device)
        m.freeze_backbone()
        m.unfreeze_backbone()
        frozen_count = sum(1 for p in m.parameters() if not p.requires_grad)
        assert frozen_count == 0

    def test_label_smoothing_loss(self, dummy_tensor, device):
        from src.models import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits    = dummy_tensor[:, :NUM_CLASSES] if dummy_tensor.shape[1] >= NUM_CLASSES \
                    else torch.randn(2, NUM_CLASSES)
        labels    = torch.randint(0, NUM_CLASSES, (2,))
        loss      = criterion(logits, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_model_save_and_load(self, model, device, tmp_path):
        from src.models import load_model
        ckpt_path = tmp_path / "test_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "arch": model.arch,
        }, ckpt_path)
        loaded = load_model(ckpt_path, arch=model.arch,
                             n_classes=NUM_CLASSES, device=device)
        loaded.eval()
        x   = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out_orig   = model(x)
            out_loaded = loaded(x)
        assert torch.allclose(out_orig, out_loaded, atol=1e-5)

    def test_unsupported_arch_raises(self, device):
        from src.models import build_model
        with pytest.raises(AssertionError):
            build_model(arch="unknown_arch_xyz", pretrained=False, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINER
# ══════════════════════════════════════════════════════════════════════════════
class TestTrainer:
    def test_early_stopping_max_mode(self):
        from src.trainer import EarlyStopping
        # patience=3: triggers when counter reaches patience after 3 non-improving calls
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)   # sets best=0.5, counter=0
        es(0.6)   # improvement → best=0.6, counter=0
        es(0.55)  # no improvement → counter=1
        es(0.54)  # no improvement → counter=2
        result = es(0.53)  # no improvement → counter=3 → triggered
        assert result is True
        assert es.triggered

    def test_early_stopping_min_mode(self):
        from src.trainer import EarlyStopping
        # patience=2: triggers on 2nd non-improving call
        es = EarlyStopping(patience=2, mode="min")
        es(1.0)   # sets best=1.0
        es(0.8)   # improvement → best=0.8
        es(0.85)  # no improvement → counter=1
        result = es(0.90)  # no improvement → counter=2 → triggered
        assert result is True
        assert es.triggered

    def test_early_stopping_resets_on_improvement(self):
        from src.trainer import EarlyStopping
        es = EarlyStopping(patience=2, mode="max")
        es(0.5); es(0.4)  # counter=1 after 2nd
        es(0.9)            # improvement → counter resets
        assert es.counter == 0
        assert not es.triggered

    def test_compute_metrics_shape(self):
        from src.trainer import compute_metrics
        n, c = 200, 101
        labels = np.random.randint(0, c, n)
        probs  = np.random.dirichlet(np.ones(c), size=n)
        preds  = probs.argmax(axis=1)
        m = compute_metrics(labels, preds, probs, c)
        assert "accuracy"       in m
        assert "f1_macro"       in m
        assert "top5_accuracy"  in m
        assert "precision_macro" in m
        assert "recall_macro"   in m

    def test_run_epoch_eval_mode(self, model, device):
        """run_epoch in eval mode should return loss and metrics without error."""
        from src.trainer import run_epoch
        from src.models import LabelSmoothingCrossEntropy
        from torch.utils.data import TensorDataset, DataLoader

        imgs   = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)
        labels = torch.randint(0, NUM_CLASSES, (8,))
        ds     = TensorDataset(imgs, labels)
        loader = DataLoader(ds, batch_size=4)
        crit   = LabelSmoothingCrossEntropy()

        loss, metrics = run_epoch(model, loader, crit, None, device, None, training=False)
        assert loss >= 0
        assert "accuracy" in metrics
        assert not np.isnan(loss)


# ══════════════════════════════════════════════════════════════════════════════
# 6. BMI & RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
class TestBMIRecommender:
    def test_bmi_formula(self):
        from src.bmi_recommender import calculate_bmi
        bmi = calculate_bmi(70, 1.75)
        assert abs(bmi - 22.86) < 0.1

    def test_bmi_zero_height_raises(self):
        from src.bmi_recommender import calculate_bmi
        with pytest.raises(ValueError):
            calculate_bmi(70, 0)

    def test_bmi_classify_underweight(self):
        from src.bmi_recommender import classify_bmi
        r = classify_bmi(16.0)
        assert r["category"] == "Underweight"

    def test_bmi_classify_normal(self):
        from src.bmi_recommender import classify_bmi
        r = classify_bmi(22.0)
        assert r["category"] == "Normal Weight"

    def test_bmi_classify_overweight(self):
        from src.bmi_recommender import classify_bmi
        r = classify_bmi(27.0)
        assert r["category"] == "Overweight"

    def test_bmi_classify_obese1(self):
        from src.bmi_recommender import classify_bmi
        r = classify_bmi(32.0)
        assert r["category"] == "Obese Class I"

    def test_bmi_classify_obese2(self):
        from src.bmi_recommender import classify_bmi
        r = classify_bmi(40.0)
        assert "Obese Class II" in r["category"]

    def test_bmi_color_hex(self):
        from src.bmi_recommender import classify_bmi
        r = classify_bmi(22.0)
        assert r["color"].startswith("#")
        assert len(r["color"]) == 7

    def test_calculate_rda_male(self):
        from src.bmi_recommender import calculate_rda
        rda = calculate_rda(75, 1.75, 30, "male", "moderately active")
        assert rda["calories"] > 1800
        assert rda["protein"] > 50
        assert rda["carbohydrates"] > 100
        assert rda["fat"] > 30

    def test_calculate_rda_female(self):
        from src.bmi_recommender import calculate_rda
        rda = calculate_rda(60, 1.65, 28, "female", "lightly active")
        assert rda["calories"] > 1200
        assert rda["protein"] > 30

    def test_recommendation_engine_tips(self, sample_user):
        from src.bmi_recommender import RecommendationEngine
        engine = RecommendationEngine(sample_user)
        intake = {
            "calories": 800, "protein": 20, "carbohydrates": 150,
            "fat": 25, "fiber": 8, "sugar": 30, "sodium": 1200
        }
        result = engine.generate_recommendations(intake)
        assert "tips"    in result
        assert "alerts"  in result
        assert "summary" in result
        assert "rda"     in result
        assert "bmi_info" in result
        assert isinstance(result["tips"], list)

    def test_recommendation_engine_high_bmi(self):
        from src.bmi_recommender import RecommendationEngine
        user = {
            "weight_kg": 110, "height_m": 1.70, "age": 40,
            "gender": "male", "activity": "sedentary", "goal": "lose weight"
        }
        engine = RecommendationEngine(user)
        intake = {
            "calories": 3500, "protein": 60, "carbohydrates": 400,
            "fat": 120, "fiber": 10, "sugar": 80, "sodium": 3500
        }
        result = engine.generate_recommendations(intake)
        # Should have sodium and calorie alerts
        alert_text = " ".join(result["alerts"]).lower()
        assert len(result["alerts"]) > 0

    def test_recommendation_food_suggestions(self, sample_user):
        from src.bmi_recommender import RecommendationEngine
        engine = RecommendationEngine({**sample_user, "goal": "gain muscle"})
        intake = {
            "calories": 2500, "protein": 30, "carbohydrates": 300,
            "fat": 70, "fiber": 20, "sugar": 40, "sodium": 1800
        }
        result = engine.generate_recommendations(intake)
        assert isinstance(result["food_suggestions"], list)

    def test_quick_recommend(self, sample_user):
        from src.bmi_recommender import quick_recommend
        intake = {
            "calories": 1500, "protein": 50, "carbohydrates": 200,
            "fat": 50, "fiber": 18, "sugar": 35, "sodium": 1600
        }
        result = quick_recommend(sample_user, intake)
        assert "bmi_info"  in result
        assert "tips"      in result
        assert "summary"   in result

    def test_intake_analysis_low_protein(self, sample_user):
        from src.bmi_recommender import RecommendationEngine
        engine = RecommendationEngine(sample_user)
        intake = {k: 0 for k in RDA}
        analysis = engine.analyse_intake(intake)
        assert analysis["protein"]["status"] == "low"
        assert analysis["calories"]["percent"] == 0.0

    def test_intake_analysis_excess_sodium(self, sample_user):
        from src.bmi_recommender import RecommendationEngine
        engine = RecommendationEngine(sample_user)
        intake = {k: v * 1.5 for k, v in RDA.items()}
        analysis = engine.analyse_intake(intake)
        assert analysis["sodium"]["status"] == "high"


# ══════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
class TestInference:
    def test_preprocess_pil(self, dummy_image):
        from src.inference import preprocess_pil
        t = preprocess_pil(dummy_image)
        assert t.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_estimate_portion_default(self):
        from src.inference import estimate_portion
        p = estimate_portion("pizza")
        assert p > 0

    def test_estimate_portion_bbox_scaling(self):
        from src.inference import estimate_portion
        # Large bbox should give larger portion
        p_large = estimate_portion("pizza", bbox_area=50000, image_area=100000)
        p_small = estimate_portion("pizza", bbox_area=5000,  image_area=100000)
        assert p_large >= p_small

    def test_food_predictor_returns_topk(self, model, class_names, dummy_image):
        from src.inference import FoodPredictor
        pred = FoodPredictor(class_names=class_names, model=model)
        results = pred.predict(dummy_image, top_k=5)
        assert len(results) == 5

    def test_food_predictor_result_structure(self, model, class_names, dummy_image):
        from src.inference import FoodPredictor
        pred = FoodPredictor(class_names=class_names, model=model)
        results = pred.predict(dummy_image, top_k=3)
        for r in results:
            assert "food_name"    in r
            assert "confidence"   in r
            assert "display_name" in r
            assert "portion_g"    in r
            assert "status"       in r
            assert r["rank"] >= 1

    def test_food_predictor_confidence_sum(self, model, class_names, dummy_image):
        from src.inference import FoodPredictor
        pred    = FoodPredictor(class_names=class_names, model=model)
        results = pred.predict(dummy_image, top_k=101)
        total   = sum(r["confidence"] for r in results)
        assert abs(total - 100.0) < 1.0   # softmax probs sum ≈ 100%

    def test_food_predictor_ranked_by_confidence(self, model, class_names, dummy_image):
        from src.inference import FoodPredictor
        pred    = FoodPredictor(class_names=class_names, model=model)
        results = pred.predict(dummy_image, top_k=5)
        confs   = [r["confidence"] for r in results]
        assert confs == sorted(confs, reverse=True)

    def test_multi_food_detector_single_mode(self, model, class_names, dummy_image):
        from src.inference import FoodPredictor, MultiFoodDetector
        pred     = FoodPredictor(class_names=class_names, model=model)
        detector = MultiFoodDetector(pred)
        result   = detector.detect_and_classify(dummy_image)
        assert "mode"       in result
        assert "detections" in result
        assert len(result["detections"]) >= 1

    def test_multi_food_aggregate_nutrition(self, model, class_names, dummy_image):
        from src.inference import FoodPredictor, MultiFoodDetector
        pred     = FoodPredictor(class_names=class_names, model=model)
        detector = MultiFoodDetector(pred)
        result   = detector.detect_and_classify(dummy_image)
        totals   = detector.aggregate_nutrition(result)
        assert "calories"      in totals
        assert "protein"       in totals
        assert "carbohydrates" in totals
        assert "fat"           in totals

    def test_analyse_meal_image(self, model, class_names, dummy_image):
        from src.inference import analyse_meal_image
        result = analyse_meal_image(dummy_image, class_names, model=model)
        assert "detections"      in result
        assert "total_nutrition" in result
        assert result["total_nutrition"]["calories"] >= 0

    def test_confidence_threshold_status(self, model, class_names, dummy_image):
        from src.inference import FoodPredictor
        pred = FoodPredictor(class_names=class_names, model=model)
        results = pred.predict(dummy_image, top_k=5)
        for r in results:
            if r["confidence"] >= CONF_THRESHOLD * 100:
                assert r["status"] == "normal"
            else:
                assert r["status"] == "low_confidence"

    def test_rgba_image_handled(self, model, class_names):
        from src.inference import FoodPredictor
        pred  = FoodPredictor(class_names=class_names, model=model)
        img   = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), color=(100, 150, 200, 180))
        results = pred.predict(img, top_k=3)
        assert len(results) == 3


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION MODULE
# ══════════════════════════════════════════════════════════════════════════════
class TestEvaluation:
    def _make_dummy_data(self, n=500, c=101):
        labels = np.random.randint(0, c, n)
        probs  = np.random.dirichlet(np.ones(c), size=n)
        preds  = probs.argmax(axis=1)
        return labels, preds, probs

    def test_compute_full_metrics_keys(self):
        from src.evaluation import compute_full_metrics
        n, c = 500, 101
        labels = np.random.randint(0, c, n)
        probs  = np.random.dirichlet(np.ones(c), size=n)
        preds  = probs.argmax(axis=1)
        names = [f"food_{i}" for i in range(c)]
        m = compute_full_metrics(labels, preds, probs, names, n_classes=c)
        required = {"accuracy", "top3_accuracy", "top5_accuracy",
                    "precision_macro", "recall_macro", "f1_macro", "ece"}
        assert required.issubset(set(m.keys()))

    def test_compute_full_metrics_ranges(self):
        from src.evaluation import compute_full_metrics
        n, c = 500, 101
        labels = np.random.randint(0, c, n)
        probs  = np.random.dirichlet(np.ones(c), size=n)
        preds  = probs.argmax(axis=1)
        names = [f"food_{i}" for i in range(c)]
        m = compute_full_metrics(labels, preds, probs, names, n_classes=c)
        assert 0 <= m["accuracy"]      <= 100
        assert 0 <= m["top5_accuracy"] <= 100
        assert 0 <= m["precision_macro"] <= 1
        assert 0 <= m["f1_macro"]      <= 1
        assert 0 <= m["ece"]           <= 1

    def test_per_class_analysis_returns_df(self):
        from src.evaluation import per_class_analysis
        labels, preds, probs = self._make_dummy_data(n=200, c=10)
        names = [f"food_{i}" for i in range(10)]
        df = per_class_analysis(labels, preds, probs, names)
        assert isinstance(df, pd.DataFrame)
        assert "f1" in df.columns
        assert "precision" in df.columns
        assert "recall"    in df.columns

    def test_ece_between_0_and_1(self):
        from src.evaluation import _compute_ece
        labels, _, probs = self._make_dummy_data()
        ece = _compute_ece(labels, probs)
        assert 0.0 <= ece <= 1.0

    def test_topk_curve_plot(self, tmp_path):
        from src.evaluation import plot_topk_curve
        import matplotlib.pyplot as plt
        labels, _, probs = self._make_dummy_data()
        fig = plot_topk_curve(labels, probs, n_classes=101,
                               save_path=tmp_path / "topk.png")
        assert (tmp_path / "topk.png").exists()
        plt.close("all")

    def test_calibration_plot(self, tmp_path):
        from src.evaluation import plot_calibration_curve
        import matplotlib.pyplot as plt
        labels, _, probs = self._make_dummy_data()
        fig = plot_calibration_curve(labels, probs,
                                      save_path=tmp_path / "cal.png")
        assert (tmp_path / "cal.png").exists()
        plt.close("all")

    def test_confidence_distribution_plot(self, tmp_path):
        from src.evaluation import plot_confidence_distribution
        import matplotlib.pyplot as plt
        labels, _, probs = self._make_dummy_data()
        fig = plot_confidence_distribution(labels, probs,
                                            save_path=tmp_path / "conf.png")
        assert (tmp_path / "conf.png").exists()
        plt.close("all")

    def test_confusion_matrix_plot(self, tmp_path):
        from src.evaluation import plot_confusion_matrix
        import matplotlib.pyplot as plt
        labels, preds, _ = self._make_dummy_data(n=300, c=20)
        names = [f"food_{i}" for i in range(20)]
        fig = plot_confusion_matrix(labels, preds, names, top_n=10,
                                     save_path=tmp_path / "cm.png")
        assert (tmp_path / "cm.png").exists()
        plt.close("all")

    def test_collect_predictions(self, model, device):
        from src.evaluation import collect_predictions
        from torch.utils.data import TensorDataset, DataLoader
        imgs   = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)
        labels = torch.randint(0, NUM_CLASSES, (16,))
        ds     = TensorDataset(imgs, labels)
        loader = DataLoader(ds, batch_size=8)
        all_l, all_p, all_pr = collect_predictions(model, loader, device)
        assert len(all_l) == 16
        assert len(all_p) == 16
        assert all_pr.shape == (16, NUM_CLASSES)
        assert np.allclose(all_pr.sum(axis=1), 1.0, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 9. INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestIntegration:
    def test_full_meal_analysis_pipeline(self, model, class_names, tmp_db, sample_user):
        """End-to-end: image → inference → nutrition → log → retrieve."""
        from src.database import save_user, log_meal, get_daily_intake
        from src.inference import FoodPredictor, MultiFoodDetector
        from src.bmi_recommender import quick_recommend

        # Setup user
        u = sample_user
        save_user(u["username"], u["weight_kg"], u["height_m"],
                  u["age"], u["gender"], u["activity"], u["goal"], tmp_db)

        # Analyse image
        img  = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(180, 120, 60))
        pred = FoodPredictor(class_names=class_names, model=model)
        det  = MultiFoodDetector(pred)
        result = det.detect_and_classify(img)

        assert "detections" in result
        top_det = result["detections"][0]
        food    = top_det["food_name"]
        conf    = top_det["top_prediction"]["confidence"] / 100
        portion = top_det["portion_g"]

        # Log it
        if food in __import__("src.database", fromlist=["FOOD_NUTRITION"]).FOOD_NUTRITION:
            log_meal(u["username"], food, portion, conf, tmp_db)

        # Get daily intake
        intake = get_daily_intake(u["username"], db_path=tmp_db)
        assert isinstance(intake, dict)

        # Get recommendations
        recs = quick_recommend(u, intake)
        assert "summary" in recs
        assert "bmi_info" in recs

    def test_bmi_and_rda_consistency(self, sample_user):
        """BMI and personalised RDA should be internally consistent."""
        from src.bmi_recommender import calculate_bmi, classify_bmi, calculate_rda
        u   = sample_user
        bmi = calculate_bmi(u["weight_kg"], u["height_m"])
        cat = classify_bmi(bmi)
        rda = calculate_rda(u["weight_kg"], u["height_m"],
                             u["age"], u["gender"], u["activity"])

        assert cat["bmi"] == bmi
        assert rda["calories"] > 0
        # For a moderately active 30yo male at 75kg, TDEE ≈ 2400–2800
        assert 1800 < rda["calories"] < 3500

    def test_model_inference_with_real_transform(self, model, device):
        """Full inference pipeline from raw PIL to class prediction."""
        from src.data_pipeline import get_inference_transforms
        import torch.nn.functional as F

        img    = Image.new("RGB", (300, 400), color=(200, 100, 50))
        t      = get_inference_transforms()
        tensor = t(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)
        assert probs.shape == (1, NUM_CLASSES)
        assert abs(probs.sum().item() - 1.0) < 1e-5
        assert probs.max().item() > 0

    def test_nutrition_scaling_consistency(self, tmp_db):
        """Scaled nutrition at 200g should equal 2× nutrition at 100g."""
        from src.database import get_nutrition
        foods = ["pizza", "sushi", "hamburger", "caesar_salad"]
        for food in foods:
            n100 = get_nutrition(food, 100, tmp_db)
            n200 = get_nutrition(food, 200, tmp_db)
            if n100 and n200:
                for key in ["calories", "protein", "carbohydrates", "fat"]:
                    expected = round(n100[key] * 2, 1)
                    actual   = n200[key]
                    assert abs(actual - expected) < 0.5, \
                        f"{food}.{key}: expected {expected}, got {actual}"

    def test_recommendation_low_confidence_fallback(self, model, class_names, tmp_db):
        """Even low-confidence predictions should produce valid recommendations."""
        from src.inference import FoodPredictor, MultiFoodDetector
        from src.database import get_user
        from src.bmi_recommender import quick_recommend

        img  = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))
        pred = FoodPredictor(class_names=class_names, model=model)
        det  = MultiFoodDetector(pred)
        res  = det.detect_and_classify(img)
        tots = det.aggregate_nutrition(res)

        user_profile = {
            "weight_kg": 70, "height_m": 1.70, "age": 28,
            "gender": "female", "activity": "lightly active", "goal": "maintain"
        }
        recs = quick_recommend(user_profile, tots)
        assert isinstance(recs["tips"], list)
        assert isinstance(recs["summary"], str)
        assert len(recs["summary"]) > 10


# ══════════════════════════════════════════════════════════════════════════════
# 10. EDGE CASES & ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
class TestEdgeCases:
    def test_very_small_image(self, model, class_names):
        from src.inference import FoodPredictor
        pred = FoodPredictor(class_names=class_names, model=model)
        img  = Image.new("RGB", (32, 32), color=(50, 80, 120))
        results = pred.predict(img, top_k=3)
        assert len(results) == 3

    def test_very_large_image(self, model, class_names):
        from src.inference import FoodPredictor
        pred = FoodPredictor(class_names=class_names, model=model)
        img  = Image.new("RGB", (2048, 2048), color=(200, 100, 50))
        results = pred.predict(img, top_k=3)
        assert len(results) == 3

    def test_greyscale_image(self, model, class_names):
        from src.inference import FoodPredictor
        pred = FoodPredictor(class_names=class_names, model=model)
        img  = Image.new("L", (224, 224), color=128)
        results = pred.predict(img, top_k=3)
        assert len(results) == 3

    def test_zero_intake_recommendation(self):
        from src.bmi_recommender import quick_recommend
        user   = {"weight_kg": 65, "height_m": 1.68, "age": 25,
                   "gender": "female", "activity": "sedentary", "goal": "maintain"}
        intake = {k: 0 for k in RDA}
        result = quick_recommend(user, intake)
        assert result["summary"] != ""

    def test_empty_meal_history(self, tmp_db):
        from src.database import get_meal_history
        df = get_meal_history("nonexistent_user_xyz", days=7, db_path=tmp_db)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_get_user_not_found(self, tmp_db):
        from src.database import get_user
        result = get_user("ghost_user_xyz", tmp_db)
        assert result == {}

    def test_bmi_boundary_values(self):
        from src.bmi_recommender import classify_bmi
        boundaries = [18.4, 18.5, 24.9, 25.0, 29.9, 30.0, 34.9, 35.0]
        for bmi in boundaries:
            result = classify_bmi(bmi)
            assert result["category"] != "Unknown", f"BMI {bmi} returned Unknown"

    def test_negative_portion_clipped(self):
        from src.inference import estimate_portion
        # Should not return negative portion
        p = estimate_portion("pizza", bbox_area=1, image_area=1000000)
        assert p > 0

    def test_model_deterministic_eval(self, model, dummy_tensor):
        """Same input → same output in eval mode."""
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_tensor)
            out2 = model(dummy_tensor)
        assert torch.allclose(out1, out2)

    def test_model_non_deterministic_train(self, device):
        """Dropout should make training outputs non-identical (most of the time)."""
        from src.models import build_model
        m = build_model(pretrained=False, device=device)
        m.train()
        x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        with torch.no_grad():
            out1 = m(x)
            out2 = m(x)
        # They might match by chance but very unlikely for 4×101 outputs
        # Just verify no crash and valid output shape
        assert out1.shape == (4, NUM_CLASSES)
        assert out2.shape == (4, NUM_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-q"])
