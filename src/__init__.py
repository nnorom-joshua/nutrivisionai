# src/__init__.py
from src.database import build_database, get_nutrition, log_meal, get_daily_intake, get_meal_history, save_user, get_user, FOOD101_CLASSES, FOOD_NUTRITION
from src.models import build_model, load_model, FoodClassifier, LabelSmoothingCrossEntropy
from src.data_pipeline import get_dataloaders, get_train_transforms, get_val_transforms, preprocess_image, DataCleaner
from src.trainer import Trainer, evaluate_on_test, run_epoch, EarlyStopping
from src.hyperparameter_tuning import run_model_selection, run_hyperparameter_search, full_tuning_pipeline
from src.inference import FoodPredictor, MultiFoodDetector, analyse_meal_image
from src.bmi_recommender import calculate_bmi, classify_bmi, calculate_rda, RecommendationEngine, quick_recommend
from src.evaluation import run_full_evaluation, compute_full_metrics, per_class_analysis, collect_predictions
