"""
src/inference.py
Real-time inference pipeline:
- Single image → EfficientNet-B0 → top-k food predictions
- Multi-food images → YOLOv8 detect ROIs → EfficientNet per ROI
- Portion estimation via reference object scaling
- Returns structured prediction results with nutritional lookup
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import io

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    NUM_CLASSES, CONF_THRESHOLD, IMAGE_SIZE,
    DATASET_MEAN, DATASET_STD, BEST_MODEL_PATH,
    YOLO_MODEL, YOLO_CONF, YOLO_IOU, DB_PATH,
    SELECTED_MODEL
)
from src.models import load_model, build_model, FoodClassifier
from src.database import get_nutrition, FOOD101_CLASSES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─── Transform ────────────────────────────────────────────────────────────────
INFER_TRANSFORM = T.Compose([
    T.Resize(int(IMAGE_SIZE * 1.14)),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
])


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    return INFER_TRANSFORM(image.convert("RGB")).unsqueeze(0)


# ─── Portion Estimator ────────────────────────────────────────────────────────
# Rough average serving sizes (grams) for each Food-101 class
# Used when no reference object is detected
DEFAULT_PORTION_G = {
    "apple_pie": 150, "baby_back_ribs": 300, "baklava": 60,
    "beef_carpaccio": 120, "beef_tartare": 150, "beet_salad": 180,
    "beignets": 80,  "bibimbap": 350, "bread_pudding": 180,
    "breakfast_burrito": 220, "bruschetta": 100, "caesar_salad": 200,
    "cannoli": 90,   "caprese_salad": 180, "carrot_cake": 120,
    "ceviche": 180,  "cheesecake": 120, "cheese_plate": 100,
    "chicken_curry": 300, "chicken_quesadilla": 200, "chicken_wings": 250,
    "chocolate_cake": 120, "chocolate_mousse": 120, "churros": 100,
    "clam_chowder": 300, "club_sandwich": 250, "crab_cakes": 180,
    "creme_brulee": 150, "croque_madame": 200, "cup_cakes": 100,
    "deviled_eggs": 80, "donuts": 80, "dumplings": 150,
    "edamame": 120, "eggs_benedict": 250, "escargots": 120,
    "falafel": 150, "filet_mignon": 200, "fish_and_chips": 350,
    "foie_gras": 80, "french_fries": 150, "french_onion_soup": 350,
    "french_toast": 200, "fried_calamari": 200, "fried_rice": 300,
    "frozen_yogurt": 180, "garlic_bread": 100, "gnocchi": 250,
    "greek_salad": 250, "grilled_cheese_sandwich": 200, "grilled_salmon": 200,
    "guacamole": 100, "gyoza": 150, "hamburger": 250,
    "hot_and_sour_soup": 350, "hot_dog": 180, "huevos_rancheros": 250,
    "hummus": 100, "ice_cream": 150, "lasagna": 300,
    "lobster_bisque": 350, "lobster_roll_sandwich": 250, "macaroni_and_cheese": 300,
    "macarons": 40, "miso_soup": 300, "mussels": 250,
    "nachos": 200, "omelette": 200, "onion_rings": 150,
    "oysters": 120, "pad_thai": 350, "paella": 350,
    "pancakes": 200, "panna_cotta": 120, "peking_duck": 250,
    "pho": 400, "pizza": 200, "pork_chop": 200,
    "poutine": 300, "prime_rib": 300, "pulled_pork_sandwich": 300,
    "ramen": 400, "ravioli": 250, "red_velvet_cake": 120,
    "risotto": 300, "samosa": 100, "sashimi": 150,
    "scallops": 150, "seaweed_salad": 150, "shrimp_and_grits": 300,
    "spaghetti_bolognese": 350, "spaghetti_carbonara": 300, "spring_rolls": 120,
    "steak": 250, "strawberry_shortcake": 150, "sushi": 180,
    "tacos": 200, "takoyaki": 150, "tiramisu": 150,
    "tuna_tartare": 150, "waffles": 180,
}


def estimate_portion(food_name: str, bbox_area: Optional[float] = None,
                     image_area: Optional[float] = None) -> float:
    """
    Estimate portion size in grams.
    If bbox and image areas are provided, scale default portion by bbox ratio.
    """
    base = DEFAULT_PORTION_G.get(food_name, 150)
    if bbox_area and image_area and image_area > 0:
        area_ratio = min(bbox_area / image_area, 1.0)
        # Map ratio to portion multiplier: 0.1→0.5x, 0.5→1x, 1.0→2x
        scale = max(0.4, min(2.0, area_ratio * 4))
        return round(base * scale)
    return float(base)


# ── Class-agnostic region segmentation (YOLO fallback) ───────────────────────
def segment_food_regions(
    image: Image.Image,
    min_area_ratio: float = 0.03,
    max_area_ratio: float = 0.65,
    max_regions: int = 4,
) -> List[Tuple[int, int, int, int]]:
    """
    Splits an image into candidate food regions without needing to know what
    the food is. Used when YOLO (trained on COCO's 80 classes) fails to find
    any boxes -- which happens for most Food-101 dishes since they aren't
    COCO categories.

    Strategy:
      1. Cluster pixels by colour in LAB space into a handful of groups.
      2. Identify the cluster that most resembles background/plate (the one
         with the most image-border pixels) and discard it.
      3. Extract contours -> bounding boxes from the remaining clusters.
      4. Drop boxes that are too small, too large (likely still background),
         or near-duplicates of a larger box.

    Returns a list of (x1, y1, x2, y2) boxes. Classification of each crop is
    still done downstream by FoodPredictor -- this function only proposes
    regions, it never predicts a class itself.
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    img_area = h * w

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)

    k = max_regions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape((h, w))

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = True
    border_mask[:, 0] = border_mask[:, -1] = True
    border_counts = [int(((labels == cid) & border_mask).sum()) for cid in range(k)]
    bg_cluster = int(np.argmax(border_counts))

    boxes = []
    for cid in range(k):
        if cid == bg_cluster:
            continue
        mask = np.uint8(labels == cid) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            ratio = area / img_area
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((x, y, x + bw, y + bh, area))

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[4], reverse=True)
    kept = []
    for b in boxes:
        x1, y1, x2, y2, area = b
        is_dup = False
        for kx1, ky1, kx2, ky2, _ in kept:
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / area > 0.85:
                is_dup = True
                break
        if not is_dup:
            kept.append(b)
        if len(kept) >= max_regions:
            break

    return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in kept]


# ─── Classifier Inference ─────────────────────────────────────────────────────
class FoodPredictor:
    """
    Wraps FoodClassifier for single-image and batch inference.
    """
    def __init__(
        self,
        class_names: List[str],
        model: Optional[FoodClassifier] = None,
        device: Optional[torch.device]  = None,
    ):
        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            self._load_or_build()

    def _load_or_build(self):
        if BEST_MODEL_PATH.exists():
            log.info(f"[Predictor] Loading checkpoint from {BEST_MODEL_PATH}")
            self.model = load_model(BEST_MODEL_PATH, arch=SELECTED_MODEL,
                                    n_classes=NUM_CLASSES, device=self.device)
        else:
            log.warning("[Predictor] No checkpoint found — using randomly initialised model (for demo).")
            self.model = build_model(arch=SELECTED_MODEL, n_classes=NUM_CLASSES,
                                     pretrained=True, device=self.device)
            self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """
        Run inference on a PIL image.
        Returns list of top-k predictions sorted by confidence.
        """
        tensor = preprocess_pil(image).to(self.device)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

        top_probs, top_idxs = probs.topk(top_k)
        top_probs = top_probs.cpu().numpy()
        top_idxs  = top_idxs.cpu().numpy()

        results = []
        for prob, idx in zip(top_probs, top_idxs):
            food_name  = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            conf       = float(prob)
            status     = "normal" if conf >= CONF_THRESHOLD else "low_confidence"
            portion_g  = estimate_portion(food_name)
            nutrition  = get_nutrition(food_name, portion_g) if conf >= CONF_THRESHOLD else {}

            results.append({
                "rank":        len(results) + 1,
                "food_name":   food_name,
                "display_name": food_name.replace("_", " ").title(),
                "confidence":  round(conf * 100, 2),
                "status":      status,
                "portion_g":   portion_g,
                "nutrition":   nutrition,
            })

        return results

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image]) -> List[List[Dict]]:
        """Run predict() on each image in the list."""
        return [self.predict(img) for img in images]


# ─── Multi-food Detector (YOLO) ───────────────────────────────────────────────
class MultiFoodDetector:
    """
    Uses YOLOv8 to detect multiple food items in a single image.
    Each detected ROI is cropped and passed to FoodPredictor.
    Falls back gracefully if ultralytics not installed or model unavailable.
    """
    def __init__(self, predictor: FoodPredictor):
        self.predictor = predictor
        self.yolo      = None
        self._load_yolo()

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(YOLO_MODEL)
            log.info("[YOLO] YOLOv8 loaded.")
        except Exception as e:
            log.warning(f"[YOLO] Could not load YOLOv8: {e}. Falling back to single-food mode.")

    def detect_and_classify(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> Dict:
        """
        Detect food regions via YOLO (if available), classify each crop.
        Falls back to full-image classification if YOLO unavailable.

        Returns:
            {
              "mode": "multi" | "single",
              "detections": [
                {
                  "bbox": [x1, y1, x2, y2],
                  "yolo_class": str,
                  "predictions": [...],   # from FoodPredictor
                  "top_prediction": {...},
                  "portion_g": float,
                  "nutrition": {...},
                },
                ...
              ]
            }
        """
        img_w, img_h = image.size
        img_area     = img_w * img_h

        if self.yolo is not None:
            try:
                results = self.yolo(np.array(image), conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
                boxes   = results[0].boxes
                detections = []

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_w, x2), min(img_h, y2)

                        # Skip tiny boxes
                        if (x2 - x1) < 20 or (y2 - y1) < 20:
                            continue

                        crop      = image.crop((x1, y1, x2, y2))
                        bbox_area = (x2 - x1) * (y2 - y1)
                        preds     = self.predictor.predict(crop, top_k=top_k)
                        top_pred  = preds[0] if preds else {}
                        food_name = top_pred.get("food_name", "unknown")
                        portion_g = estimate_portion(food_name, bbox_area, img_area)
                        nutrition = get_nutrition(food_name, portion_g)

                        detections.append({
                            "bbox":         [x1, y1, x2, y2],
                            "yolo_conf":    float(box.conf[0]),
                            "predictions":  preds,
                            "top_prediction": top_pred,
                            "food_name":    food_name,
                            "display_name": food_name.replace("_", " ").title(),
                            "portion_g":    portion_g,
                            "nutrition":    nutrition,
                        })

                    if detections:
                        return {"mode": "multi", "detections": detections}

            except Exception as e:
                log.warning(f"[YOLO] Detection failed: {e}. Trying segmentation fallback.")

        # ── Fallback 1: class-agnostic segmentation ──────────────────────────
        # YOLO (COCO-pretrained) returns no boxes for most Food-101 dishes
        # since they aren't COCO categories. Before giving up to single-image
        # mode, try splitting the image into food-shaped regions by colour/
        # texture and classify each one independently.
        try:
            regions = segment_food_regions(image)
        except Exception as e:
            log.warning(f"[Segmentation] Failed: {e}. Using full-image fallback.")
            regions = []

        if len(regions) > 1:
            detections = []
            for (x1, y1, x2, y2) in regions:
                crop      = image.crop((x1, y1, x2, y2))
                bbox_area = (x2 - x1) * (y2 - y1)
                preds     = self.predictor.predict(crop, top_k=top_k)
                top_pred  = preds[0] if preds else {}
                food_name = top_pred.get("food_name", "unknown")
                portion_g = estimate_portion(food_name, bbox_area, img_area)
                nutrition = get_nutrition(food_name, portion_g)

                detections.append({
                    "bbox":         [x1, y1, x2, y2],
                    "yolo_conf":    None,
                    "source":       "segmentation",
                    "predictions":  preds,
                    "top_prediction": top_pred,
                    "food_name":    food_name,
                    "display_name": food_name.replace("_", " ").title(),
                    "portion_g":    portion_g,
                    "nutrition":    nutrition,
                })

            if detections:
                return {"mode": "multi", "detections": detections}

        # ── Fallback 2: full-image single classification ────────────────────
        preds     = self.predictor.predict(image, top_k=top_k)
        top_pred  = preds[0] if preds else {}
        food_name = top_pred.get("food_name", "unknown")
        portion_g = estimate_portion(food_name)
        nutrition = get_nutrition(food_name, portion_g)

        return {
            "mode": "single",
            "detections": [{
                "bbox":         [0, 0, img_w, img_h],
                "yolo_conf":    None,
                "predictions":  preds,
                "top_prediction": top_pred,
                "food_name":    food_name,
                "display_name": food_name.replace("_", " ").title(),
                "portion_g":    portion_g,
                "nutrition":    nutrition,
            }]
        }

    def aggregate_nutrition(self, detection_result: Dict) -> Dict:
        """Sum nutritional values across all detected food items."""
        totals = {
            "calories": 0.0, "protein": 0.0, "carbohydrates": 0.0,
            "fat": 0.0, "fiber": 0.0, "sugar": 0.0, "sodium": 0.0,
        }
        for det in detection_result.get("detections", []):
            n = det.get("nutrition", {})
            for key in totals:
                totals[key] += n.get(key, 0.0)

        return {k: round(v, 1) for k, v in totals.items()}


# ─── Convenience wrapper ──────────────────────────────────────────────────────
def analyse_meal_image(
    image: Image.Image,
    class_names: List[str],
    model: Optional[FoodClassifier] = None,
) -> Dict:
    """
    One-call interface: returns full detection + nutrition result for a meal image.
    """
    predictor = FoodPredictor(class_names=class_names, model=model)
    detector  = MultiFoodDetector(predictor)
    result    = detector.detect_and_classify(image)
    total_nutrition = detector.aggregate_nutrition(result)
    result["total_nutrition"] = total_nutrition
    return result


if __name__ == "__main__":
    # Smoke test with a blank image
    test_img = Image.new("RGB", (224, 224), color=(200, 150, 100))
    class_names = sorted(DEFAULT_PORTION_G.keys())
    result = analyse_meal_image(test_img, class_names)
    print("Mode:", result["mode"])
    print("Top prediction:", result["detections"][0]["top_prediction"])
    print("Total nutrition:", result["total_nutrition"])
