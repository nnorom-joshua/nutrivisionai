"""
src/multi_food_detector.py

Streamlit-app-only variant of the multi-food detection pipeline.

WHY THIS FILE EXISTS
---------------------
`src/inference.py` is also imported by the training/Colab notebook, so it's
left completely untouched. This module reuses everything from
`src/inference.py` that didn't need to change (FoodPredictor, estimate_portion,
segment_food_regions, DEFAULT_PORTION_G) and only reimplements
`MultiFoodDetector`, with the multi-food bug fixed.

THE BUG THIS FIXES
-------------------
The original `MultiFoodDetector.detect_and_classify` returned immediately in
"multi" mode as soon as YOLO found *any* single box, and only fell back to
class-agnostic segmentation when YOLO found zero boxes. Since the bundled
YOLOv8n model is COCO-pretrained (only a handful of its 80 classes are food —
pizza, hot dog, cake, donut, sandwich, banana, etc.), it typically detects at
most one item on a real plate. Every other food on the plate — rice, chicken,
fries, veggies, none of which are COCO classes — was silently dropped because
segmentation never got the chance to run alongside it.

Fix: YOLO and segmentation now both always run, their results are merged
(de-duplicating segmentation regions that overlap an existing YOLO box), and
`mode` is derived from the final merged count rather than being set the
moment YOLO fired at all.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from configs.config import YOLO_MODEL, YOLO_CONF, YOLO_IOU
from src.database import get_nutrition
from src.inference import (
    FoodPredictor,
    estimate_portion,
    segment_food_regions,
)

log = logging.getLogger(__name__)


class MultiFoodDetector:
    """
    Uses YOLOv8 *and* class-agnostic colour/texture segmentation together to
    detect multiple food items in a single image. Each detected ROI is
    cropped and passed to FoodPredictor. Falls back gracefully at every
    stage — a failure in one detector degrades to the next instead of
    crashing, down to a guaranteed full-image classification as the last
    resort.
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
            log.warning(f"[YOLO] Could not load YOLOv8: {e}. Segmentation-only mode.")

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """Standard IoU between two (x1,y1,x2,y2) boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _classify_box(self, image: Image.Image, box: Tuple[int, int, int, int],
                       img_area: int, top_k: int, yolo_conf: Optional[float] = None,
                       source: str = "yolo") -> Dict:
        x1, y1, x2, y2 = box
        crop      = image.crop((x1, y1, x2, y2))
        bbox_area = (x2 - x1) * (y2 - y1)
        preds     = self.predictor.predict(crop, top_k=top_k)
        top_pred  = preds[0] if preds else {}
        food_name = top_pred.get("food_name", "unknown")
        portion_g = estimate_portion(food_name, bbox_area, img_area)
        nutrition = get_nutrition(food_name, portion_g)

        return {
            "bbox":         [x1, y1, x2, y2],
            "yolo_conf":    yolo_conf,
            "source":       source,
            "predictions":  preds,
            "top_prediction": top_pred,
            "food_name":    food_name,
            "display_name": food_name.replace("_", " ").title(),
            "portion_g":    portion_g,
            "nutrition":    nutrition,
        }

    def _run_yolo(self, image: Image.Image, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, float]]:
        """Returns a list of (x1, y1, x2, y2, conf) boxes. Never raises."""
        if self.yolo is None:
            return []
        try:
            results = self.yolo(np.array(image), conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
            boxes = results[0].boxes
            out = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue
                    out.append((x1, y1, x2, y2, float(box.conf[0])))
            return out
        except Exception as e:
            log.warning(f"[YOLO] Detection failed: {e}.")
            return []

    def detect_and_classify(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> Dict:
        """
        Detect food regions and classify each crop.

        Strategy:
          1. Run YOLO — good for the handful of COCO-overlapping foods
             (pizza, hot dog, cake, sandwich, banana, etc).
          2. ALWAYS also run class-agnostic colour/texture segmentation —
             most Food-101 dishes aren't COCO categories, so YOLO alone
             typically finds at most one (or zero) of several plated items.
          3. Merge the two candidate sets, dropping segmentation regions
             that overlap an existing YOLO box (IoU > 0.4) so the same
             physical item isn't classified twice.
          4. Classify every surviving region. mode is "multi" only when
             more than one distinct item was found — otherwise "single".

        Returns:
            {
              "mode": "multi" | "single",
              "detections": [
                {
                  "bbox": [x1, y1, x2, y2],
                  "source": "yolo" | "segmentation" | "full_image",
                  "yolo_conf": float | None,
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

        yolo_boxes = self._run_yolo(image, img_w, img_h)

        # Class-agnostic segmentation always runs too — it's what actually
        # catches multi-food plates, since most dishes aren't COCO classes.
        try:
            seg_regions = segment_food_regions(image)
        except Exception as e:
            log.warning(f"[Segmentation] Failed: {e}.")
            seg_regions = []

        candidate_boxes: List[Tuple[Tuple[int, int, int, int], Optional[float], str]] = []
        for (x1, y1, x2, y2, conf) in yolo_boxes:
            candidate_boxes.append(((x1, y1, x2, y2), conf, "yolo"))

        for (x1, y1, x2, y2) in seg_regions:
            box = (x1, y1, x2, y2)
            # Skip a segmentation region that's basically a duplicate of a
            # YOLO box already found (same physical food item).
            if any(self._iou(box, yb[:4]) > 0.4 for yb in yolo_boxes):
                continue
            candidate_boxes.append((box, None, "segmentation"))

        detections = []
        for box, conf, source in candidate_boxes:
            try:
                detections.append(
                    self._classify_box(image, box, img_area, top_k, conf, source)
                )
            except Exception as e:
                log.warning(f"[Classify] Skipping region {box}: {e}")

        if len(detections) >= 1:
            mode = "multi" if len(detections) > 1 else "single"
            return {"mode": mode, "detections": detections}

        # ── Last-resort fallback: full-image single classification ──────────
        try:
            preds = self.predictor.predict(image, top_k=top_k)
        except Exception as e:
            log.warning(f"[Classify] Full-image classification failed: {e}")
            preds = []
        top_pred  = preds[0] if preds else {}
        food_name = top_pred.get("food_name", "unknown")
        portion_g = estimate_portion(food_name)
        nutrition = get_nutrition(food_name, portion_g) if food_name != "unknown" else {}

        return {
            "mode": "single",
            "detections": [{
                "bbox":         [0, 0, img_w, img_h],
                "yolo_conf":    None,
                "source":       "full_image",
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


# ─── Convenience wrapper (same interface as src.inference.analyse_meal_image) ──
def analyse_meal_image(
    image: Image.Image,
    class_names: List[str],
    model=None,
) -> Dict:
    """One-call interface: returns full detection + nutrition result for a meal image."""
    predictor = FoodPredictor(class_names=class_names, model=model)
    detector  = MultiFoodDetector(predictor)
    result    = detector.detect_and_classify(image)
    total_nutrition = detector.aggregate_nutrition(result)
    result["total_nutrition"] = total_nutrition
    return result
