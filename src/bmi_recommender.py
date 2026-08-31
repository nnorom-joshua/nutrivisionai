"""
src/bmi_recommender.py
BMI calculation, classification, and personalized dietary recommendation engine.
Hybrid: rule-based filtering + structured recommendation generation.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import BMI_CATEGORIES, RDA

log = logging.getLogger(__name__)


# ─── BMI Module ───────────────────────────────────────────────────────────────
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """WHO BMI formula: weight(kg) / height(m)²"""
    if height_m <= 0:
        raise ValueError("Height must be > 0")
    return round(weight_kg / (height_m ** 2), 2)


def classify_bmi(bmi: float) -> Dict:
    """Return BMI category, description, color, and risk level."""
    for low, high, label, color in BMI_CATEGORIES:
        if low <= bmi < high:
            risk_map = {
                "Underweight":    "Moderate",
                "Normal Weight":  "Low",
                "Overweight":     "Increased",
                "Obese Class I":  "High",
                "Obese Class II/III": "Very High",
            }
            return {
                "bmi":       bmi,
                "category":  label,
                "color":     color,
                "risk":      risk_map.get(label, "Unknown"),
            }
    return {"bmi": bmi, "category": "Unknown", "color": "#888888", "risk": "Unknown"}


def calculate_rda(weight_kg: float, height_m: float, age: int,
                  gender: str, activity: str) -> Dict:
    """
    Compute personalised RDA based on Harris-Benedict BMR + activity factor.
    """
    # ── BMR (Mifflin-St Jeor) ─────────────────────────────────────────────
    if gender.lower() in ("male", "m"):
        bmr = 10 * weight_kg + 6.25 * (height_m * 100) - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * (height_m * 100) - 5 * age - 161

    activity_factors = {
        "sedentary":     1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active":   1.725,
        "extra active":  1.9,
    }
    factor   = activity_factors.get(activity.lower(), 1.55)
    tdee     = bmr * factor  # Total Daily Energy Expenditure

    # ── Macros: standard split (20% protein, 50% carbs, 30% fat) ─────────
    protein_cal = tdee * 0.20
    carb_cal    = tdee * 0.50
    fat_cal     = tdee * 0.30

    return {
        "calories":      round(tdee),
        "protein":       round(protein_cal / 4),     # 4 kcal/g
        "carbohydrates": round(carb_cal   / 4),
        "fat":           round(fat_cal    / 9),       # 9 kcal/g
        "fiber":         RDA["fiber"],
        "sugar":         RDA["sugar"],
        "sodium":        RDA["sodium"],
    }


# ─── Rule-Based Recommendation Engine ────────────────────────────────────────
NUTRIENT_LABELS = {
    "calories":      ("kcal",  ""),
    "protein":       ("g",     ""),
    "carbohydrates": ("g",     ""),
    "fat":           ("g",     ""),
    "fiber":         ("g",     ""),
    "sugar":         ("g",     ""),
    "sodium":        ("mg",    ""),
}

HIGH_PROTEIN_FOODS = [
    "grilled salmon", "eggs benedict", "omelette", "beef tartare",
    "scallops", "filet mignon", "chicken curry", "edamame",
    "sashimi", "steak", "mussels",
]
LOW_CAL_FOODS = [
    "greek salad", "seaweed salad", "miso soup", "pho",
    "ceviche", "beet salad", "caesar salad", "oysters",
]
HIGH_FIBER_FOODS = [
    "guacamole", "edamame", "falafel", "hummus", "greek salad",
    "beet salad", "caesar salad",
]
LOW_SODIUM_FOODS = [
    "sashimi", "grilled salmon", "omelette", "steak", "pork chop",
    "filet mignon", "oysters",
]


class RecommendationEngine:
    """
    Generates personalised dietary recommendations from:
    - BMI classification
    - Daily nutritional intake vs personalised RDA
    - Meal history patterns
    - User goal (lose weight / maintain / gain muscle)
    """

    def __init__(self, user_profile: Dict):
        self.user       = user_profile
        self.weight_kg  = float(user_profile.get("weight_kg", 70))
        self.height_m   = float(user_profile.get("height_m",  1.70))
        self.age        = int(user_profile.get("age",         30))
        self.gender     = user_profile.get("gender",         "male")
        self.activity   = user_profile.get("activity",       "moderately active")
        self.goal       = user_profile.get("goal",           "maintain")

        self.bmi_info   = classify_bmi(calculate_bmi(self.weight_kg, self.height_m))
        self.rda        = calculate_rda(self.weight_kg, self.height_m,
                                        self.age, self.gender, self.activity)

        # Adjust RDA for goal
        if self.goal == "lose weight":
            self.rda["calories"] = int(self.rda["calories"] * 0.85)  # -15% deficit
        elif self.goal == "gain muscle":
            self.rda["calories"] = int(self.rda["calories"] * 1.10)  # +10% surplus
            self.rda["protein"]  = int(self.weight_kg * 1.8)          # 1.8g/kg

    def analyse_intake(self, daily_intake: Dict) -> Dict:
        """
        Compare daily_intake against personalised RDA.
        Returns dict of nutrient status (ok / low / high) and % achieved.
        """
        analysis = {}
        for nutrient, rda_val in self.rda.items():
            intake = daily_intake.get(nutrient, 0.0)
            pct    = (intake / rda_val * 100) if rda_val > 0 else 0
            if pct < 70:
                status = "low"
            elif pct > 120:
                status = "high"
            else:
                status = "ok"
            analysis[nutrient] = {
                "intake":  intake,
                "rda":     rda_val,
                "percent": round(pct, 1),
                "status":  status,
            }
        return analysis

    def generate_recommendations(self, daily_intake: Dict, meal_names: List[str] = None) -> Dict:
        """
        Full recommendation generation:
        1. Intake analysis vs RDA
        2. BMI-aware calorie guidance
        3. Specific food suggestions
        4. Alerts for critical values
        5. Summary paragraph
        """
        meal_names = meal_names or []
        analysis   = self.analyse_intake(daily_intake)
        bmi_cat    = self.bmi_info["category"]
        goal       = self.goal

        tips       = []
        alerts     = []
        food_sug   = []

        # ── Calorie guidance ──────────────────────────────────────────────
        cal_status = analysis.get("calories", {}).get("status", "ok")
        cal_pct    = analysis.get("calories", {}).get("percent", 0)

        if bmi_cat in ("Obese Class I", "Obese Class II/III"):
            alerts.append(" Your BMI indicates obesity. Strict calorie management is advised.")
            if cal_status == "high":
                tips.append(" You've exceeded your calorie target. Consider lighter meals for the rest of the day.")
            food_sug.extend(LOW_CAL_FOODS[:3])

        elif bmi_cat == "Overweight":
            tips.append(" Aim for a modest calorie deficit. Focus on whole foods and reduce processed options.")
            if cal_status != "high":
                food_sug.extend(LOW_CAL_FOODS[:2])

        elif bmi_cat == "Underweight":
            alerts.append(" Your BMI suggests underweight. Increasing caloric and protein intake is recommended.")
            tips.append(" Add more calorie-dense, nutritious foods such as eggs, salmon, and legumes.")
            food_sug.extend(HIGH_PROTEIN_FOODS[:3])

        else:  # Normal
            tips.append("Your BMI is in the healthy range. Maintain your balanced diet.")

        # ── Protein guidance ───────────────────────────────────────────────
        prot_status = analysis.get("protein", {}).get("status", "ok")
        if prot_status == "low":
            tips.append(" Your protein intake is below target. Consider adding lean proteins to your next meal.")
            food_sug.extend(HIGH_PROTEIN_FOODS[:2])
        elif goal == "gain muscle" and prot_status != "high":
            tips.append(f" For muscle gain, target {self.rda['protein']}g of protein daily. "
                        f"You're currently at {daily_intake.get('protein', 0):.0f}g.")
            food_sug.extend(HIGH_PROTEIN_FOODS[:2])

        # ── Fiber guidance ─────────────────────────────────────────────────
        fiber_status = analysis.get("fiber", {}).get("status", "ok")
        if fiber_status == "low":
            tips.append(" Low fiber detected. Include more vegetables, legumes, or whole grains.")
            food_sug.extend(HIGH_FIBER_FOODS[:2])

        # ── Sodium alert ───────────────────────────────────────────────────
        sodium_status = analysis.get("sodium", {}).get("status", "ok")
        if sodium_status == "high":
            alerts.append("🧂 High sodium detected. Excess sodium increases hypertension risk.")
            tips.append("Try fresh herbs and spices instead of salt, and reduce processed foods.")
            food_sug.extend(LOW_SODIUM_FOODS[:2])

        # ── Sugar alert ────────────────────────────────────────────────────
        sugar_status = analysis.get("sugar", {}).get("status", "ok")
        if sugar_status == "high":
            alerts.append(" Sugar intake exceeds recommended limits. Reduce sugary beverages and desserts.")

        # ── Carbohydrate ───────────────────────────────────────────────────
        carb_pct = analysis.get("carbohydrates", {}).get("percent", 0)
        if carb_pct > 130:
            tips.append(" Carbohydrate intake is elevated. Consider substituting refined carbs with whole grains, "
                        "vegetables, or lean proteins.")

        # ── Completeness tip ──────────────────────────────────────────────
        if cal_pct < 50:
            tips.append(f" You've only consumed {cal_pct:.0f}% of your daily calorie target. "
                        "Ensure you eat enough to meet your energy needs.")

        # ── Food suggestion dedup ─────────────────────────────────────────
        seen  = set(meal_names)
        final_sug = []
        for food in food_sug:
            if food not in seen:
                final_sug.append(food.replace("_", " ").title())
                seen.add(food)
        final_sug = final_sug[:5]

        # ── Summary paragraph ─────────────────────────────────────────────
        summary = self._build_summary(bmi_cat, cal_pct, prot_status, fiber_status, goal)

        return {
            "bmi_info":       self.bmi_info,
            "rda":            self.rda,
            "analysis":       analysis,
            "tips":           tips,
            "alerts":         alerts,
            "food_suggestions": final_sug,
            "summary":        summary,
            "goal":           goal,
        }

    def _build_summary(self, bmi_cat, cal_pct, prot_status, fiber_status, goal) -> str:
        lines = [
            f"Based on your BMI classification ({bmi_cat}) and today's nutritional intake, "
            f"here is a personalised summary:"
        ]

        if bmi_cat == "Normal Weight":
            lines.append(
                "You are maintaining a healthy weight. Continue focusing on balanced macronutrients "
                "and regular physical activity to sustain your current health status."
            )
        elif bmi_cat == "Overweight":
            lines.append(
                "A moderate calorie deficit combined with increased physical activity can help you "
                "progress toward a healthier weight. Prioritise vegetables, lean proteins, and whole grains."
            )
        elif bmi_cat in ("Obese Class I", "Obese Class II/III"):
            lines.append(
                "Medical consultation and a structured dietary plan are strongly recommended. "
                "Focus on whole, minimally processed foods and establish consistent meal timing."
            )
        elif bmi_cat == "Underweight":
            lines.append(
                "Increasing energy and protein intake is important. Include calorie-dense, "
                "nutrient-rich foods across all meals and consider consulting a registered dietitian."
            )

        if goal == "lose weight":
            lines.append("Your goal is weight loss — maintain a sustainable deficit of 300–500 kcal/day.")
        elif goal == "gain muscle":
            lines.append("Your goal is muscle gain — ensure a protein-rich diet and resistance training.")
        else:
            lines.append("Your goal is to maintain — keep a consistent routine of balanced eating and exercise.")

        return " ".join(lines)


# ─── Convenience functions ────────────────────────────────────────────────────
def get_bmi_summary(weight_kg: float, height_m: float) -> Dict:
    bmi      = calculate_bmi(weight_kg, height_m)
    bmi_info = classify_bmi(bmi)
    return bmi_info


def quick_recommend(user_profile: Dict, daily_intake: Dict,
                    meal_names: List[str] = None) -> Dict:
    engine = RecommendationEngine(user_profile)
    return engine.generate_recommendations(daily_intake, meal_names)


if __name__ == "__main__":
    # Smoke test
    profile = {
        "weight_kg": 85, "height_m": 1.75, "age": 32,
        "gender": "male", "activity": "moderately active", "goal": "lose weight"
    }
    intake = {
        "calories": 1200, "protein": 45, "carbohydrates": 180,
        "fat": 40, "fiber": 12, "sugar": 45, "sodium": 2800
    }
    result = quick_recommend(profile, intake, meal_names=["pizza", "hamburger"])
    print("BMI:", result["bmi_info"])
    print("Tips:", result["tips"])
    print("Alerts:", result["alerts"])
    print("Suggestions:", result["food_suggestions"])
