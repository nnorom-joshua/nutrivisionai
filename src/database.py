"""
src/database.py
Builds and manages the SQLite nutritional database.
Maps all 101 Food-101 classes to nutritional values (per 100g serving).
Data sourced from USDA FoodData Central averages.
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import DB_PATH, NUTRITION_CSV

# ─── Full Food-101 Nutritional Data (per 100g) ────────────────────────────────
# Fields: calories(kcal), protein(g), carbs(g), fat(g), fiber(g), sugar(g), sodium(mg)
FOOD_NUTRITION = {
    "apple_pie":           {"calories": 237, "protein": 2.0, "carbohydrates": 34.0, "fat": 11.0, "fiber": 1.5, "sugar": 17.0, "sodium": 207},
    "baby_back_ribs":      {"calories": 258, "protein": 23.0, "carbohydrates": 5.0, "fat": 16.0, "fiber": 0.0, "sugar": 2.0, "sodium": 530},
    "baklava":             {"calories": 428, "protein": 5.6, "carbohydrates": 52.0, "fat": 23.0, "fiber": 1.8, "sugar": 28.0, "sodium": 230},
    "beef_carpaccio":      {"calories": 164, "protein": 18.0, "carbohydrates": 0.5, "fat": 10.0, "fiber": 0.0, "sugar": 0.0, "sodium": 320},
    "beef_tartare":        {"calories": 196, "protein": 20.0, "carbohydrates": 1.0, "fat": 12.0, "fiber": 0.0, "sugar": 0.5, "sodium": 380},
    "beet_salad":          {"calories": 74,  "protein": 2.2, "carbohydrates": 13.0, "fat": 2.0, "fiber": 2.8, "sugar": 9.0, "sodium": 210},
    "beignets":            {"calories": 320, "protein": 5.0, "carbohydrates": 45.0, "fat": 14.0, "fiber": 1.0, "sugar": 12.0, "sodium": 280},
    "bibimbap":            {"calories": 152, "protein": 8.5, "carbohydrates": 20.0, "fat": 4.5, "fiber": 2.5, "sugar": 3.0, "sodium": 490},
    "bread_pudding":       {"calories": 218, "protein": 5.8, "carbohydrates": 34.0, "fat": 7.0, "fiber": 0.8, "sugar": 18.0, "sodium": 230},
    "breakfast_burrito":   {"calories": 218, "protein": 10.0, "carbohydrates": 24.0, "fat": 9.5, "fiber": 2.0, "sugar": 2.0, "sodium": 590},
    "bruschetta":          {"calories": 195, "protein": 5.0, "carbohydrates": 30.0, "fat": 6.0, "fiber": 2.0, "sugar": 3.0, "sodium": 350},
    "caesar_salad":        {"calories": 108, "protein": 4.0, "carbohydrates": 7.0, "fat": 7.5, "fiber": 1.5, "sugar": 1.5, "sodium": 390},
    "cannoli":             {"calories": 337, "protein": 7.0, "carbohydrates": 38.0, "fat": 17.5, "fiber": 0.5, "sugar": 22.0, "sodium": 135},
    "caprese_salad":       {"calories": 127, "protein": 8.0, "carbohydrates": 4.0, "fat": 9.0, "fiber": 0.8, "sugar": 3.0, "sodium": 310},
    "carrot_cake":         {"calories": 353, "protein": 3.5, "carbohydrates": 52.0, "fat": 16.0, "fiber": 1.2, "sugar": 35.0, "sodium": 340},
    "ceviche":             {"calories": 90,  "protein": 14.0, "carbohydrates": 5.0, "fat": 2.0, "fiber": 1.0, "sugar": 2.5, "sodium": 430},
    "cheesecake":          {"calories": 321, "protein": 5.5, "carbohydrates": 35.0, "fat": 18.0, "fiber": 0.4, "sugar": 26.0, "sodium": 210},
    "cheese_plate":        {"calories": 372, "protein": 20.0, "carbohydrates": 3.0, "fat": 31.0, "fiber": 0.0, "sugar": 0.5, "sodium": 620},
    "chicken_curry":       {"calories": 165, "protein": 15.0, "carbohydrates": 8.0, "fat": 8.5, "fiber": 1.5, "sugar": 3.0, "sodium": 450},
    "chicken_quesadilla":  {"calories": 226, "protein": 14.0, "carbohydrates": 20.0, "fat": 9.5, "fiber": 1.2, "sugar": 1.5, "sodium": 510},
    "chicken_wings":       {"calories": 266, "protein": 20.0, "carbohydrates": 5.5, "fat": 18.5, "fiber": 0.2, "sugar": 0.5, "sodium": 590},
    "chocolate_cake":      {"calories": 371, "protein": 4.5, "carbohydrates": 54.0, "fat": 17.0, "fiber": 2.0, "sugar": 38.0, "sodium": 330},
    "chocolate_mousse":    {"calories": 254, "protein": 4.5, "carbohydrates": 26.0, "fat": 15.0, "fiber": 1.5, "sugar": 22.0, "sodium": 80},
    "churros":             {"calories": 328, "protein": 4.0, "carbohydrates": 45.0, "fat": 15.0, "fiber": 1.5, "sugar": 14.0, "sodium": 310},
    "clam_chowder":        {"calories": 90,  "protein": 5.0, "carbohydrates": 10.0, "fat": 3.5, "fiber": 0.5, "sugar": 2.0, "sodium": 520},
    "club_sandwich":       {"calories": 266, "protein": 17.0, "carbohydrates": 26.0, "fat": 10.0, "fiber": 2.0, "sugar": 3.0, "sodium": 780},
    "crab_cakes":          {"calories": 178, "protein": 12.0, "carbohydrates": 10.0, "fat": 10.0, "fiber": 0.5, "sugar": 1.0, "sodium": 520},
    "creme_brulee":        {"calories": 272, "protein": 4.0, "carbohydrates": 28.0, "fat": 16.0, "fiber": 0.0, "sugar": 25.0, "sodium": 65},
    "croque_madame":       {"calories": 290, "protein": 15.0, "carbohydrates": 22.0, "fat": 16.0, "fiber": 1.0, "sugar": 4.0, "sodium": 680},
    "cup_cakes":           {"calories": 380, "protein": 3.5, "carbohydrates": 56.0, "fat": 17.0, "fiber": 0.5, "sugar": 42.0, "sodium": 320},
    "deviled_eggs":        {"calories": 147, "protein": 9.5, "carbohydrates": 1.5, "fat": 11.5, "fiber": 0.0, "sugar": 0.5, "sodium": 220},
    "donuts":              {"calories": 400, "protein": 5.5, "carbohydrates": 52.0, "fat": 20.0, "fiber": 1.0, "sugar": 23.0, "sodium": 390},
    "dumplings":           {"calories": 175, "protein": 7.5, "carbohydrates": 26.0, "fat": 5.0, "fiber": 1.2, "sugar": 2.0, "sodium": 400},
    "edamame":             {"calories": 122, "protein": 11.0, "carbohydrates": 9.0, "fat": 5.2, "fiber": 5.2, "sugar": 2.2, "sodium": 63},
    "eggs_benedict":       {"calories": 292, "protein": 13.0, "carbohydrates": 22.0, "fat": 17.0, "fiber": 1.0, "sugar": 3.0, "sodium": 730},
    "escargots":           {"calories": 172, "protein": 19.0, "carbohydrates": 5.0, "fat": 8.5, "fiber": 0.0, "sugar": 0.0, "sodium": 345},
    "falafel":             {"calories": 333, "protein": 13.0, "carbohydrates": 32.0, "fat": 18.0, "fiber": 5.0, "sugar": 2.0, "sodium": 294},
    "filet_mignon":        {"calories": 206, "protein": 26.0, "carbohydrates": 0.0, "fat": 11.0, "fiber": 0.0, "sugar": 0.0, "sodium": 70},
    "fish_and_chips":      {"calories": 283, "protein": 16.0, "carbohydrates": 27.0, "fat": 12.0, "fiber": 2.0, "sugar": 1.0, "sodium": 510},
    "foie_gras":           {"calories": 462, "protein": 11.5, "carbohydrates": 4.5, "fat": 44.0, "fiber": 0.0, "sugar": 0.0, "sodium": 510},
    "french_fries":        {"calories": 312, "protein": 3.5, "carbohydrates": 41.0, "fat": 15.0, "fiber": 3.5, "sugar": 0.5, "sodium": 400},
    "french_onion_soup":   {"calories": 78,  "protein": 3.5, "carbohydrates": 10.0, "fat": 2.5, "fiber": 1.0, "sugar": 4.0, "sodium": 610},
    "french_toast":        {"calories": 229, "protein": 7.0, "carbohydrates": 30.0, "fat": 9.5, "fiber": 1.0, "sugar": 12.0, "sodium": 330},
    "fried_calamari":      {"calories": 228, "protein": 16.0, "carbohydrates": 16.0, "fat": 10.0, "fiber": 0.5, "sugar": 0.5, "sodium": 430},
    "fried_rice":          {"calories": 174, "protein": 4.5, "carbohydrates": 26.0, "fat": 5.5, "fiber": 1.0, "sugar": 1.5, "sodium": 620},
    "frozen_yogurt":       {"calories": 127, "protein": 3.5, "carbohydrates": 26.0, "fat": 1.5, "fiber": 0.0, "sugar": 20.0, "sodium": 75},
    "garlic_bread":        {"calories": 350, "protein": 8.0, "carbohydrates": 44.0, "fat": 16.0, "fiber": 2.0, "sugar": 2.5, "sodium": 580},
    "gnocchi":             {"calories": 176, "protein": 4.0, "carbohydrates": 36.0, "fat": 2.0, "fiber": 1.5, "sugar": 1.5, "sodium": 420},
    "greek_salad":         {"calories": 115, "protein": 3.5, "carbohydrates": 7.0, "fat": 8.5, "fiber": 2.0, "sugar": 4.5, "sodium": 450},
    "grilled_cheese_sandwich": {"calories": 290, "protein": 12.0, "carbohydrates": 26.0, "fat": 16.0, "fiber": 1.0, "sugar": 3.5, "sodium": 620},
    "grilled_salmon":      {"calories": 208, "protein": 20.0, "carbohydrates": 0.0, "fat": 13.5, "fiber": 0.0, "sugar": 0.0, "sodium": 310},
    "guacamole":           {"calories": 155, "protein": 2.0, "carbohydrates": 9.0, "fat": 14.0, "fiber": 6.5, "sugar": 1.0, "sodium": 270},
    "gyoza":               {"calories": 212, "protein": 9.0, "carbohydrates": 24.0, "fat": 9.0, "fiber": 1.5, "sugar": 2.0, "sodium": 510},
    "hamburger":           {"calories": 295, "protein": 17.0, "carbohydrates": 24.0, "fat": 14.0, "fiber": 1.5, "sugar": 5.0, "sodium": 540},
    "hot_and_sour_soup":   {"calories": 56,  "protein": 4.0, "carbohydrates": 7.0, "fat": 1.5, "fiber": 0.5, "sugar": 2.0, "sodium": 820},
    "hot_dog":             {"calories": 290, "protein": 11.0, "carbohydrates": 24.0, "fat": 17.0, "fiber": 1.0, "sugar": 5.0, "sodium": 670},
    "huevos_rancheros":    {"calories": 185, "protein": 9.5, "carbohydrates": 15.0, "fat": 10.0, "fiber": 3.0, "sugar": 3.5, "sodium": 490},
    "hummus":              {"calories": 177, "protein": 7.5, "carbohydrates": 20.0, "fat": 9.5, "fiber": 6.0, "sugar": 0.5, "sodium": 430},
    "ice_cream":           {"calories": 207, "protein": 3.5, "carbohydrates": 24.0, "fat": 11.0, "fiber": 0.7, "sugar": 21.0, "sodium": 80},
    "lasagna":             {"calories": 166, "protein": 9.5, "carbohydrates": 16.5, "fat": 7.0, "fiber": 1.5, "sugar": 4.0, "sodium": 520},
    "lobster_bisque":      {"calories": 95,  "protein": 6.5, "carbohydrates": 8.0, "fat": 4.5, "fiber": 0.5, "sugar": 3.0, "sodium": 580},
    "lobster_roll_sandwich": {"calories": 245, "protein": 15.5, "carbohydrates": 24.0, "fat": 10.0, "fiber": 1.0, "sugar": 3.5, "sodium": 610},
    "macaroni_and_cheese": {"calories": 220, "protein": 9.0, "carbohydrates": 26.0, "fat": 8.5, "fiber": 1.0, "sugar": 4.5, "sodium": 610},
    "macarons":            {"calories": 392, "protein": 7.0, "carbohydrates": 60.0, "fat": 15.0, "fiber": 1.0, "sugar": 50.0, "sodium": 50},
    "miso_soup":           {"calories": 40,  "protein": 3.0, "carbohydrates": 5.0, "fat": 1.5, "fiber": 0.8, "sugar": 1.5, "sodium": 980},
    "mussels":             {"calories": 86,  "protein": 12.0, "carbohydrates": 4.0, "fat": 2.5, "fiber": 0.0, "sugar": 0.0, "sodium": 320},
    "nachos":              {"calories": 306, "protein": 8.0, "carbohydrates": 34.0, "fat": 16.0, "fiber": 3.5, "sugar": 2.5, "sodium": 640},
    "omelette":            {"calories": 154, "protein": 11.0, "carbohydrates": 1.5, "fat": 12.0, "fiber": 0.0, "sugar": 1.0, "sodium": 420},
    "onion_rings":         {"calories": 276, "protein": 3.5, "carbohydrates": 31.0, "fat": 15.5, "fiber": 2.0, "sugar": 3.0, "sodium": 480},
    "oysters":             {"calories": 51,  "protein": 5.5, "carbohydrates": 3.0, "fat": 1.5, "fiber": 0.0, "sugar": 0.0, "sodium": 195},
    "pad_thai":            {"calories": 196, "protein": 8.5, "carbohydrates": 29.0, "fat": 6.0, "fiber": 2.0, "sugar": 4.5, "sodium": 620},
    "paella":              {"calories": 183, "protein": 12.0, "carbohydrates": 21.0, "fat": 5.5, "fiber": 1.5, "sugar": 2.0, "sodium": 550},
    "pancakes":            {"calories": 227, "protein": 6.0, "carbohydrates": 38.0, "fat": 6.5, "fiber": 1.0, "sugar": 10.0, "sodium": 450},
    "panna_cotta":         {"calories": 153, "protein": 2.5, "carbohydrates": 17.0, "fat": 8.5, "fiber": 0.0, "sugar": 16.0, "sodium": 45},
    "peking_duck":         {"calories": 337, "protein": 19.0, "carbohydrates": 14.0, "fat": 24.0, "fiber": 0.5, "sugar": 7.0, "sodium": 470},
    "pho":                 {"calories": 73,  "protein": 6.0, "carbohydrates": 8.5, "fat": 1.5, "fiber": 0.8, "sugar": 1.5, "sodium": 720},
    "pizza":               {"calories": 266, "protein": 11.0, "carbohydrates": 33.0, "fat": 10.0, "fiber": 2.3, "sugar": 3.6, "sodium": 598},
    "pork_chop":           {"calories": 231, "protein": 27.0, "carbohydrates": 0.0, "fat": 13.0, "fiber": 0.0, "sugar": 0.0, "sodium": 68},
    "poutine":             {"calories": 265, "protein": 8.0, "carbohydrates": 30.0, "fat": 13.0, "fiber": 2.5, "sugar": 2.0, "sodium": 610},
    "prime_rib":           {"calories": 315, "protein": 27.0, "carbohydrates": 0.0, "fat": 22.0, "fiber": 0.0, "sugar": 0.0, "sodium": 75},
    "pulled_pork_sandwich":{"calories": 264, "protein": 21.0, "carbohydrates": 20.0, "fat": 10.5, "fiber": 1.0, "sugar": 8.0, "sodium": 640},
    "ramen":               {"calories": 144, "protein": 8.0, "carbohydrates": 22.0, "fat": 3.5, "fiber": 1.0, "sugar": 2.5, "sodium": 780},
    "ravioli":             {"calories": 186, "protein": 8.5, "carbohydrates": 28.0, "fat": 5.0, "fiber": 1.5, "sugar": 2.5, "sodium": 440},
    "red_velvet_cake":     {"calories": 367, "protein": 4.0, "carbohydrates": 54.0, "fat": 16.0, "fiber": 0.8, "sugar": 38.0, "sodium": 380},
    "risotto":             {"calories": 185, "protein": 5.5, "carbohydrates": 30.0, "fat": 5.5, "fiber": 1.0, "sugar": 2.0, "sodium": 460},
    "samosa":              {"calories": 308, "protein": 6.5, "carbohydrates": 38.0, "fat": 15.0, "fiber": 3.5, "sugar": 2.5, "sodium": 380},
    "sashimi":             {"calories": 127, "protein": 20.0, "carbohydrates": 0.0, "fat": 5.0, "fiber": 0.0, "sugar": 0.0, "sodium": 310},
    "scallops":            {"calories": 111, "protein": 20.5, "carbohydrates": 5.5, "fat": 1.5, "fiber": 0.0, "sugar": 0.0, "sodium": 390},
    "seaweed_salad":       {"calories": 65,  "protein": 1.5, "carbohydrates": 11.0, "fat": 2.0, "fiber": 2.5, "sugar": 5.0, "sodium": 600},
    "shrimp_and_grits":    {"calories": 205, "protein": 14.0, "carbohydrates": 18.0, "fat": 8.5, "fiber": 1.0, "sugar": 2.0, "sodium": 580},
    "spaghetti_bolognese": {"calories": 171, "protein": 10.0, "carbohydrates": 22.0, "fat": 5.5, "fiber": 2.0, "sugar": 5.5, "sodium": 480},
    "spaghetti_carbonara": {"calories": 268, "protein": 12.5, "carbohydrates": 30.0, "fat": 11.5, "fiber": 1.5, "sugar": 2.0, "sodium": 520},
    "spring_rolls":        {"calories": 166, "protein": 5.0, "carbohydrates": 24.0, "fat": 6.0, "fiber": 2.0, "sugar": 3.5, "sodium": 370},
    "steak":               {"calories": 271, "protein": 26.0, "carbohydrates": 0.0, "fat": 18.0, "fiber": 0.0, "sugar": 0.0, "sodium": 59},
    "strawberry_shortcake":{"calories": 264, "protein": 4.0, "carbohydrates": 40.0, "fat": 10.0, "fiber": 1.5, "sugar": 22.0, "sodium": 210},
    "sushi":               {"calories": 143, "protein": 5.5, "carbohydrates": 28.0, "fat": 1.5, "fiber": 0.5, "sugar": 4.0, "sodium": 440},
    "tacos":               {"calories": 218, "protein": 12.0, "carbohydrates": 20.0, "fat": 10.0, "fiber": 2.5, "sugar": 2.5, "sodium": 520},
    "takoyaki":            {"calories": 209, "protein": 9.0, "carbohydrates": 27.0, "fat": 7.5, "fiber": 0.8, "sugar": 3.0, "sodium": 570},
    "tiramisu":            {"calories": 240, "protein": 5.0, "carbohydrates": 27.0, "fat": 13.0, "fiber": 0.5, "sugar": 18.0, "sodium": 90},
    "tuna_tartare":        {"calories": 128, "protein": 17.0, "carbohydrates": 3.0, "fat": 5.5, "fiber": 0.5, "sugar": 1.5, "sodium": 360},
    "waffles":             {"calories": 291, "protein": 7.5, "carbohydrates": 41.0, "fat": 12.0, "fiber": 1.5, "sugar": 11.0, "sodium": 540},
}

# All 101 Food-101 class names (sorted)
FOOD101_CLASSES = sorted(FOOD_NUTRITION.keys())


def build_database(db_path: Path = DB_PATH) -> None:
    """Create / rebuild the SQLite nutritional database."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ── Food nutrition table ─────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS food_nutrition (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            food_name   TEXT    UNIQUE NOT NULL,
            class_idx   INTEGER NOT NULL,
            calories    REAL    NOT NULL,
            protein     REAL    NOT NULL,
            carbohydrates REAL  NOT NULL,
            fat         REAL    NOT NULL,
            fiber       REAL    NOT NULL,
            sugar       REAL    NOT NULL,
            sodium      REAL    NOT NULL,
            serving_g   REAL    DEFAULT 100.0
        )
    """)

    # ── User table ────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            weight_kg   REAL,
            height_m    REAL,
            age         INTEGER,
            gender      TEXT,
            activity    TEXT,
            goal        TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Meal logs table ────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meal_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL,
            food_name     TEXT    NOT NULL,
            portion_g     REAL    NOT NULL,
            calories      REAL,
            protein       REAL,
            carbohydrates REAL,
            fat           REAL,
            fiber         REAL,
            sugar         REAL,
            sodium        REAL,
            confidence    REAL,
            logged_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Insert nutrition data ──────────────────────────────────────────────────
    for idx, (name, n) in enumerate(sorted(FOOD_NUTRITION.items())):
        cur.execute("""
            INSERT OR REPLACE INTO food_nutrition
            (food_name, class_idx, calories, protein, carbohydrates, fat, fiber, sugar, sodium)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, idx, n["calories"], n["protein"], n["carbohydrates"],
              n["fat"], n["fiber"], n["sugar"], n["sodium"]))

    conn.commit()
    conn.close()
    print(f"[DB] Database built at {db_path} with {len(FOOD_NUTRITION)} food entries.")


def get_nutrition(food_name: str, portion_g: float = 100.0, db_path: Path = DB_PATH) -> dict:
    """
    Retrieve nutritional values for a food item scaled to the given portion size.
    Returns dict with all macro/micronutrient values.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT * FROM food_nutrition WHERE food_name = ?", (food_name,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return {}

    cols  = ["id", "food_name", "class_idx", "calories", "protein",
             "carbohydrates", "fat", "fiber", "sugar", "sodium", "serving_g"]
    data  = dict(zip(cols, row))
    scale = portion_g / 100.0

    return {
        "food_name":     data["food_name"],
        "portion_g":     portion_g,
        "calories":      round(data["calories"]      * scale, 1),
        "protein":       round(data["protein"]       * scale, 1),
        "carbohydrates": round(data["carbohydrates"] * scale, 1),
        "fat":           round(data["fat"]           * scale, 1),
        "fiber":         round(data["fiber"]         * scale, 1),
        "sugar":         round(data["sugar"]         * scale, 1),
        "sodium":        round(data["sodium"]        * scale, 1),
    }


def log_meal(username: str, food_name: str, portion_g: float,
             confidence: float, db_path: Path = DB_PATH) -> None:
    """Insert a meal entry into the meal_logs table."""
    nutrition = get_nutrition(food_name, portion_g, db_path)
    if not nutrition:
        return
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO meal_logs
        (username, food_name, portion_g, calories, protein, carbohydrates,
         fat, fiber, sugar, sodium, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, food_name, portion_g,
          nutrition["calories"], nutrition["protein"], nutrition["carbohydrates"],
          nutrition["fat"], nutrition["fiber"], nutrition["sugar"], nutrition["sodium"],
          confidence))
    conn.commit()
    conn.close()


def get_daily_intake(username: str, date_str: str = None, db_path: Path = DB_PATH) -> dict:
    """Sum all nutrients logged by a user on a given date (default: today)."""
    import datetime
    if date_str is None:
        date_str = datetime.date.today().isoformat()

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        SELECT SUM(calories), SUM(protein), SUM(carbohydrates),
               SUM(fat), SUM(fiber), SUM(sugar), SUM(sodium)
        FROM meal_logs
        WHERE username = ?
          AND DATE(logged_at) = ?
    """, (username, date_str))
    row = cur.fetchone()
    conn.close()

    if row is None or row[0] is None:
        return {k: 0.0 for k in ["calories","protein","carbohydrates","fat","fiber","sugar","sodium"]}

    return {
        "calories":      round(row[0] or 0, 1),
        "protein":       round(row[1] or 0, 1),
        "carbohydrates": round(row[2] or 0, 1),
        "fat":           round(row[3] or 0, 1),
        "fiber":         round(row[4] or 0, 1),
        "sugar":         round(row[5] or 0, 1),
        "sodium":        round(row[6] or 0, 1),
    }


def get_meal_history(username: str, days: int = 7, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return the last N days of meal logs for a user as a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT food_name, portion_g, calories, protein, carbohydrates,
               fat, fiber, sugar, sodium, confidence,
               DATE(logged_at) as date, TIME(logged_at) as time
        FROM meal_logs
        WHERE username = ?
          AND logged_at >= datetime('now', ?)
        ORDER BY logged_at DESC
    """, conn, params=(username, f"-{days} days"))
    conn.close()
    return df


def save_user(username: str, weight_kg: float, height_m: float,
              age: int, gender: str, activity: str, goal: str,
              db_path: Path = DB_PATH) -> None:
    """Upsert user health profile."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO users (username, weight_kg, height_m, age, gender, activity, goal)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            weight_kg=excluded.weight_kg,
            height_m=excluded.height_m,
            age=excluded.age,
            gender=excluded.gender,
            activity=excluded.activity,
            goal=excluded.goal
    """, (username, weight_kg, height_m, age, gender, activity, goal))
    conn.commit()
    conn.close()


def get_user(username: str, db_path: Path = DB_PATH) -> dict:
    """Retrieve user profile dict."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return {}
    cols = ["id","username","weight_kg","height_m","age","gender","activity","goal","created_at"]
    return dict(zip(cols, row))


if __name__ == "__main__":
    build_database()
    # Quick sanity check
    n = get_nutrition("pizza", 200)
    print("Pizza 200g:", n)
