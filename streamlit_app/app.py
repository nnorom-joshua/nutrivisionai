"""
streamlit_app/app.py
NutriVision AI — Full Streamlit Application
Pages:
  1. 🏠 Home / Dashboard
  2. 📸 Analyse Meal
  # 3. 📊 Daily Tracker
  4. 👤 My Profile
  5. 📈 Model Insights
  6. ℹ️  About
"""

import sys
import json
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON, PRIMARY_COLOR, ACCENT_COLOR,
    RDA, BMI_CATEGORIES, CONF_THRESHOLD, DB_PATH, LOGS_DIR, BEST_MODEL_PATH,
    NUM_CLASSES, SELECTED_MODEL
)
from src.database import (
    build_database, get_nutrition, log_meal,
    get_daily_intake, get_meal_history, save_user, get_user,
    FOOD101_CLASSES, FOOD_NUTRITION
)

# The trained checkpoint has a 20-way output head (Chapter 3.4; verified in
# configs/config.py as NUM_CLASSES=20), not a 101-way one, so the predictor
# must be loaded with the twenty selected training classes, not the full
# Food-101 label set. Verified directly against configs/food_classes.py in
# the joshua-nnorom/nutritionalai repository.
# FOOD101_CLASSES is still used elsewhere (e.g. manual entry) since the
# nutrition database covers all 101 classes even though only twenty were
# trained.
from configs.food_classes import SELECTED_CLASSES
from src.bmi_recommender import (
    calculate_bmi, classify_bmi, calculate_rda,
    RecommendationEngine, quick_recommend
)
from src.inference import FoodPredictor, MultiFoodDetector, analyse_meal_image

logging.basicConfig(level=logging.WARNING)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Global font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  /* Hide default header */
  #MainMenu {{visibility: hidden;}}
  header {{visibility: hidden;}}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: linear-gradient(160deg, #0f172a 0%, #1e3a5f 100%);
    color: white;
  }}
  [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {{
    color: #cbd5e1 !important;
  }}
  [data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: white;
  }}

  /* Metric cards */
  .metric-card {{
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border-left: 4px solid {PRIMARY_COLOR};
    margin-bottom: 12px;
  }}
  .metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {PRIMARY_COLOR};
    margin: 0;
  }}
  .metric-label {{
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  /* Nutrient bars */
  .nutrient-row {{
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
  }}
  .nutrient-name {{
    width: 130px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #374151;
  }}
  .bar-wrap {{
    flex: 1;
    background: #f1f5f9;
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
  }}
  .nutrient-val {{
    width: 90px;
    font-size: 0.8rem;
    color: #64748b;
    text-align: right;
  }}

  /* BMI dial area */
  .bmi-badge {{
    display: inline-block;
    padding: 8px 20px;
    border-radius: 99px;
    font-weight: 700;
    font-size: 1rem;
    color: white;
    margin: 8px 0;
  }}

  /* Food card */
  .food-card {{
    background: white;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 10px;
  }}
  .food-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
  }}
  .confidence-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    background: {ACCENT_COLOR}22;
    color: {ACCENT_COLOR};
  }}

  /* Tip card */
  .tip-card {{
    background: #f0fdf4;
    border-left: 4px solid {ACCENT_COLOR};
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #166534;
    font-size: 0.9rem;
  }}
  .alert-card {{
    background: #fff7ed;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #92400e;
    font-size: 0.9rem;
  }}

  /* Header banner */
  .app-header {{
    background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, #1e40af 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 18px;
    margin-bottom: 24px;
  }}
  .app-header h1 {{ margin: 0; font-size: 2rem; }}
  .app-header p  {{ margin: 4px 0 0; opacity: 0.85; font-size: 1rem; }}

  /* Section title */
  .section-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
  }}

  /* Upload zone */
  [data-testid="stFileUploader"] {{
    border: 2px dashed #93c5fd;
    border-radius: 14px;
    padding: 12px;
    background: #eff6ff;
  }}

  /* Table styling */
  .dataframe {{ border-radius: 10px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialisation ─────────────────────────────────────────────
def init_session():
    defaults = {
        "username":       "guest",
        "logged_in":      False,
        "user_profile":   {},
        "last_result":    None,
        "predictor":      None,
        "detector":       None,
        "class_names":    SELECTED_CLASSES,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Load Model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model …")
def load_predictor(class_names):
    predictor = FoodPredictor(class_names=class_names)
    detector  = MultiFoodDetector(predictor)
    return predictor, detector


# ─── DB bootstrap ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def ensure_db():
    build_database()
    return True


# ─── Plotting helpers ─────────────────────────────────────────────────────────
def nutrient_bar_html(label: str, intake: float, rda: float,
                      unit: str, emoji: str) -> str:
    pct   = min((intake / rda * 100) if rda > 0 else 0, 140)
    color = "#22c55e" if pct <= 100 else "#ef4444"
    width = min(pct, 100)
    return f"""
    <div class="nutrient-row">
      <div class="nutrient-name">{emoji} {label}</div>
      <div class="bar-wrap"><div class="bar-fill" style="width:{width:.0f}%;background:{color};"></div></div>
      <div class="nutrient-val">{intake:.0f}/{rda}{unit}</div>
    </div>"""


def macro_donut(nutrition: Dict, title: str = "Macros") -> go.Figure:
    labels = ["Protein", "Carbs", "Fat"]
    values = [nutrition.get("protein", 0), nutrition.get("carbohydrates", 0), nutrition.get("fat", 0)]
    colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    fig    = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors),
        textinfo="percent+label",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e293b")),
        showlegend=False, margin=dict(t=40, b=10, l=10, r=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def history_line(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("date")[["calories","protein","carbohydrates","fat"]].sum().reset_index()
    fig   = go.Figure()
    colors = {"calories": "#ef4444", "protein": "#3b82f6",
              "carbohydrates": "#f59e0b", "fat": "#8b5cf6"}
    for col in ["calories", "protein", "carbohydrates", "fat"]:
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily[col], name=col.title(),
            mode="lines+markers", line=dict(color=colors[col], width=2),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title="7-Day Nutritional History",
        xaxis_title="Date", yaxis_title="Amount",
        height=320, margin=dict(t=40, b=30, l=40, r=10),
        legend=dict(orientation="h", y=1.1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    )
    return fig


def bmi_gauge(bmi: float, category: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={"text": f"BMI — {category}", "font": {"size": 14, "color": "#1e293b"}},
        number={"font": {"size": 32, "color": color}},
        gauge={
            "axis":  {"range": [10, 45], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar":   {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [10,  18.5], "color": "#bfdbfe"},
                {"range": [18.5, 25],  "color": "#bbf7d0"},
                {"range": [25,   30],  "color": "#fde68a"},
                {"range": [30,   35],  "color": "#fecaca"},
                {"range": [35,   45],  "color": "#e9d5ff"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75, "value": bmi,
            },
        },
    ))
    fig.update_layout(
        height=250, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 0 10px;">
          <div style="font-size:3rem;">{APP_ICON}</div>
          <div style="font-size:1.2rem;font-weight:700;color:white;">{APP_TITLE}</div>
          <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">AI Nutritional Analysis</div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.1);margin:10px 0 20px;">
        """, unsafe_allow_html=True)

        username = st.text_input("👤 Your Name / Username",
                                  value=st.session_state["username"],
                                  key="username_input")
        if username:
            st.session_state["username"] = username

        st.markdown("---")
        page = st.selectbox(
            "Navigate",
            ["🏠 Home", "📸 Analyse Meal", "📊 Daily Tracker",
             "👤 My Profile", "ℹ️  About"],
            label_visibility="collapsed",
        )

        
        # Your selectbox code
        page = st.selectbox(
            "Navigate", 
            ["🏠 Home", "📸 Analyse Meal", "📊 Daily Tracker", "👤 My Profile", "ℹ️ About"], 
            label_visibility="collapsed"
        )
        # "📈 Model Insights"
        # Quick stats in sidebar
        if st.session_state["username"] != "guest":
            st.markdown("---")
            today = datetime.date.today().isoformat()
            intake = get_daily_intake(st.session_state["username"], today)
            rda_cal = RDA["calories"]
            pct = int(intake["calories"] / rda_cal * 100) if rda_cal > 0 else 0
            st.markdown(f"""
            <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
              Today's Progress
            </div>
            <div style="color:white;font-size:1.5rem;font-weight:700;">{intake['calories']:.0f}
              <span style="font-size:0.9rem;font-weight:400;color:#94a3b8;">/ {rda_cal} kcal</span>
            </div>
            <div style="background:rgba(255,255,255,0.1);border-radius:99px;height:6px;margin:6px 0;">
              <div style="background:{ACCENT_COLOR};width:{min(pct,100)}%;height:100%;border-radius:99px;"></div>
            </div>
            <div style="color:#94a3b8;font-size:0.75rem;">{pct}% of daily goal</div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="color:#475569;font-size:0.7rem;text-align:center;">NutriVision AI v1.0<br>Real-Time Nutritional Analysis</div>',
                    unsafe_allow_html=True)

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown(f"""
    <div class="app-header">
      <h1>{APP_ICON} {APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.session_state["username"]
    today    = datetime.date.today().isoformat()
    intake   = get_daily_intake(username, today)
    user     = get_user(username)

    # ── KPI row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
          <p class="metric-label">🔥 Calories Today</p>
          <p class="metric-value">{intake['calories']:.0f}</p>
          <p style="color:#94a3b8;font-size:0.8rem;margin:0;">of {RDA['calories']} kcal goal</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
          <p class="metric-label">💪 Protein</p>
          <p class="metric-value">{intake['protein']:.0f}g</p>
          <p style="color:#94a3b8;font-size:0.8rem;margin:0;">of {RDA['protein']}g goal</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
          <p class="metric-label">🍞 Carbohydrates</p>
          <p class="metric-value">{intake['carbohydrates']:.0f}g</p>
          <p style="color:#94a3b8;font-size:0.8rem;margin:0;">of {RDA['carbohydrates']}g goal</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
          <p class="metric-label">🥑 Fat</p>
          <p class="metric-value">{intake['fat']:.0f}g</p>
          <p style="color:#94a3b8;font-size:0.8rem;margin:0;">of {RDA['fat']}g goal</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Nutrient progress + Macro donut ──────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">📊 Daily Nutrient Progress</div>', unsafe_allow_html=True)
        rda_now = RDA
        if user:
            try:
                rda_now = calculate_rda(
                    float(user.get("weight_kg", 70)),
                    float(user.get("height_m",  1.70)),
                    int(user.get("age", 30)),
                    user.get("gender", "male"),
                    user.get("activity", "moderately active"),
                )
            except Exception:
                pass

        nutrient_cfg = [
            ("Calories", "calories", "kcal", "🔥"),
            ("Protein",  "protein",  "g",    "💪"),
            ("Carbs",    "carbohydrates", "g", "🍞"),
            ("Fat",      "fat",      "g",    "🥑"),
            ("Fiber",    "fiber",    "g",    "🥦"),
            ("Sugar",    "sugar",    "g",    "🍬"),
            ("Sodium",   "sodium",   "mg",   "🧂"),
        ]
        html = ""
        for label, key, unit, emoji in nutrient_cfg:
            html += nutrient_bar_html(label, intake.get(key, 0), rda_now.get(key, RDA[key]), unit, emoji)
        st.markdown(html, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">🥧 Macronutrient Split</div>', unsafe_allow_html=True)
        fig = macro_donut(intake, "Today's Macros")
        st.plotly_chart(fig, use_container_width=True)

    # ── BMI card if profile set ───────────────────────────────────────────
    if user and user.get("weight_kg") and user.get("height_m"):
        st.markdown("---")
        st.markdown('<div class="section-title">⚖️ Your BMI</div>', unsafe_allow_html=True)
        bmi      = calculate_bmi(float(user["weight_kg"]), float(user["height_m"]))
        bmi_info = classify_bmi(bmi)
        col_bmi, col_tips = st.columns([1, 2])
        with col_bmi:
            st.plotly_chart(bmi_gauge(bmi, bmi_info["category"], bmi_info["color"]),
                            use_container_width=True)
            st.caption(
                "ℹ️ BMI is a screening signal, not a diagnosis. It can't tell fat "
                "from muscle, and WHO's standard thresholds are known to under- "
                "or over-classify risk for some ethnic groups."
            )
        with col_tips:
            recs = quick_recommend(
                {**user, "goal": user.get("goal", "maintain")}, intake
            )
            st.markdown('<div class="section-title">💡 Today\'s Tips</div>', unsafe_allow_html=True)
            for tip in recs.get("tips", [])[:3]:
                st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)
            for alert in recs.get("alerts", []):
                st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)

    # ── Quick-start CTA ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">🚀 Quick Start</div>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        st.info("📸 **Analyse a Meal**\nUpload or take a photo of your meal to get instant nutritional breakdown.")
    with q2:
        st.info("👤 **Set Up Profile**\nEnter your height, weight, age and goals for personalised RDA targets.")
    with q3:
        st.info("📊 **Track Progress**\nView your nutritional history and weekly trends in the Daily Tracker.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE MEAL
# ══════════════════════════════════════════════════════════════════════════════
def page_analyse_meal():
    st.markdown('<div class="section-title" style="font-size:1.5rem;border:none;">📸 Analyse Your Meal</div>',
                unsafe_allow_html=True)
    st.markdown("Upload a meal image — our AI will detect food items, estimate nutrition, and give personalised advice.")

    # ── Input ─────────────────────────────────────────────────────────────
    tab_upload, tab_camera, tab_manual = st.tabs(["📁 Upload Image", "📷 Camera", "✏️ Manual Entry"])

    image = None

    with tab_upload:
        uploaded = st.file_uploader("Drop a meal photo here",
                                     type=["jpg", "jpeg", "png", "webp"],
                                     label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded)

    with tab_camera:
        cam = st.camera_input("Take a photo of your meal")
        if cam:
            image = Image.open(cam)

    with tab_manual:
        st.markdown("**Search and add foods manually:**")
        mc1, mc2 = st.columns([3, 1])
        with mc1:
            manual_food = st.selectbox(
                "Select food",
                options=[f.replace("_", " ").title() for f in sorted(FOOD_NUTRITION.keys())],
                key="manual_food_select",
            )
        with mc2:
            manual_portion = st.number_input("Portion (g)", min_value=10, max_value=1000, value=150, step=10)

        if st.button("➕ Add to Log", type="primary"):
            food_key = manual_food.lower().replace(" ", "_")
            nutrition = get_nutrition(food_key, manual_portion)
            if nutrition:
                log_meal(st.session_state["username"], food_key, manual_portion, 1.0)
                st.success(f"✅ {manual_food} ({manual_portion}g) logged!")
                st.markdown(f"""
                <div class="food-card">
                  <div class="food-title">{manual_food}</div>
                  <span class="confidence-badge">Manual Entry</span>
                  <div style="margin-top:10px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">
                    <div><b>{nutrition['calories']}</b><br><small>kcal</small></div>
                    <div><b>{nutrition['protein']}g</b><br><small>Protein</small></div>
                    <div><b>{nutrition['carbohydrates']}g</b><br><small>Carbs</small></div>
                    <div><b>{nutrition['fat']}g</b><br><small>Fat</small></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── AI Analysis ───────────────────────────────────────────────────────
    if image is not None:
        st.markdown("---")
        col_img, col_result = st.columns([1, 2])

        with col_img:
            st.image(image, caption="Your meal", use_container_width=True)
            portion_override = st.slider("Adjust portion size (g)", 50, 800, 200, 25)

        with col_result:
            with st.spinner("🤖 Analysing your meal …"):
                predictor, detector = load_predictor(tuple(st.session_state["class_names"]))
                result = detector.detect_and_classify(image, top_k=5)
                total  = detector.aggregate_nutrition(result)

                # Scale nutrition to portion override
                det    = result["detections"]
                mode   = result["mode"]

            # ── Detection mode badge ───────────────────────────────────
            st.markdown(
                f'<span style="background:{"#dbeafe" if mode=="single" else "#dcfce7"}; '
                f'color:{"#1d4ed8" if mode=="single" else "#166534"};'
                f'padding:4px 12px;border-radius:99px;font-size:0.78rem;font-weight:600;">'
                f'{"🔎 Single Food Mode" if mode=="single" else f"🔍 Multi-Food Detected ({len(det)} items)"}'
                f'</span>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Per-detection cards ────────────────────────────────────
            for i, d in enumerate(det):
                top  = d["top_prediction"]
                conf = top.get("confidence", 0)
                name = d.get("display_name", "Unknown")
                food_key = d.get("food_name", "unknown")
                port_g   = portion_override if mode == "single" else d.get("portion_g", 150)
                n = get_nutrition(food_key, port_g) if food_key in FOOD_NUTRITION else d.get("nutrition", {})

                st.markdown(f"""
                <div class="food-card">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div class="food-title">{'🍽️' if i==0 else '➕'} {name}</div>
                    <span class="confidence-badge">{conf:.1f}% confidence</span>
                  </div>
                  <div style="color:#64748b;font-size:0.82rem;margin:4px 0;">{port_g}g serving</div>
                  <div style="margin-top:10px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center;">
                    <div style="background:#fef3c7;border-radius:8px;padding:8px;">
                      <div style="font-size:1.2rem;font-weight:700;color:#b45309;">{n.get('calories',0):.0f}</div>
                      <div style="font-size:0.7rem;color:#92400e;">kcal</div>
                    </div>
                    <div style="background:#dbeafe;border-radius:8px;padding:8px;">
                      <div style="font-size:1.2rem;font-weight:700;color:#1d4ed8;">{n.get('protein',0):.1f}g</div>
                      <div style="font-size:0.7rem;color:#1e3a8a;">Protein</div>
                    </div>
                    <div style="background:#fef9c3;border-radius:8px;padding:8px;">
                      <div style="font-size:1.2rem;font-weight:700;color:#a16207;">{n.get('carbohydrates',0):.1f}g</div>
                      <div style="font-size:0.7rem;color:#854d0e;">Carbs</div>
                    </div>
                    <div style="background:#fce7f3;border-radius:8px;padding:8px;">
                      <div style="font-size:1.2rem;font-weight:700;color:#be185d;">{n.get('fat',0):.1f}g</div>
                      <div style="font-size:0.7rem;color:#9d174d;">Fat</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Top-K alternatives
                with st.expander(f"See other possibilities for item {i+1}"):
                    preds = d.get("predictions", [])
                    for p in preds[1:]:
                        st.write(f"• **{p['display_name']}** — {p['confidence']:.1f}%")

            # ── Log button ─────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Log This Meal", type="primary", use_container_width=True):
                for d in det:
                    food_key = d.get("food_name", "unknown")
                    if food_key in FOOD_NUTRITION:
                        port_g = portion_override if mode == "single" else d.get("portion_g", 150)
                        conf   = d["top_prediction"].get("confidence", 0) / 100
                        log_meal(st.session_state["username"], food_key, port_g, conf)
                st.success("✅ Meal logged to your daily tracker!")
                st.session_state["last_result"] = result

        # ── Recommendations ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">💡 Personalised Recommendations</div>', unsafe_allow_html=True)
        user = get_user(st.session_state["username"])

        if not user:
            st.info("👤 Set up your profile to get personalised BMI-based recommendations.")
        else:
            today   = datetime.date.today().isoformat()
            daily   = get_daily_intake(st.session_state["username"], today)
            recs    = quick_recommend(
                {**user, "goal": user.get("goal","maintain")},
                daily,
                meal_names=[d.get("food_name","") for d in det],
            )

            r1, r2 = st.columns(2)
            with r1:
                if recs.get("alerts"):
                    for a in recs["alerts"]:
                        st.markdown(f'<div class="alert-card">{a}</div>', unsafe_allow_html=True)
                for tip in recs.get("tips", [])[:4]:
                    st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)
            with r2:
                suggs = recs.get("food_suggestions", [])
                if suggs:
                    st.markdown("**🍽️ Suggested for Next Meal:**")
                    for s in suggs:
                        st.markdown(f"• {s}")
                st.markdown(f"""
                <div style="background:#f8fafc;border-radius:12px;padding:14px;margin-top:10px;
                            font-size:0.88rem;color:#374151;line-height:1.6;">
                  {recs.get('summary','')}
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DAILY TRACKER
# ══════════════════════════════════════════════════════════════════════════════
def page_daily_tracker():
    st.markdown('<div class="section-title" style="font-size:1.5rem;border:none;">📊 Daily Nutritional Tracker</div>',
                unsafe_allow_html=True)

    username = st.session_state["username"]
    today    = datetime.date.today().isoformat()

    # ── Date selector ─────────────────────────────────────────────────────
    dc1, dc2 = st.columns([2, 3])
    with dc1:
        selected_date = st.date_input("Select date", value=datetime.date.today(),
                                       max_value=datetime.date.today())
    date_str = selected_date.isoformat()

    intake = get_daily_intake(username, date_str)
    user   = get_user(username)

    rda_now = RDA.copy()
    if user:
        try:
            rda_now = calculate_rda(float(user.get("weight_kg",70)), float(user.get("height_m",1.7)),
                                     int(user.get("age",30)), user.get("gender","male"),
                                     user.get("activity","moderately active"))
        except Exception:
            pass

    # ── Summary cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    def kpi(col, label, val, goal, unit):
        pct = int(val/goal*100) if goal > 0 else 0
        col.metric(label=label, value=f"{val:.0f}{unit}", delta=f"{pct}% of {goal}{unit} goal")

    kpi(c1, "🔥 Calories",  intake["calories"],      rda_now["calories"],      " kcal")
    kpi(c2, "💪 Protein",   intake["protein"],       rda_now["protein"],       "g")
    kpi(c3, "🍞 Carbs",     intake["carbohydrates"], rda_now["carbohydrates"], "g")
    kpi(c4, "🥑 Fat",       intake["fat"],           rda_now["fat"],           "g")
    kpi(c5, "🥦 Fiber",     intake["fiber"],         rda_now["fiber"],         "g")

    st.markdown("---")

    # ── Nutrient bars + donut ─────────────────────────────────────────────
    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="section-title">Nutrient Progress vs Goal</div>', unsafe_allow_html=True)
        html = ""
        for label, key, unit, emoji in [
            ("Calories","calories","kcal","🔥"), ("Protein","protein","g","💪"),
            ("Carbs","carbohydrates","g","🍞"),  ("Fat","fat","g","🥑"),
            ("Fiber","fiber","g","🥦"),           ("Sugar","sugar","g","🍬"),
            ("Sodium","sodium","mg","🧂"),
        ]:
            html += nutrient_bar_html(label, intake.get(key,0), rda_now.get(key, RDA[key]), unit, emoji)
        st.markdown(html, unsafe_allow_html=True)
    with right:
        st.plotly_chart(macro_donut(intake, "Macro Split"), use_container_width=True)

    st.markdown("---")

    # ── 7-day history ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 7-Day History</div>', unsafe_allow_html=True)
    hist_df = get_meal_history(username, days=7)
    if hist_df.empty:
        st.info("No meal history yet. Analyse a meal and log it to see trends here.")
    else:
        st.plotly_chart(history_line(hist_df), use_container_width=True)

        st.markdown('<div class="section-title">🍽️ Meal Log</div>', unsafe_allow_html=True)
        display_cols = ["date", "time", "food_name", "portion_g", "calories",
                        "protein", "carbohydrates", "fat", "confidence"]
        show_df = hist_df[display_cols].copy()
        show_df.columns = ["Date", "Time", "Food", "Portion(g)", "Calories",
                            "Protein(g)", "Carbs(g)", "Fat(g)", "Confidence"]
        show_df["Food"] = show_df["Food"].str.replace("_", " ").str.title()
        show_df["Confidence"] = (show_df["Confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # Export
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export as CSV", csv, "meal_history.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MY PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def page_my_profile():
    st.markdown('<div class="section-title" style="font-size:1.5rem;border:none;">👤 My Health Profile</div>',
                unsafe_allow_html=True)

    username = st.session_state["username"]
    existing = get_user(username) or {}

    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            weight = st.number_input("⚖️ Weight (kg)",   min_value=20.0,  max_value=300.0,
                                      value=float(existing.get("weight_kg", 70.0)), step=0.5)
            height = st.number_input("📏 Height (m)",    min_value=0.5,   max_value=2.5,
                                      value=float(existing.get("height_m",  1.70)), step=0.01,
                                      format="%.2f")
            age    = st.number_input("🎂 Age",           min_value=10,    max_value=100,
                                      value=int(existing.get("age", 30)))
        with c2:
            gender   = st.selectbox("🚻 Gender",
                                     ["Male", "Female", "Other"],
                                     index=["Male","Female","Other"].index(
                                         existing.get("gender","Male").capitalize())
                                     if existing.get("gender","").capitalize() in ["Male","Female","Other"] else 0)
            activity = st.selectbox("🏃 Activity Level",
                                     ["Sedentary", "Lightly Active", "Moderately Active",
                                      "Very Active", "Extra Active"],
                                     index=2)
            goal     = st.selectbox("🎯 Health Goal",
                                     ["Maintain", "Lose Weight", "Gain Muscle"],
                                     index=["maintain","lose weight","gain muscle"].index(
                                         existing.get("goal","maintain")) if existing.get("goal") else 0)

        submitted = st.form_submit_button("💾 Save Profile", type="primary", use_container_width=True)

    if submitted:
        save_user(username, weight, height, age, gender.lower(),
                  activity.lower(), goal.lower())
        st.success("✅ Profile saved!")
        st.rerun()

    # ── Show BMI + RDA after save ──────────────────────────────────────────
    if existing and existing.get("weight_kg"):
        st.markdown("---")
        bmi      = calculate_bmi(float(existing["weight_kg"]), float(existing["height_m"]))
        bmi_info = classify_bmi(bmi)
        rda_p    = calculate_rda(float(existing["weight_kg"]), float(existing["height_m"]),
                                  int(existing.get("age",30)), existing.get("gender","male"),
                                  existing.get("activity","moderately active"))

        b1, b2 = st.columns(2)
        with b1:
            st.markdown('<div class="section-title">⚖️ BMI Analysis</div>', unsafe_allow_html=True)
            st.plotly_chart(bmi_gauge(bmi, bmi_info["category"], bmi_info["color"]),
                            use_container_width=True)
            st.markdown(f"""
            <div style="background:{bmi_info['color']}22;border-radius:12px;padding:16px;text-align:center;">
              <div style="font-size:2.5rem;font-weight:800;color:{bmi_info['color']};">{bmi}</div>
              <div style="font-size:1rem;font-weight:600;color:{bmi_info['color']};margin:4px 0;">
                {bmi_info['category']}
              </div>
              <div style="font-size:0.8rem;color:#64748b;">Health Risk: {bmi_info['risk']}</div>
            </div>""", unsafe_allow_html=True)
            st.caption(
                "ℹ️ BMI is a screening signal, not a diagnosis. It doesn't "
                "distinguish fat from muscle, and WHO's standard thresholds are "
                "known to under- or over-classify risk for some ethnic groups "
                "(e.g. lower cutoffs are more appropriate for some South Asian, "
                "Black African, and Black Caribbean populations)."
            )

        with b2:
            st.markdown('<div class="section-title">🎯 Your Personalised RDA</div>', unsafe_allow_html=True)
            rda_df = pd.DataFrame([
                {"Nutrient": k.title().replace("_"," "), "Your Target": v,
                 "Unit": "kcal" if k=="calories" else ("mg" if k=="sodium" else "g")}
                for k, v in rda_p.items()
            ])
            st.dataframe(rda_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def page_model_insights():
    st.markdown('<div class="section-title" style="font-size:1.5rem;border:none;">📈 Model Insights & Performance</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Training Metrics", "🔬 Evaluation", "🔧 HP Search", "🏗️ Architecture"])

    # ── Tab 1: Training history ────────────────────────────────────────────
    with tab1:
        history_files = list(LOGS_DIR.glob("*_history.json"))
        if history_files:
            selected_run = st.selectbox("Select training run",
                                         [f.stem for f in history_files])
            hist_path = LOGS_DIR / f"{selected_run}.json"
            with open(hist_path) as f:
                history = json.load(f)

            df = pd.DataFrame(history)
            if not df.empty:
                # Loss curves
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"],
                    name="Train Loss", line=dict(color="#3b82f6", width=2)))
                fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"],
                    name="Val Loss", line=dict(color="#ef4444", width=2, dash="dot")))
                fig_loss.update_layout(title="Loss Curves", height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_loss, use_container_width=True)

                # Accuracy curves
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=df["epoch"], y=df["train_acc"],
                    name="Train Acc", line=dict(color="#3b82f6", width=2)))
                fig_acc.add_trace(go.Scatter(x=df["epoch"], y=df["val_acc"],
                    name="Val Acc", line=dict(color="#ef4444", width=2, dash="dot")))
                fig_acc.add_trace(go.Scatter(x=df["epoch"], y=df["val_top5"],
                    name="Val Top-5 Acc", line=dict(color="#22c55e", width=2, dash="dash")))
                fig_acc.update_layout(title="Accuracy Curves", height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_acc, use_container_width=True)

                # F1 and LR
                c1, c2 = st.columns(2)
                with c1:
                    fig_f1 = px.line(df, x="epoch", y="val_f1", title="Val Macro F1")
                    fig_f1.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_f1, use_container_width=True)
                with c2:
                    fig_lr = px.line(df, x="epoch", y="lr", title="Learning Rate Schedule",
                                      log_y=True)
                    fig_lr.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_lr, use_container_width=True)

                # Raw table
                with st.expander("View raw metrics table"):
                    st.dataframe(df.round(4), use_container_width=True)
        else:
            st.info("No training history found. Train the model first with `python train.py`.")
            _show_placeholder_charts()

    # ── Tab 2: Test evaluation ─────────────────────────────────────────────
    with tab2:
        eval_path = LOGS_DIR / "test_evaluation.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)

            metrics = eval_data.get("metrics", {})
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Top-1 Accuracy",  f"{metrics.get('accuracy', 0):.2f}%")
            mc2.metric("Top-5 Accuracy",  f"{metrics.get('top5_accuracy', 0):.2f}%")
            mc3.metric("Macro Precision", f"{metrics.get('precision_macro', 0):.4f}")
            mc4.metric("Macro F1",        f"{metrics.get('f1_macro', 0):.4f}")

            # Per-class breakdown
            report = eval_data.get("classification_report", {})
            if report:
                per_class = []
                for cls, vals in report.items():
                    if isinstance(vals, dict):
                        per_class.append({
                            "Class": cls.replace("_"," ").title(),
                            "Precision": round(vals.get("precision",0), 3),
                            "Recall":    round(vals.get("recall",   0), 3),
                            "F1":        round(vals.get("f1-score", 0), 3),
                            "Support":   int(vals.get("support",    0)),
                        })
                per_class_df = pd.DataFrame(per_class)
                if not per_class_df.empty:
                    top10 = per_class_df.nlargest(10, "F1")
                    fig_bar = px.bar(top10, x="F1", y="Class", orientation="h",
                                     title="Top-10 Classes by F1 Score",
                                     color="F1", color_continuous_scale="Blues")
                    fig_bar.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    with st.expander("Full per-class report"):
                        st.dataframe(per_class_df.sort_values("F1", ascending=False),
                                      use_container_width=True)
        else:
            st.info("No test evaluation found. Run `python train.py --mode evaluate`.")
            _show_placeholder_eval()

    # ── Tab 3: HP search results ───────────────────────────────────────────
    with tab3:
        hp_files = list(LOGS_DIR.glob("hp_search_*.json"))
        if hp_files:
            for hp_file in hp_files:
                with open(hp_file) as f:
                    hp = json.load(f)
                st.subheader(f"🔧 {hp.get('arch', '?')} HP Search Results")
                m1, m2, m3 = st.columns(3)
                m1.metric("Best Val Accuracy", f"{hp.get('best_val_acc', 0):.2f}%")
                m2.metric("Completed Trials",  hp.get("completed", 0))
                m3.metric("Pruned Trials",     hp.get("pruned", 0))
                st.json(hp.get("best_params", {}))
                st.markdown("---")
        else:
            st.info("No HP search results yet. Run `python train.py --mode full`.")

        # Model selection comparison
        sel_path = LOGS_DIR / "model_selection.json"
        if sel_path.exists():
            with open(sel_path) as f:
                sel = json.load(f)
            st.subheader("🏆 Model Selection Results")
            rows = []
            for arch, r in sel["results"].items():
                rows.append({"Architecture": arch,
                              "Val Accuracy": round(r.get("val_acc",0), 2),
                              "Status": r.get("status","?")})
            sel_df = pd.DataFrame(rows).sort_values("Val Accuracy", ascending=False)
            st.dataframe(sel_df, use_container_width=True, hide_index=True)
            winner = sel.get("winner", "?")
            st.success(f"🏆 Winner: **{winner}**")

    # ── Tab 4: Architecture ────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        ### 🏗️ System Architecture

        **Food Classification Model — EfficientNet-B0**
        - Pre-trained on ImageNet-1K (1.2M images)
        - Custom classification head: `Dropout(0.4) → Linear(512) → BN → ReLU → Dropout(0.2) → Linear(20)`
        - Two-phase training: 5 epochs frozen backbone → fine-tuning all layers
        - Input: 224 × 224 RGB images, ImageNet normalisation
        - Output: 20-class softmax probabilities (twenty selected Food-101 classes; see Table 3.1)

        **Food Detection — YOLOv8n**
        - Detects multiple food items in a single image
        - Each region-of-interest (ROI) is cropped and passed to EfficientNet
        - Falls back to full-image classification if YOLO unavailable

        **Portion Estimation**
        - A fixed default gram weight per class (`DEFAULT_PORTION_G`), scaled
          by the ratio of detected bounding-box area to total image area
        - Not measured against ground truth — a stated heuristic, not a
          validated estimate (see Chapter Six, Limitations)

        **Nutritional Database**
        - 101 food categories from USDA FoodData Central
        - SQLite backend for fast local lookup
        - Returns: calories, protein, carbs, fat, fiber, sugar, sodium per 100g

        **BMI + Recommendation Engine**
        - WHO BMI classification (5 tiers)
        - Personalised RDA via Mifflin-St Jeor equation + activity factor
        - Rule-based dietary filtering → structured recommendation output

        **Deployment**
        - Streamlit web app (this interface)
        - Model checkpoint auto-loaded from `models/best_model.pth`
        - SQLite user database for meal logging and history tracking
        """)

        # Architecture diagram
        st.markdown("---")
        st.markdown("**Pipeline Flow:**")
        st.code("""
Meal Image
    │
    ▼
[Pre-processing]  →  Resize 224×224, Normalise (ImageNet)
    │
    ▼
[YOLOv8n Detector]  →  Detect food ROIs
    │
    ▼ (per ROI)
[EfficientNet-B0]   →  20-class softmax prediction
    │
    ├─ confidence ≥ 0.60  →  Normal record
    │                          → USDA nutritional lookup
    │                          → Log to SQLite
    │
    └─ confidence < 0.60  →  Flagged low_confidence
                               → Nutrition lookup withheld
                               → Alternatives shown to user
    │
    ▼
[BMI Module]  →  weight & height → BMI → WHO classification
    │
    ▼
[Recommendation Engine]
    ├─ Rule-based filters (WHO dietary guidelines)
    └─ Structured output → Tips, Alerts, Food Suggestions
    │
    ▼
[Streamlit Dashboard]  →  Real-time display
        """, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown(f"""
    <div class="app-header">
      <h1>{APP_ICON} {APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 📖 About This System
        NutriVision AI is a research prototype developed as part of an academic project on
        **Real-Time AI-Powered Automated Nutritional Analysis of Meals Using Deep Learning,
        Computer Vision, BMI Analysis, and Personalized Recommendation System**.

        The system combines:
        - **EfficientNet-B0** for food image classification (twenty selected Food-101 categories)
        - **YOLOv8** for multi-food object detection in meal images
        - **USDA FoodData Central** data for nutritional estimation
        - **WHO BMI framework** for health status assessment
        - A **hybrid recommendation engine** (rule-based + structured NLP)

        ### 🛠️ Technologies
        | Component | Technology |
        |-----------|------------|
        | Deep Learning | PyTorch + EfficientNet-B0 |
        | Object Detection | YOLOv8 (Ultralytics) |
        | HP Tuning | Optuna (TPE sampler) |
        | Nutritional DB | SQLite + USDA FoodData |
        | Web Interface | Streamlit |
        | Data Processing | scikit-learn, pandas, numpy |
        | Visualisation | Plotly |
        """)
    with c2:
        st.markdown("""
        ### 📊 Dataset
        - **Food-101** (Bossard et al., 2014) — 101,000 images across 101 categories
        - Training restricted to twenty commonly eaten classes (Table 3.1)
        - 85% / 15% train/validation split plus the official Food-101 test
          split, giving 400 training batches, 45 validation batches, and
          149 test batches at batch size 32 (~4,750 test images)
        - 224 × 224 pixel input resolution

        ### 📐 Model Performance Targets
        | Metric | Target |
        |--------|--------|
        | Top-1 Accuracy | ≥ 85% |
        | Top-5 Accuracy | ≥ 97% |
        | Macro F1-Score | ≥ 0.82 |
        | Inference Latency | ≤ 3 sec |
        | SUS Usability Score | ≥ 70 |

        ### 🔒 Privacy
        All user data is stored locally in an SQLite database.
        No data is transmitted to external servers.
        Users can clear their data at any time.

        ### 📚 References
        - Bossard et al. (2014). Food-101 – Mining Discriminative Components with Random Forests.
        - Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs.
        - Jocher, Chaurasia & Qiu (2023). YOLOv8 (Ultralytics).
        - WHO (2000). Obesity: Preventing and Managing the Global Epidemic.
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#94a3b8;font-size:0.85rem;padding:20px 0;">
      NutriVision AI v1.0 · Built with PyTorch, Streamlit & other liberies ·
      Academic Masters Research Project · Joshua Nnorom
    </div>
    """, unsafe_allow_html=True)


# ─── Placeholder charts (when no training data yet) ───────────────────────────
def _show_placeholder_charts():
    """Show synthetic charts to demonstrate UI before training."""
    epochs = list(range(1, 31))
    import random
    random.seed(42)
    train_loss = [2.5 * (0.88 ** i) + random.uniform(-0.05, 0.05) for i in epochs]
    val_loss   = [2.7 * (0.89 ** i) + random.uniform(-0.08, 0.08) for i in epochs]
    train_acc  = [min(95, 30 + i * 2.2 + random.uniform(-1, 1)) for i in epochs]
    val_acc    = [min(90, 25 + i * 2.1 + random.uniform(-1.5, 1.5)) for i in epochs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, name="Train Acc",
                              line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, name="Val Acc",
                              line=dict(color="#ef4444", width=2, dash="dot")))
    fig.update_layout(title="Sample Training Curves (Demo — train model to see real results)",
                       height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

def _show_placeholder_eval():
    import random
    random.seed(1)
    classes = [f.replace("_"," ").title() for f in SELECTED_CLASSES[:15]]
    f1s     = [round(random.uniform(0.70, 0.95), 3) for _ in classes]
    fig = px.bar(x=f1s, y=classes, orientation="h",
                  title="Sample F1 Scores by Class (Demo)",
                  color=f1s, color_continuous_scale="Blues")
    fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ─── Router ───────────────────────────────────────────────────────────────────
def main():
    init_session()
    ensure_db()

    page = render_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "📸 Analyse Meal":
        page_analyse_meal()
    elif page == "📊 Daily Tracker":
        page_daily_tracker()
    elif page == "👤 My Profile":
        page_my_profile()
    elif page == "📈 Model Insights":
        page_model_insights()
    elif page == "ℹ️  About":
        page_about()


if __name__ == "__main__":
    main()
