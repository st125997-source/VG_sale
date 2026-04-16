from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Video Game Sales Oracle",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Playfair+Display:wght@700&display=swap');

    :root {
        --ink: #0c1220;
        --mist: #eef3ff;
        --sun: #ffb347;
        --teal: #00a6a6;
        --rose: #ff6f61;
        --card: rgba(255, 255, 255, 0.78);
    }

    .stApp {
        font-family: 'Space Grotesk', sans-serif;
        background:
            radial-gradient(circle at 12% 18%, rgba(255, 179, 71, 0.25), transparent 26%),
            radial-gradient(circle at 82% 22%, rgba(0, 166, 166, 0.22), transparent 30%),
            linear-gradient(135deg, #f6f8ff 0%, #e9f8f7 45%, #fff3e7 100%);
        color: var(--ink);
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(255,111,97,0.18), rgba(0,166,166,0.20));
        border: 1px solid rgba(12, 18, 32, 0.08);
        box-shadow: 0 12px 32px rgba(12, 18, 32, 0.10);
        animation: riseIn 0.7s ease-out;
    }

    .hero h1 {
        margin: 0;
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.5px;
        color: #0a1b2f;
    }

    .hero p {
        margin: 0.6rem 0 0 0;
        opacity: 0.86;
    }

    .metric-card {
        background: var(--card);
        border: 1px solid rgba(12,18,32,0.08);
        border-radius: 16px;
        padding: 0.8rem 1rem;
        box-shadow: 0 8px 24px rgba(12, 18, 32, 0.08);
        animation: riseIn 0.7s ease-out;
    }

    @keyframes riseIn {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stButton > button {
        background: linear-gradient(90deg, #00a6a6, #ff6f61);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-weight: 700;
    }

    .stSelectbox label, .stNumberInput label {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "trained_models" / "best_model.pkl"
CONFIG_PATH = BASE_DIR / "feature_config.json"
DATA_PATH = BASE_DIR / "video_games_processed.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_assets():
    missing = []
    for path in [MODEL_PATH, CONFIG_PATH, DATA_PATH]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        st.error("Missing required project files:")
        for m in missing:
            st.write(f"- {m}")
        st.stop()


ensure_assets()
model = load_model()
df = load_data()
cfg = load_config()

all_features = cfg["all_features"]
cat_features = cfg["categorical_features"]
num_features = cfg["numerical_features"]
bin_features = cfg["binary_features"]


st.markdown(
    """
    <div class="hero">
      <h1>Video Game Sales Oracle</h1>
      <p>Interactive prediction studio powered by your trained Random Forest model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Rows", f"{len(df):,}")
    st.markdown('</div>', unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Unique Consoles", int(df["console"].nunique()))
    st.markdown('</div>', unsafe_allow_html=True)
with col_c:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Median Sales (M)", f"{df['total_sales(mil)'].median():.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.05, 1.4])

with left:
    st.subheader("Prediction Controls")
    sample = df.sample(1, random_state=17).iloc[0]

    user_input = {}

    for feature in cat_features:
        options = sorted([x for x in df[feature].dropna().unique().tolist()])
        default_val = sample.get(feature, options[0] if options else "")
        default_idx = options.index(default_val) if default_val in options else 0
        user_input[feature] = st.selectbox(feature, options, index=default_idx)

    for feature in num_features:
        val = float(sample.get(feature, df[feature].median()))
        user_input[feature] = st.number_input(feature, value=val, step=0.1)

    for feature in bin_features:
        val = int(sample.get(feature, 0))
        user_input[feature] = st.selectbox(feature, [0, 1], index=val if val in [0, 1] else 0)

    predict = st.button("Predict Sales")

    if predict:
        row = pd.DataFrame([user_input])[all_features]
        pred_log = model.predict(row)[0]
        pred_sales = max(float(np.expm1(pred_log)), 0.0)
        st.success(f"Predicted total_sales(mil): {pred_sales:.3f}")

with right:
    st.subheader("Sales Landscape")
    chart_df = (
        df.groupby("genre", as_index=False)["total_sales(mil)"]
        .sum()
        .sort_values("total_sales(mil)", ascending=False)
        .head(12)
    )

    fig = px.bar(
        chart_df,
        x="total_sales(mil)",
        y="genre",
        orientation="h",
        color="total_sales(mil)",
        color_continuous_scale="Tealrose",
        title="Top Genres by Total Sales",
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=54, b=16), yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Model Feature Importance")
fi_path = BASE_DIR / "models" / "trained_models" / "feature_importance.csv"
if fi_path.exists():
    fi = pd.read_csv(fi_path).head(15)
    fig2 = px.bar(
        fi.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Sunset",
        title="Top 15 Feature Importances",
    )
    fig2.update_layout(height=520, margin=dict(l=10, r=10, t=54, b=16))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("feature_importance.csv not found. Run modeling to generate it.")

st.caption("Built for the Video Game Sales prediction project.")
