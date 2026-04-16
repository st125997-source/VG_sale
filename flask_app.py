from pathlib import Path
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "trained_models" / "best_model.pkl"
CONFIG_PATH = BASE_DIR / "feature_config.json"
DATA_PATH = BASE_DIR / "video_games_processed.csv"

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@lru_cache(maxsize=1)
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing feature config: {CONFIG_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed data: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return model, df, config


def model_input_features(config: dict) -> list:
    required = [
        "console",
        "genre",
        "publisher",
        "release_year",
        "release_month",
        "critic_score_filled",
    ]
    configured = config.get("all_features", [])
    if configured != required:
        raise ValueError(f"Expected 6-feature config {required}, found {configured}")
    return required


@app.route("/", methods=["GET", "POST"])
def index():
    model, df, config = load_assets()
    required_features = model_input_features(config)

    consoles = sorted(df["console"].dropna().unique().tolist())
    genres = sorted(df["genre"].dropna().unique().tolist())
    publishers = sorted(df["publisher"].dropna().unique().tolist())

    selected_console = consoles[0] if consoles else "PS4"
    selected_genre = genres[0] if genres else "Action"
    selected_publisher = publishers[0] if publishers else "Nintendo"
    selected_year = 2015
    selected_month = 11
    selected_critic = 75.0

    result = None
    feature_payload = None

    if request.method == "POST":
        selected_console = request.form.get("console", selected_console)
        selected_genre = request.form.get("genre", selected_genre)
        selected_publisher = request.form.get("publisher", selected_publisher)
        selected_year = int(request.form.get("release_year", selected_year))
        selected_month = int(request.form.get("release_month", selected_month))
        selected_critic = float(request.form.get("critic_score", selected_critic))

        feature_payload = {
            "console": selected_console,
            "genre": selected_genre,
            "publisher": selected_publisher,
            "release_year": selected_year,
            "release_month": selected_month,
            "critic_score_filled": selected_critic,
        }

        row = pd.DataFrame([feature_payload])[required_features]
        prediction_log = float(model.predict(row)[0])
        prediction = max(float(np.expm1(prediction_log)), 0.0)

        result = {
            "prediction": prediction,
            "prediction_log": prediction_log,
            "feature_payload": feature_payload,
        }

    # Dashboard data
    top_genres = (
        df.groupby("genre", as_index=False)["total_sales(mil)"]
        .sum()
        .sort_values("total_sales(mil)", ascending=False)
        .head(8)
        .to_dict(orient="records")
    )

    top_consoles = (
        df.groupby("console", as_index=False)["total_sales(mil)"]
        .sum()
        .sort_values("total_sales(mil)", ascending=False)
        .head(8)
        .to_dict(orient="records")
    )

    feature_importance = []
    fi_path = BASE_DIR / "models" / "trained_models" / "feature_importance.csv"
    if fi_path.exists():
        feature_importance = pd.read_csv(fi_path).head(10).to_dict(orient="records")

    top_genre_labels = [item["genre"] for item in top_genres]
    top_genre_values = [float(item["total_sales(mil)"]) for item in top_genres]
    top_console_labels = [item["console"] for item in top_consoles]
    top_console_values = [float(item["total_sales(mil)"]) for item in top_consoles]
    feature_labels = [item["feature"] for item in feature_importance]
    feature_values = [float(item["importance"]) for item in feature_importance]

    return render_template(
        "index.html",
        consoles=consoles,
        genres=genres,
        publishers=publishers,
        selected_console=selected_console,
        selected_genre=selected_genre,
        selected_publisher=selected_publisher,
        selected_year=selected_year,
        selected_month=selected_month,
        selected_critic=selected_critic,
        result=result,
        top_genres=top_genres,
        top_consoles=top_consoles,
        feature_importance=feature_importance,
        top_genre_labels=top_genre_labels,
        top_genre_values=top_genre_values,
        top_console_labels=top_console_labels,
        top_console_values=top_console_values,
        feature_labels=feature_labels,
        feature_values=feature_values,
        features=feature_payload,
        model_name="Random Forest Regressor",
        total_rows=int(len(df)),
        total_rows_formatted=f"{len(df):,}",
        median_sales=float(df["total_sales(mil)"].median()),
        unique_consoles=int(df["console"].nunique()),
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
