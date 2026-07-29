"""Train the taste-affinity model and save it to music_recommender/models/.

This is the project's "fine-tuned / specialized model" component: a small
RandomForestRegressor trained on a curated (genre, mood) -> audio-feature
dataset (music_recommender/data/taste_training_data.csv), predicting a
target energy/valence/acousticness profile for a stated taste. That
prediction feeds into the RAG pipeline as an additional ranking signal
(see music_recommender/taste_model.py and retriever.py).

Run after `pip install -r requirements.txt`:
    python scripts/train_taste_model.py

Running this script is optional — if no trained artifact is present,
TasteAffinityModel trains the same model in memory on first use.
"""
import csv
import pathlib

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "music_recommender" / "data" / "taste_training_data.csv"
MODEL_PATH = PROJECT_ROOT / "music_recommender" / "models" / "taste_affinity_model.joblib"

TARGET_COLUMNS = ["energy", "valence", "acousticness"]


def load_training_data():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    features = [{"genre": row["genre"], "mood": row["mood"]} for row in rows]
    targets = [[float(row[col]) for col in TARGET_COLUMNS] for row in rows]
    return features, targets


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorize", DictVectorizer(sparse=False)),
            ("regressor", RandomForestRegressor(n_estimators=50, random_state=42)),
        ]
    )


def main() -> None:
    features, targets = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Trained on {len(X_train)} examples, held out {len(X_test)} for evaluation.")
    print(f"Mean absolute error on held-out audio-feature targets: {mae:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
