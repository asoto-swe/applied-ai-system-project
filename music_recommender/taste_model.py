"""Predicts a target audio-feature profile (energy, valence, acousticness)
from a taste profile's stated genres and moods.

This is the project's "fine-tuned / specialized model" component: a small
scikit-learn RandomForestRegressor trained offline on a curated (genre,
mood) -> audio-feature dataset (see scripts/train_taste_model.py and
music_recommender/data/taste_training_data.csv). Its prediction becomes a
third scoring signal in the RAG pipeline, alongside categorical tag
matching (retriever.py) and Voyage AI semantic similarity (embeddings.py).
"""
from __future__ import annotations

import csv
import logging
import pathlib
from typing import Dict, List, Optional

from .data import TasteProfile

logger = logging.getLogger(__name__)

_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = _PACKAGE_DIR / "models" / "taste_affinity_model.joblib"
TRAINING_DATA_PATH = _PACKAGE_DIR / "data" / "taste_training_data.csv"
TARGET_COLUMNS = ["energy", "valence", "acousticness"]


class TasteAffinityModel:
    """Predicts an audio-feature target from a taste profile's genres/moods."""

    def __init__(self) -> None:
        self._pipeline = self._load_trained_artifact() or self._train_in_memory()

    def _load_trained_artifact(self):
        if not MODEL_PATH.exists():
            return None
        try:
            import joblib

            return joblib.load(MODEL_PATH)
        except Exception as exc:
            logger.warning("Could not load trained taste-affinity model, retraining in memory: %s", exc)
            return None

    def _train_in_memory(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.pipeline import Pipeline

        rows = self._load_training_rows()
        features = [{"genre": row["genre"], "mood": row["mood"]} for row in rows]
        targets = [[float(row[col]) for col in TARGET_COLUMNS] for row in rows]

        pipeline = Pipeline(
            steps=[
                ("vectorize", DictVectorizer(sparse=False)),
                ("regressor", RandomForestRegressor(n_estimators=50, random_state=42)),
            ]
        )
        pipeline.fit(features, targets)
        return pipeline

    @staticmethod
    def _load_training_rows() -> List[dict]:
        with open(TRAINING_DATA_PATH, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def predict(self, profile: TasteProfile) -> Optional[Dict[str, float]]:
        """Predict a target audio profile, averaging predictions across every
        (genre, mood) combination the listener stated. Returns None if the
        profile has no genre or mood signal to predict from."""
        if not profile.preferred_genres and not profile.preferred_moods:
            return None

        genres = profile.preferred_genres or [""]
        moods = profile.preferred_moods or [""]
        combos = [{"genre": genre, "mood": mood} for genre in genres for mood in moods]

        predictions = self._pipeline.predict(combos)
        averaged = predictions.mean(axis=0)
        return dict(zip(TARGET_COLUMNS, (float(value) for value in averaged)))
