"""Ranking strategies — the Strategy design pattern applied to how retrieved
evidence becomes a score.

Retrieval (retriever.py) always gathers the same four evidence signals for
a song: categorical tag matches, semantic similarity, predicted audio-fit,
and genre context. A RankingStrategy decides how much each signal counts
toward the final score — i.e. *what the recommender optimizes for* — without
touching retrieval, diversity re-ranking, or explanation generation at all.
Swapping the strategy is a one-line change in MusicRecommender.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RankingStrategy(ABC):
    """A strategy turns one song's retrieved evidence into a single score.
    Higher is better. The score is a raw evidence score before the
    recommender's shared +2 baseline and diversity penalties are applied."""

    name: str = "unnamed"

    @abstractmethod
    def score(self, evidence: dict) -> float:
        raise NotImplementedError


# A theme match used to be worth exactly 1 point, the same flat weight as any
# other single-tag match. In practice that was rarely enough to reorder
# anything — themes only "worked" when a broader ranking-mode swap moved the
# score enough to notice, which read as themes not mattering on their own.
# Weighting a theme match above the 1-point default gives it a real chance to
# shift the ranking by itself, without letting it override an actual genre or
# mood match (still worth 10x in the strategies that prioritize those).
THEME_WEIGHT = 2.0


def _categorical_count(evidence: dict) -> float:
    return (
        len(evidence["matched_genres"])
        + len(evidence["matched_moods"])
        + len(evidence["matched_themes"]) * THEME_WEIGHT
        + len(evidence["matched_artists"])
    )


def _generated_attribute_bonus(evidence: dict) -> float:
    """A small, consistently-applied bonus from the AI-generated song
    attributes (song_attributes.py): a finer-grained mood match than the
    single `mood` field can catch, plus a mainstream-popularity nudge
    mirroring how real recommenders factor in popularity alongside
    relevance (see README's "How Music Recommendation Systems Work").
    Deliberately small relative to the tag/semantic/audio signals above —
    this enriches ranking, it doesn't dominate it."""
    detailed_mood_bonus = len(evidence.get("matched_detailed_moods", [])) * 0.5
    popularity_bonus = evidence.get("popularity_score", 0.0) * 0.5
    return detailed_mood_bonus + popularity_bonus


class BalancedStrategy(RankingStrategy):
    """The default: categorical tags, semantic meaning, and predicted audio
    fit all contribute, with semantic similarity weighted slightly highest."""

    name = "balanced"

    def score(self, evidence: dict) -> float:
        return (
            _categorical_count(evidence)
            + evidence.get("semantic_score", 0.0) * 3.0
            + evidence.get("audio_fit_score", 0.0) * 1.5
            + _generated_attribute_bonus(evidence)
        )


class GenreFirstStrategy(RankingStrategy):
    """Prioritizes an exact genre match above every other signal — for a
    listener who cares most about staying within a specific genre."""

    name = "genre-first"

    def score(self, evidence: dict) -> float:
        genre_score = len(evidence["matched_genres"]) * 10.0
        other_tags = (
            len(evidence["matched_moods"]) + len(evidence["matched_themes"]) * THEME_WEIGHT + len(evidence["matched_artists"])
        )
        return (
            genre_score + other_tags + evidence.get("semantic_score", 0.0)
            + evidence.get("audio_fit_score", 0.0) * 0.5 + _generated_attribute_bonus(evidence)
        )


class MoodFirstStrategy(RankingStrategy):
    """Prioritizes mood and semantic (felt meaning) matches over exact genre
    labels — for a listener who cares more about how a song feels than what
    it's officially tagged as."""

    name = "mood-first"

    def score(self, evidence: dict) -> float:
        mood_score = len(evidence["matched_moods"]) * 10.0
        semantic = evidence.get("semantic_score", 0.0) * 5.0
        other_tags = len(evidence["matched_genres"]) + len(evidence["matched_themes"]) * THEME_WEIGHT + len(evidence["matched_artists"])
        return (
            mood_score + semantic + other_tags + evidence.get("audio_fit_score", 0.0) * 2.0
            + _generated_attribute_bonus(evidence)
        )


class EnergySimilarityStrategy(RankingStrategy):
    """Prioritizes the trained taste-affinity model's predicted audio-feature
    fit (energy/valence/acousticness closeness) above tags or semantics — for
    a listener who cares most about a song's overall sound."""

    name = "energy-similarity"

    def score(self, evidence: dict) -> float:
        audio = evidence.get("audio_fit_score", 0.0) * 10.0
        return (
            audio + _categorical_count(evidence) + evidence.get("semantic_score", 0.0)
            + _generated_attribute_bonus(evidence)
        )


STRATEGIES = {
    "balanced": BalancedStrategy,
    "genre-first": GenreFirstStrategy,
    "mood-first": MoodFirstStrategy,
    "energy-similarity": EnergySimilarityStrategy,
}


def get_strategy(name: str) -> RankingStrategy:
    try:
        return STRATEGIES[name]()
    except KeyError:
        valid = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown ranking strategy '{name}'. Valid options: {valid}")
