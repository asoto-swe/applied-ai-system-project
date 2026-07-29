import logging
from typing import List

from .data import Recommendation, Song, TasteProfile
from .explainer import ClaudeExplainer
from .retriever import SongRetriever

logger = logging.getLogger(__name__)

# Max points a perfect semantic match (cosine similarity 1.0) can contribute,
# relative to the +1 a single categorical match (genre/mood/theme/artist) is worth.
SEMANTIC_WEIGHT = 3.0

# Max points a perfect audio-feature fit (from the taste-affinity model) can
# contribute. Weighted lower than semantic similarity since it's a coarser,
# indirect signal predicted from only genre + mood, not the full profile.
AUDIO_WEIGHT = 1.5

# Below this similarity, semantic evidence isn't worth mentioning in the explanation.
SEMANTIC_EXPLANATION_THRESHOLD = 0.6

# Below this closeness, predicted audio fit isn't worth mentioning in the explanation.
AUDIO_EXPLANATION_THRESHOLD = 0.8


class MusicRecommender:
    def __init__(self, retriever: SongRetriever | None = None, explainer: ClaudeExplainer | None = None) -> None:
        self.retriever = retriever or SongRetriever()
        self.explainer = explainer or ClaudeExplainer()

    def recommend(self, profile: TasteProfile, songs: List[Song], limit: int = 3) -> List[Recommendation]:
        if not profile.preferred_genres and not profile.preferred_moods and not profile.preferred_themes and not profile.favorite_artists:
            raise ValueError("Taste profile cannot be empty")

        retrieved = self.retriever.retrieve(profile, songs)
        ranked: List[Recommendation] = []
        for item in retrieved:
            song = item["song"]
            categorical_score = len(item["matched_genres"]) + len(item["matched_moods"]) + len(item["matched_themes"]) + len(item["matched_artists"])
            semantic_score = item.get("semantic_score", 0.0) * SEMANTIC_WEIGHT
            audio_score = item.get("audio_fit_score", 0.0) * AUDIO_WEIGHT
            evidence_score = categorical_score + semantic_score + audio_score
            if evidence_score == 0:
                continue

            explanation = self._explain(profile, song, item)
            ranked.append(
                Recommendation(
                    title=song.title,
                    artist=song.artist,
                    score=round(evidence_score + 2, 2),
                    explanation=explanation,
                )
            )

        ranked.sort(key=lambda rec: rec.score, reverse=True)
        return ranked[:limit]

    def _explain(self, profile: TasteProfile, song: Song, evidence: dict) -> str:
        """Prefer an LLM-generated explanation grounded in retrieved evidence;
        fall back to a template if Claude is unavailable or errors out."""
        try:
            return self.explainer.explain(profile, song, evidence)
        except Exception as exc:
            logger.warning("AI-generated explanation unavailable, falling back to template: %s", exc)
            return self._build_explanation(profile, song, evidence)

    def _build_explanation(self, profile: TasteProfile, song: Song, evidence: dict) -> str:
        parts = []
        if evidence["matched_genres"]:
            parts.append(f"it matches your interest in {', '.join(evidence['matched_genres'])}.")
        if evidence["matched_moods"]:
            parts.append(f"its {', '.join(evidence['matched_moods'])} mood fits your taste.")
        if evidence["matched_themes"]:
            parts.append(f"its themes of {', '.join(evidence['matched_themes'])} align with your preferences.")
        if evidence["matched_artists"]:
            parts.append(f"you already like {', '.join(evidence['matched_artists'])}.")
        if evidence.get("semantic_score", 0.0) >= SEMANTIC_EXPLANATION_THRESHOLD:
            parts.append("its lyrics and themes are a close semantic match for your taste, even beyond exact tags.")
        if evidence.get("audio_fit_score", 0.0) >= AUDIO_EXPLANATION_THRESHOLD:
            parts.append("its overall sound (how energetic, upbeat, and acoustic it is) closely matches what your stated taste predicts you'd enjoy.")

        if not parts:
            parts.append("it shares a strong overall feel with your current taste profile.")

        lyric_context = song.lyrics_excerpt.strip()
        if lyric_context:
            parts.append(f"The lyrics excerpt \"{lyric_context}\" also reinforce the connection.")

        return f"{profile.name}, this song is a good fit because " + " ".join(parts)


def recommend_songs(profile: TasteProfile, songs: List[Song], limit: int = 3) -> List[Recommendation]:
    return MusicRecommender().recommend(profile, songs, limit=limit)
