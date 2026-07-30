import logging
from typing import List

from .data import Recommendation, Song, TasteProfile
from .explainer import ClaudeExplainer
from .ranking_strategies import BalancedStrategy, RankingStrategy
from .retriever import SongRetriever

logger = logging.getLogger(__name__)

# Below this similarity, semantic evidence isn't worth mentioning in the explanation.
SEMANTIC_EXPLANATION_THRESHOLD = 0.6

# Below this closeness, predicted audio fit isn't worth mentioning in the explanation.
AUDIO_EXPLANATION_THRESHOLD = 0.8

# Diversity/fairness penalties applied while building the final top-`limit`
# list, so recommendations don't collapse into repeats of one artist or
# genre — the same filter-bubble problem the original TuneMatch's model
# card documented for its energy-weighted scoring. Weighted harsher on
# repeat artists (hearing the same artist twice feels worse than two songs
# of the same genre), mirroring the source project's own weighting.
ARTIST_PENALTY = 2.0
GENRE_PENALTY = 0.75


class MusicRecommender:
    def __init__(
        self,
        retriever: SongRetriever | None = None,
        explainer: ClaudeExplainer | None = None,
        ranking_strategy: RankingStrategy | None = None,
    ) -> None:
        self.retriever = retriever or SongRetriever()
        self.explainer = explainer or ClaudeExplainer()
        self.ranking_strategy = ranking_strategy or BalancedStrategy()

    def recommend(self, profile: TasteProfile, songs: List[Song], limit: int = 3) -> List[Recommendation]:
        if not profile.preferred_genres and not profile.preferred_moods and not profile.preferred_themes and not profile.favorite_artists:
            raise ValueError("Taste profile cannot be empty")

        retrieved = self.retriever.retrieve(profile, songs)
        candidates = []
        for item in retrieved:
            song = item["song"]
            evidence_score = self.ranking_strategy.score(item)
            if evidence_score <= 0:
                continue
            explanation = self._explain(profile, song, item)
            candidates.append({"song": song, "score": evidence_score + 2, "explanation": explanation})

        return self._diversify(candidates, limit)

    def _diversify(self, candidates: List[dict], limit: int) -> List[Recommendation]:
        """Greedily builds the top-`limit` list, subtracting a penalty each
        time a candidate's artist or genre is already present among the
        picks made so far. This is the project's Diversity/Fairness
        component: without it, a listener's top results can collapse into
        several songs by the same artist or genre, the exact "safe middle"
        filter-bubble effect documented in the original TuneMatch."""
        remaining = sorted(candidates, key=lambda c: c["score"], reverse=True)
        chosen: List[Recommendation] = []
        artist_counts: dict = {}
        genre_counts: dict = {}

        while remaining and len(chosen) < limit:
            best_index, best_adjusted, best_penalty = 0, None, (0.0, 0.0)
            for index, candidate in enumerate(remaining):
                artist_pen = ARTIST_PENALTY * artist_counts.get(candidate["song"].artist, 0)
                genre_pen = GENRE_PENALTY * genre_counts.get(candidate["song"].genre, 0)
                adjusted = candidate["score"] - artist_pen - genre_pen
                if best_adjusted is None or adjusted > best_adjusted:
                    best_index, best_adjusted, best_penalty = index, adjusted, (artist_pen, genre_pen)

            picked = remaining.pop(best_index)
            song = picked["song"]
            explanation = picked["explanation"]
            artist_pen, genre_pen = best_penalty
            if artist_pen > 0 or genre_pen > 0:
                explanation += (
                    f" (Ranked slightly lower to keep your list varied — you already have a "
                    f"pick from {'this artist' if artist_pen > 0 else 'this genre'} above.)"
                )

            chosen.append(
                Recommendation(
                    title=song.title, artist=song.artist, genre=song.genre,
                    score=round(best_adjusted, 2), explanation=explanation,
                )
            )
            artist_counts[song.artist] = artist_counts.get(song.artist, 0) + 1
            genre_counts[song.genre] = genre_counts.get(song.genre, 0) + 1

        return chosen

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

        explanation = f"{profile.name}, this song is a good fit because " + " ".join(parts)

        genre_context = evidence.get("genre_context")
        if genre_context:
            explanation += f" For context, {song.genre} music tends to be: {genre_context}"

        return explanation


def recommend_songs(profile: TasteProfile, songs: List[Song], limit: int = 3) -> List[Recommendation]:
    return MusicRecommender().recommend(profile, songs, limit=limit)
