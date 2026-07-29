from typing import List

from .data import Recommendation, Song, TasteProfile
from .retriever import SongRetriever

# Max points a perfect semantic match (cosine similarity 1.0) can contribute,
# relative to the +1 a single categorical match (genre/mood/theme/artist) is worth.
SEMANTIC_WEIGHT = 3.0

# Below this similarity, semantic evidence isn't worth mentioning in the explanation.
SEMANTIC_EXPLANATION_THRESHOLD = 0.6


class MusicRecommender:
    def __init__(self, retriever: SongRetriever | None = None) -> None:
        self.retriever = retriever or SongRetriever()

    def recommend(self, profile: TasteProfile, songs: List[Song], limit: int = 3) -> List[Recommendation]:
        if not profile.preferred_genres and not profile.preferred_moods and not profile.preferred_themes and not profile.favorite_artists:
            raise ValueError("Taste profile cannot be empty")

        retrieved = self.retriever.retrieve(profile, songs)
        ranked: List[Recommendation] = []
        for item in retrieved:
            song = item["song"]
            categorical_score = len(item["matched_genres"]) + len(item["matched_moods"]) + len(item["matched_themes"]) + len(item["matched_artists"])
            semantic_score = item.get("semantic_score", 0.0) * SEMANTIC_WEIGHT
            evidence_score = categorical_score + semantic_score
            if evidence_score == 0:
                continue

            explanation = self._build_explanation(profile, song, item)
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

        if not parts:
            parts.append("it shares a strong overall feel with your current taste profile.")

        lyric_context = song.lyrics_excerpt.strip()
        if lyric_context:
            parts.append(f"The lyrics excerpt \"{lyric_context}\" also reinforce the connection.")

        return f"{profile.name}, this song is a good fit because " + " ".join(parts)


def recommend_songs(profile: TasteProfile, songs: List[Song], limit: int = 3) -> List[Recommendation]:
    return MusicRecommender().recommend(profile, songs, limit=limit)
