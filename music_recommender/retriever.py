import logging
from typing import List, Optional

from .data import Song, TasteProfile
from .embeddings import VoyageEmbeddingClient, cosine_similarity
from .taste_model import TasteAffinityModel

logger = logging.getLogger(__name__)

# A song with no categorical match still counts as "relevant" if its meaning
# is close enough to the taste profile's description.
SEMANTIC_MATCH_THRESHOLD = 0.5


def _profile_text(profile: TasteProfile) -> str:
    parts = [
        f"Genres: {', '.join(profile.preferred_genres)}" if profile.preferred_genres else "",
        f"Moods: {', '.join(profile.preferred_moods)}" if profile.preferred_moods else "",
        f"Themes: {', '.join(profile.preferred_themes)}" if profile.preferred_themes else "",
        f"Favorite artists: {', '.join(profile.favorite_artists)}" if profile.favorite_artists else "",
    ]
    return ". ".join(part for part in parts if part)


def _song_text(song: Song) -> str:
    parts = [
        f"Genre: {song.genre}",
        f"Mood: {song.mood}",
        f"Themes: {', '.join(song.themes)}" if song.themes else "",
        song.lyrics_excerpt,
    ]
    return ". ".join(part for part in parts if part)


def _default_taste_model() -> Optional[TasteAffinityModel]:
    try:
        return TasteAffinityModel()
    except Exception as exc:
        logger.warning("Taste-affinity model unavailable, skipping audio-fit scoring: %s", exc)
        return None


class SongRetriever:
    """Retrieve relevant songs and supporting evidence for a taste profile.

    Combines three signals:
      - Exact categorical matching (genre/mood/theme/artist).
      - Semantic similarity over song meaning (lyrics/themes) via Voyage AI
        embeddings, so a song can surface because it *means* something close
        to the profile even without a literal tag match.
      - A predicted audio-feature fit (energy/valence/acousticness) from the
        taste-affinity model (see taste_model.py), a small model trained on a
        curated (genre, mood) -> audio-feature dataset.

    Each signal degrades gracefully and independently: if Voyage embeddings
    or the taste-affinity model are unavailable, retrieval falls back to
    whichever signals still work rather than failing outright.
    """

    def __init__(
        self,
        embedding_client: Optional[VoyageEmbeddingClient] = None,
        taste_model: Optional[TasteAffinityModel] = None,
    ) -> None:
        self.embedding_client = embedding_client or VoyageEmbeddingClient()
        self.taste_model = taste_model if taste_model is not None else _default_taste_model()

    def retrieve(self, profile: TasteProfile, songs: List[Song]) -> List[dict]:
        if not profile.preferred_genres and not profile.preferred_moods and not profile.preferred_themes and not profile.favorite_artists:
            raise ValueError("Taste profile cannot be empty")

        semantic_scores = self._semantic_scores(profile, songs)
        audio_target = self._predict_audio_target(profile)

        retrieved: List[dict] = []
        for song in songs:
            evidence = {
                "song": song,
                "matched_genres": [genre for genre in profile.preferred_genres if genre.lower() == song.genre.lower()],
                "matched_moods": [mood for mood in profile.preferred_moods if mood.lower() == song.mood.lower()],
                "matched_themes": [theme for theme in profile.preferred_themes if theme.lower() in [item.lower() for item in song.themes]],
                "matched_artists": [artist for artist in profile.favorite_artists if artist.lower() == song.artist.lower()],
                "semantic_score": semantic_scores.get(song.title, 0.0),
                "audio_fit_score": self._audio_fit_score(song, audio_target),
            }
            has_categorical_match = any(
                evidence[key] for key in ("matched_genres", "matched_moods", "matched_themes", "matched_artists")
            )
            if has_categorical_match or evidence["semantic_score"] >= SEMANTIC_MATCH_THRESHOLD:
                retrieved.append(evidence)
        return retrieved

    def _semantic_scores(self, profile: TasteProfile, songs: List[Song]) -> dict:
        if not songs:
            return {}
        try:
            query_embedding = self.embedding_client.embed_query(_profile_text(profile))
            song_embeddings = self.embedding_client.embed_documents([_song_text(song) for song in songs])
        except Exception as exc:
            logger.warning("Semantic retrieval unavailable, falling back to categorical matching only: %s", exc)
            return {}

        return {
            song.title: cosine_similarity(query_embedding, embedding)
            for song, embedding in zip(songs, song_embeddings)
        }

    def _predict_audio_target(self, profile: TasteProfile) -> Optional[dict]:
        if self.taste_model is None:
            return None
        try:
            return self.taste_model.predict(profile)
        except Exception as exc:
            logger.warning("Taste-affinity prediction failed, skipping audio-fit scoring: %s", exc)
            return None

    @staticmethod
    def _audio_fit_score(song: Song, target: Optional[dict]) -> float:
        if target is None:
            return 0.0
        diffs = [
            abs(song.energy - target["energy"]),
            abs(song.valence - target["valence"]),
            abs(song.acousticness - target["acousticness"]),
        ]
        closeness = 1.0 - (sum(diffs) / len(diffs))
        return max(0.0, min(1.0, closeness))
