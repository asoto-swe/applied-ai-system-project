import logging
from typing import List, Optional

from .data import Song, TasteProfile
from .embeddings import VoyageEmbeddingClient, cosine_similarity

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


class SongRetriever:
    """Retrieve relevant songs and supporting evidence for a taste profile.

    Combines exact categorical matching (genre/mood/theme/artist) with
    semantic similarity over song meaning (lyrics/themes) via Voyage AI
    embeddings, so a song can surface because it *means* something close to
    the profile even without a literal tag match. If embeddings are
    unavailable (no API key, network error), retrieval degrades gracefully
    to categorical matching only.
    """

    def __init__(self, embedding_client: Optional[VoyageEmbeddingClient] = None) -> None:
        self.embedding_client = embedding_client or VoyageEmbeddingClient()

    def retrieve(self, profile: TasteProfile, songs: List[Song]) -> List[dict]:
        if not profile.preferred_genres and not profile.preferred_moods and not profile.preferred_themes and not profile.favorite_artists:
            raise ValueError("Taste profile cannot be empty")

        semantic_scores = self._semantic_scores(profile, songs)

        retrieved: List[dict] = []
        for song in songs:
            evidence = {
                "song": song,
                "matched_genres": [genre for genre in profile.preferred_genres if genre.lower() == song.genre.lower()],
                "matched_moods": [mood for mood in profile.preferred_moods if mood.lower() == song.mood.lower()],
                "matched_themes": [theme for theme in profile.preferred_themes if theme.lower() in [item.lower() for item in song.themes]],
                "matched_artists": [artist for artist in profile.favorite_artists if artist.lower() == song.artist.lower()],
                "semantic_score": semantic_scores.get(song.title, 0.0),
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
