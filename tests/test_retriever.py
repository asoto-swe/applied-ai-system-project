from music_recommender.data import Song, TasteProfile
from music_recommender.retriever import SongRetriever


class FakeEmbeddingClient:
    """Deterministic stand-in for VoyageEmbeddingClient — no network/API key needed."""

    def __init__(self, query_vector, document_vectors):
        self.query_vector = query_vector
        self.document_vectors = document_vectors

    def embed_query(self, text):
        return self.query_vector

    def embed_documents(self, texts, input_type="document"):
        return self.document_vectors


def test_retrieve_includes_song_on_semantic_match_alone():
    profile = TasteProfile(
        name="Sam",
        preferred_genres=["indie pop"],
        preferred_moods=["reflective"],
        preferred_themes=[],
        favorite_artists=[],
    )
    songs = [
        Song(
            title="No Tag Overlap",
            artist="Someone Else",
            genre="dream pop",
            mood="introspective",
            themes=["longing"],
            lyrics_excerpt="Nothing here shares a literal tag with the profile",
        ),
    ]
    fake_client = FakeEmbeddingClient(query_vector=[1.0, 0.0], document_vectors=[[1.0, 0.0]])
    retriever = SongRetriever(embedding_client=fake_client)

    retrieved = retriever.retrieve(profile, songs)

    assert len(retrieved) == 1
    assert retrieved[0]["semantic_score"] == 1.0
    assert retrieved[0]["matched_genres"] == []


def test_retrieve_excludes_song_with_no_match_of_any_kind():
    profile = TasteProfile(
        name="Sam",
        preferred_genres=["indie pop"],
        preferred_moods=[],
        preferred_themes=[],
        favorite_artists=[],
    )
    songs = [
        Song(title="Unrelated", artist="X", genre="metal", mood="angry", themes=["rage"], lyrics_excerpt="")
    ]
    fake_client = FakeEmbeddingClient(query_vector=[1.0, 0.0], document_vectors=[[0.0, 1.0]])
    retriever = SongRetriever(embedding_client=fake_client)

    retrieved = retriever.retrieve(profile, songs)

    assert retrieved == []


def test_retrieve_falls_back_gracefully_when_embeddings_unavailable():
    class BrokenEmbeddingClient:
        def embed_query(self, text):
            raise RuntimeError("no API key")

        def embed_documents(self, texts, input_type="document"):
            raise RuntimeError("no API key")

    profile = TasteProfile(
        name="Sam",
        preferred_genres=["indie pop"],
        preferred_moods=[],
        preferred_themes=[],
        favorite_artists=[],
    )
    songs = [
        Song(title="Tagged Match", artist="X", genre="indie pop", mood="chill", themes=[], lyrics_excerpt="")
    ]
    retriever = SongRetriever(embedding_client=BrokenEmbeddingClient())

    retrieved = retriever.retrieve(profile, songs)

    assert len(retrieved) == 1
    assert retrieved[0]["semantic_score"] == 0.0
    assert retrieved[0]["matched_genres"] == ["indie pop"]
