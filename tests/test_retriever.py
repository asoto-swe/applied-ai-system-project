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


class FakeTasteModel:
    """Deterministic stand-in for TasteAffinityModel — no sklearn training needed."""

    def __init__(self, prediction):
        self.prediction = prediction
        self.last_profile = None

    def predict(self, profile):
        self.last_profile = profile
        return self.prediction


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


def test_retrieve_scores_audio_fit_from_taste_model():
    profile = TasteProfile(
        name="Sam", preferred_genres=["lofi"], preferred_moods=["chill"], preferred_themes=[], favorite_artists=[]
    )
    songs = [
        Song(
            title="Close Sound",
            artist="X",
            genre="lofi",
            mood="chill",
            themes=[],
            lyrics_excerpt="",
            energy=0.2,
            valence=0.55,
            acousticness=0.8,
        )
    ]
    fake_embedding_client = FakeEmbeddingClient(query_vector=[0.0], document_vectors=[[0.0]])
    fake_taste_model = FakeTasteModel(prediction={"energy": 0.2, "valence": 0.55, "acousticness": 0.8})
    retriever = SongRetriever(embedding_client=fake_embedding_client, taste_model=fake_taste_model)

    retrieved = retriever.retrieve(profile, songs)

    assert len(retrieved) == 1
    assert retrieved[0]["audio_fit_score"] == 1.0
    assert fake_taste_model.last_profile == profile


def test_retrieve_falls_back_gracefully_when_taste_model_unavailable():
    class BrokenTasteModel:
        def predict(self, profile):
            raise RuntimeError("model unavailable")

    profile = TasteProfile(
        name="Sam", preferred_genres=["indie pop"], preferred_moods=[], preferred_themes=[], favorite_artists=[]
    )
    songs = [
        Song(title="Tagged Match", artist="X", genre="indie pop", mood="chill", themes=[], lyrics_excerpt="")
    ]
    fake_embedding_client = FakeEmbeddingClient(query_vector=[1.0], document_vectors=[[1.0]])
    retriever = SongRetriever(embedding_client=fake_embedding_client, taste_model=BrokenTasteModel())

    retrieved = retriever.retrieve(profile, songs)

    assert len(retrieved) == 1
    assert retrieved[0]["audio_fit_score"] == 0.0


def test_retrieve_attaches_genre_context_from_second_data_source():
    class FakeGenreKnowledgeBase:
        def lookup(self, genre):
            return "test background note" if genre == "indie pop" else None

    profile = TasteProfile(
        name="Sam", preferred_genres=["indie pop"], preferred_moods=[], preferred_themes=[], favorite_artists=[]
    )
    songs = [
        Song(title="Tagged Match", artist="X", genre="indie pop", mood="chill", themes=[], lyrics_excerpt="")
    ]
    fake_embedding_client = FakeEmbeddingClient(query_vector=[1.0], document_vectors=[[1.0]])
    retriever = SongRetriever(embedding_client=fake_embedding_client, genre_notes=FakeGenreKnowledgeBase())

    retrieved = retriever.retrieve(profile, songs)

    assert retrieved[0]["genre_context"] == "test background note"


def test_retrieve_genre_context_is_none_for_unknown_genre():
    class FakeGenreKnowledgeBase:
        def lookup(self, genre):
            return None

    profile = TasteProfile(
        name="Sam", preferred_genres=["glitchcore"], preferred_moods=[], preferred_themes=[], favorite_artists=[]
    )
    songs = [
        Song(title="Obscure", artist="X", genre="glitchcore", mood="chill", themes=[], lyrics_excerpt="")
    ]
    fake_embedding_client = FakeEmbeddingClient(query_vector=[1.0], document_vectors=[[1.0]])
    retriever = SongRetriever(embedding_client=fake_embedding_client, genre_notes=FakeGenreKnowledgeBase())

    retrieved = retriever.retrieve(profile, songs)

    assert retrieved[0]["genre_context"] is None


def test_retrieve_computes_detailed_mood_match_and_popularity_score():
    profile = TasteProfile(
        name="Sam", preferred_genres=["indie pop"], preferred_moods=["wistful"],
        preferred_themes=[], favorite_artists=[],
    )
    songs = [
        Song(
            title="Tagged Match", artist="X", genre="indie pop", mood="reflective",
            themes=[], lyrics_excerpt="", detailed_mood_tags=["wistful", "tender"], popularity=60,
        )
    ]
    fake_embedding_client = FakeEmbeddingClient(query_vector=[1.0], document_vectors=[[1.0]])
    retriever = SongRetriever(embedding_client=fake_embedding_client)

    retrieved = retriever.retrieve(profile, songs)

    assert retrieved[0]["matched_detailed_moods"] == ["wistful"]
    assert retrieved[0]["popularity_score"] == 0.6


def test_retrieve_deduplicates_repeated_profile_entries_case_insensitively():
    profile = TasteProfile(
        name="Sam", preferred_genres=["rock", "Rock", "rock"], preferred_moods=["intense"] * 5,
        preferred_themes=[], favorite_artists=[],
    )
    songs = [
        Song(title="Storm Runner", artist="Voltline", genre="rock", mood="intense", themes=[], lyrics_excerpt="")
    ]
    fake_embedding_client = FakeEmbeddingClient(query_vector=[1.0], document_vectors=[[1.0]])
    retriever = SongRetriever(embedding_client=fake_embedding_client)

    retrieved = retriever.retrieve(profile, songs)

    assert retrieved[0]["matched_genres"] == ["rock"]
    assert retrieved[0]["matched_moods"] == ["intense"]


def test_retrieve_defaults_popularity_score_to_zero_when_unset():
    profile = TasteProfile(
        name="Sam", preferred_genres=["indie pop"], preferred_moods=[], preferred_themes=[], favorite_artists=[]
    )
    songs = [Song(title="No Attrs", artist="X", genre="indie pop", mood="reflective")]
    fake_embedding_client = FakeEmbeddingClient(query_vector=[1.0], document_vectors=[[1.0]])
    retriever = SongRetriever(embedding_client=fake_embedding_client)

    retrieved = retriever.retrieve(profile, songs)

    assert retrieved[0]["popularity_score"] == 0.0
    assert retrieved[0]["matched_detailed_moods"] == []
