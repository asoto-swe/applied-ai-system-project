from music_recommender.data import Song, TasteProfile
from music_recommender.recommender import MusicRecommender, recommend_songs


def test_recommend_songs_returns_ranked_matches():
    profile = TasteProfile(
        name="Maya",
        preferred_genres=["indie pop"],
        preferred_moods=["reflective"],
        preferred_themes=["nostalgia", "solitude"],
        favorite_artists=["Phoebe Bridgers"],
    )

    songs = [
        Song(
            title="Garden Song",
            artist="Phoebe Bridgers",
            genre="indie pop",
            mood="reflective",
            themes=["nostalgia", "solitude"],
            lyrics_excerpt="I miss the way we used to be",
        ),
        Song(
            title="Sunset Drive",
            artist="The xx",
            genre="dream pop",
            mood="introspective",
            themes=["late night", "dreams"],
            lyrics_excerpt="The city lights blur into the dark",
        ),
    ]

    recommendations = recommend_songs(profile, songs, limit=1)

    assert len(recommendations) == 1
    assert recommendations[0].title == "Garden Song"
    assert "nostalgia" in recommendations[0].explanation.lower()
    assert recommendations[0].score >= 3


def test_recommend_songs_rejects_empty_profile():
    profile = TasteProfile(name="Alex", preferred_genres=[], preferred_moods=[], preferred_themes=[], favorite_artists=[])
    songs = [
        Song(title="Test Song", artist="Test Artist", genre="pop", mood="energetic", themes=["joy"], lyrics_excerpt="Happy")
    ]

    try:
        recommend_songs(profile, songs, limit=1)
    except ValueError as exc:
        assert "profile" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty taste profile")


class FakeExplainer:
    """Deterministic stand-in for ClaudeExplainer — no network/API key needed."""

    def __init__(self, text=None, should_fail=False):
        self.text = text
        self.should_fail = should_fail
        self.last_call = None

    def explain(self, profile, song, evidence):
        self.last_call = (profile, song, evidence)
        if self.should_fail:
            raise RuntimeError("Claude unavailable")
        return self.text


def _sample_profile_and_songs():
    profile = TasteProfile(
        name="Maya",
        preferred_genres=["indie pop"],
        preferred_moods=["reflective"],
        preferred_themes=["nostalgia", "solitude"],
        favorite_artists=["Phoebe Bridgers"],
    )
    songs = [
        Song(
            title="Garden Song",
            artist="Phoebe Bridgers",
            genre="indie pop",
            mood="reflective",
            themes=["nostalgia", "solitude"],
            lyrics_excerpt="I miss the way we used to be",
        ),
    ]
    return profile, songs


def test_recommend_uses_ai_generated_explanation_when_available():
    profile, songs = _sample_profile_and_songs()
    explainer = FakeExplainer(text="Maya, this hits your nostalgic indie-pop sweet spot perfectly.")

    recommender = MusicRecommender(explainer=explainer)
    recommendations = recommender.recommend(profile, songs, limit=1)

    assert recommendations[0].explanation == "Maya, this hits your nostalgic indie-pop sweet spot perfectly."
    assert explainer.last_call is not None


def test_recommend_falls_back_to_template_when_ai_explanation_fails():
    profile, songs = _sample_profile_and_songs()
    explainer = FakeExplainer(should_fail=True)

    recommender = MusicRecommender(explainer=explainer)
    recommendations = recommender.recommend(profile, songs, limit=1)

    assert "nostalgia" in recommendations[0].explanation.lower()
    assert recommendations[0].explanation.startswith("Maya, this song is a good fit because")


def test_template_explanation_includes_genre_context_when_available():
    """RAG enhancement: the template explanation cites the second retrieved
    data source (genre background notes) when the retriever supplies one."""
    profile, songs = _sample_profile_and_songs()
    explainer = FakeExplainer(should_fail=True)  # forces the template path

    class GenreContextRetriever:
        def retrieve(self, profile, songs):
            evidence = {
                "song": songs[0],
                "matched_genres": ["indie pop"],
                "matched_moods": [],
                "matched_themes": [],
                "matched_artists": [],
                "semantic_score": 0.0,
                "audio_fit_score": 0.0,
                "genre_context": "test genre background note",
            }
            return [evidence]

    recommender = MusicRecommender(retriever=GenreContextRetriever(), explainer=explainer)
    recommendations = recommender.recommend(profile, songs, limit=1)

    assert "test genre background note" in recommendations[0].explanation


def test_diversity_penalty_prevents_a_single_artist_sweep():
    """Diversity/Fairness component: three strong matches from one artist and
    one weaker match from a different artist should not collapse into an
    all-same-artist top-3 — the artist penalty must pull the other artist in."""
    explainer = FakeExplainer(should_fail=True)  # force the deterministic template path
    songs = [
        Song(title="A1", artist="Artist A", genre="rock", mood="intense"),
        Song(title="A2", artist="Artist A", genre="rock", mood="intense"),
        Song(title="A3", artist="Artist A", genre="rock", mood="intense"),
        Song(title="B1", artist="Artist B", genre="rock", mood="calm"),
    ]
    profile = TasteProfile(
        name="Test", preferred_genres=["rock"], preferred_moods=["intense"], preferred_themes=[], favorite_artists=[]
    )

    recommender = MusicRecommender(explainer=explainer)
    recommendations = recommender.recommend(profile, songs, limit=3)

    artists_in_top_3 = {r.artist for r in recommendations}
    assert artists_in_top_3 == {"Artist A", "Artist B"}, (
        f"Expected the diversity penalty to include Artist B, got: {[(r.title, r.artist) for r in recommendations]}"
    )
    diversified = [r for r in recommendations if "keep your list varied" in r.explanation]
    assert len(diversified) >= 1, "At least one recommendation should note it was deprioritized for variety"
