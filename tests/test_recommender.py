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
