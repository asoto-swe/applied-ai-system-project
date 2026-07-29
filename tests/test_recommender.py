from music_recommender.data import Song, TasteProfile
from music_recommender.recommender import recommend_songs


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
