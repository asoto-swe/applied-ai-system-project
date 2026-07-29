from music_recommender.data import TasteProfile
from music_recommender.taste_model import TARGET_COLUMNS, TasteAffinityModel


def test_predict_returns_valid_audio_profile_for_known_taste():
    model = TasteAffinityModel()
    profile = TasteProfile(
        name="Maya",
        preferred_genres=["lofi"],
        preferred_moods=["chill"],
        preferred_themes=[],
        favorite_artists=[],
    )

    prediction = model.predict(profile)

    assert prediction is not None
    assert set(prediction.keys()) == set(TARGET_COLUMNS)
    for value in prediction.values():
        assert 0.0 <= value <= 1.0
    # lofi/chill training examples are low-energy, high-acousticness —
    # the model should reflect that even on a single training-set match.
    assert prediction["energy"] < 0.5
    assert prediction["acousticness"] > 0.5


def test_predict_averages_across_multiple_genres_and_moods():
    model = TasteAffinityModel()
    profile = TasteProfile(
        name="Sam",
        preferred_genres=["edm", "ambient"],
        preferred_moods=["euphoric", "calm"],
        preferred_themes=[],
        favorite_artists=[],
    )

    prediction = model.predict(profile)

    assert prediction is not None
    # edm/euphoric is high-energy, ambient/calm is low-energy — averaging
    # across the cross product should land somewhere in between the extremes.
    assert 0.0 < prediction["energy"] < 1.0


def test_predict_returns_none_without_genre_or_mood_signal():
    model = TasteAffinityModel()
    profile = TasteProfile(
        name="Alex",
        preferred_genres=[],
        preferred_moods=[],
        preferred_themes=["nostalgia"],
        favorite_artists=["Someone"],
    )

    assert model.predict(profile) is None
