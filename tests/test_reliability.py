from music_recommender.data import Song, TasteProfile
from music_recommender.recommender import MusicRecommender
from music_recommender.reliability import (
    check_consistency,
    check_empty_profile_guardrail,
    check_graceful_degradation,
)


def _catalog():
    return [
        Song(
            title="Garden Song", artist="Phoebe Bridgers", genre="indie pop", mood="reflective",
            themes=["nostalgia", "solitude"], lyrics_excerpt="I miss the way we used to be",
        ),
        Song(
            title="Ironclad", artist="Blackforge", genre="metal", mood="intense",
            themes=["resilience"], lyrics_excerpt="Forged in fire, I don't break",
        ),
    ]


def test_recommendations_are_consistent_across_repeated_runs():
    profile = TasteProfile(
        name="Maya", preferred_genres=["indie pop"], preferred_moods=["reflective"],
        preferred_themes=["nostalgia"], favorite_artists=[],
    )

    result = check_consistency(MusicRecommender(), profile, _catalog(), runs=3)

    assert result.passed, result.detail


def test_pipeline_degrades_gracefully_on_conflicting_profile():
    conflicting_profile = TasteProfile(
        name="Deshawn", preferred_genres=["metal"], preferred_moods=["calm"],
        preferred_themes=[], favorite_artists=[],
    )

    result = check_graceful_degradation(MusicRecommender(), conflicting_profile, _catalog())

    assert result.passed, result.detail


def test_empty_profile_guardrail_still_enforced():
    result = check_empty_profile_guardrail(MusicRecommender(), _catalog())

    assert result.passed, result.detail
