import pytest

from music_recommender.ranking_strategies import (
    STRATEGIES,
    BalancedStrategy,
    EnergySimilarityStrategy,
    GenreFirstStrategy,
    MoodFirstStrategy,
    get_strategy,
)

# One song is a strong genre match with weak everything else; the other is a
# weak genre match but a very strong audio-fit match. A strategy that
# actually differs by what it optimizes for should rank these two songs in
# opposite order depending on which signal it favors.
GENRE_MATCH_EVIDENCE = {
    "matched_genres": ["rock"], "matched_moods": [], "matched_themes": [], "matched_artists": [],
    "semantic_score": 0.0, "audio_fit_score": 0.1,
}
AUDIO_MATCH_EVIDENCE = {
    "matched_genres": [], "matched_moods": [], "matched_themes": [], "matched_artists": [],
    "semantic_score": 0.0, "audio_fit_score": 0.95,
}


def test_genre_first_prefers_the_genre_match():
    strategy = GenreFirstStrategy()
    assert strategy.score(GENRE_MATCH_EVIDENCE) > strategy.score(AUDIO_MATCH_EVIDENCE)


def test_energy_similarity_prefers_the_audio_match():
    strategy = EnergySimilarityStrategy()
    assert strategy.score(AUDIO_MATCH_EVIDENCE) > strategy.score(GENRE_MATCH_EVIDENCE)


def test_strategies_are_registered_and_retrievable_by_name():
    assert set(STRATEGIES) == {"balanced", "genre-first", "mood-first", "energy-similarity"}
    for name in STRATEGIES:
        strategy = get_strategy(name)
        assert strategy.name == name


def test_get_strategy_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_strategy("not-a-real-strategy")


def test_mood_first_weighs_mood_over_genre():
    strategy = MoodFirstStrategy()
    mood_match = {**GENRE_MATCH_EVIDENCE, "matched_genres": [], "matched_moods": ["calm"]}
    genre_match = {**GENRE_MATCH_EVIDENCE}
    assert strategy.score(mood_match) > strategy.score(genre_match)


def test_balanced_strategy_blends_all_signals():
    strategy = BalancedStrategy()
    # Pure categorical match should still beat a song with no evidence at all.
    no_evidence = {
        "matched_genres": [], "matched_moods": [], "matched_themes": [], "matched_artists": [],
        "semantic_score": 0.0, "audio_fit_score": 0.0,
    }
    assert strategy.score(GENRE_MATCH_EVIDENCE) > strategy.score(no_evidence)


def test_missing_generated_attribute_keys_default_to_no_bonus():
    """Evidence dicts built before the AI-generated attributes existed (no
    matched_detailed_moods/popularity_score keys) must still score fine."""
    strategy = BalancedStrategy()
    assert strategy.score(GENRE_MATCH_EVIDENCE) == strategy.score(GENRE_MATCH_EVIDENCE)  # no KeyError


def test_generated_attribute_bonus_increases_score_consistently_across_strategies():
    base = {
        "matched_genres": [], "matched_moods": [], "matched_themes": [], "matched_artists": [],
        "semantic_score": 0.0, "audio_fit_score": 0.0,
    }
    enriched = {**base, "matched_detailed_moods": ["wistful"], "popularity_score": 0.8}

    for strategy_cls in (BalancedStrategy, GenreFirstStrategy, MoodFirstStrategy, EnergySimilarityStrategy):
        strategy = strategy_cls()
        assert strategy.score(enriched) > strategy.score(base), f"{strategy.name} did not apply the bonus"
