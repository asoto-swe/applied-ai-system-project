"""Reliability and consistency evaluation report for the music recommender.

This is the project's "Reliability or Testing System": it measures whether
the AI pipeline behaves consistently and degrades gracefully on normal and
adversarial/edge-case taste profiles, rather than just checking that the
code runs correctly (that's what tests/ does).

Run:
    python scripts/evaluate_reliability.py

Exits with status 1 if any check fails, so it can be used as a CI gate.
"""
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from music_recommender.data import Song, TasteProfile
from music_recommender.recommender import MusicRecommender
from music_recommender.reliability import (
    check_consistency,
    check_empty_profile_guardrail,
    check_explanation_groundedness,
    check_graceful_degradation,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

CATALOG = [
    Song(
        title="Garden Song", artist="Phoebe Bridgers", genre="indie pop", mood="reflective",
        themes=["nostalgia", "solitude"], lyrics_excerpt="I miss the way we used to be",
        energy=0.40, valence=0.50, acousticness=0.55,
    ),
    Song(
        title="Sunset Drive", artist="The xx", genre="dream pop", mood="introspective",
        themes=["late night", "dreams"], lyrics_excerpt="The city lights blur into the dark",
        energy=0.35, valence=0.50, acousticness=0.45,
    ),
    Song(
        title="Sunrise City", artist="Neon Echo", genre="pop", mood="happy",
        themes=["new beginnings"], lyrics_excerpt="Everything feels possible today",
        energy=0.85, valence=0.90, acousticness=0.15,
    ),
    Song(
        title="Library Rain", artist="Paper Lanterns", genre="lofi", mood="chill",
        themes=["studying", "quiet nights"], lyrics_excerpt="Pages turning, rain on glass",
        energy=0.25, valence=0.55, acousticness=0.75,
    ),
    Song(
        title="Storm Runner", artist="Voltline", genre="rock", mood="intense",
        themes=["defiance"], lyrics_excerpt="I won't back down from the storm",
        energy=0.90, valence=0.55, acousticness=0.15,
    ),
    Song(
        title="Ironclad", artist="Blackforge", genre="metal", mood="intense",
        themes=["resilience"], lyrics_excerpt="Forged in fire, I don't break",
        energy=0.95, valence=0.35, acousticness=0.05,
    ),
    Song(
        title="Coffee Shop Stories", artist="Slow Stereo", genre="folk", mood="nostalgic",
        themes=["memory", "small towns"], lyrics_excerpt="Every booth here holds a story",
        energy=0.30, valence=0.55, acousticness=0.70,
    ),
    Song(
        title="Pulse Reactor", artist="Circuit Halo", genre="edm", mood="euphoric",
        themes=["release"], lyrics_excerpt="Let it go, feel the pulse take over",
        energy=0.95, valence=0.85, acousticness=0.05,
    ),
]

# Normal profiles plus adversarial/edge-case ones, in the same spirit as the
# Module 3 source project's stress tests: conflicting signals, a genre
# absent from the catalog, minimal signal, and a deliberately broad profile.
PROFILES = [
    TasteProfile(
        name="Maya", preferred_genres=["indie pop"], preferred_moods=["reflective"],
        preferred_themes=["nostalgia", "solitude"], favorite_artists=["Phoebe Bridgers"],
    ),
    TasteProfile(
        name="Deshawn (conflicting genre/mood)", preferred_genres=["metal"], preferred_moods=["calm"],
        preferred_themes=[], favorite_artists=[],
    ),
    TasteProfile(
        name="Priya (genre absent from catalog)", preferred_genres=["kpop"], preferred_moods=["happy"],
        preferred_themes=[], favorite_artists=[],
    ),
    TasteProfile(
        name="Jordan (minimal signal)", preferred_genres=[], preferred_moods=[],
        preferred_themes=[], favorite_artists=["Phoebe Bridgers"],
    ),
    TasteProfile(
        name="Riley (broad profile)",
        preferred_genres=["pop", "edm", "rock", "metal", "folk", "lofi"],
        preferred_moods=["happy", "intense", "chill"], preferred_themes=[], favorite_artists=[],
    ),
]


def main() -> None:
    recommender = MusicRecommender()
    results = []

    for profile in PROFILES:
        results.append(check_consistency(recommender, profile, CATALOG))
        results.append(check_graceful_degradation(recommender, profile, CATALOG))
        groundedness = check_explanation_groundedness(recommender, profile, CATALOG)
        if groundedness is not None:
            results.append(groundedness)

    results.append(check_empty_profile_guardrail(recommender, CATALOG))

    print("=" * 70)
    print("RELIABILITY REPORT")
    print("=" * 70)
    passed_count = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        passed_count += int(result.passed)
        print(f"[{status}] {result.name}")
        print(f"       {result.detail}")

    print("-" * 70)
    print(f"{passed_count}/{len(results)} checks passed")
    print("=" * 70)

    if passed_count < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
