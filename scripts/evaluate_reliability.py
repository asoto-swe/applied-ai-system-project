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

from music_recommender.data import TasteProfile
from music_recommender.demo_catalog import CATALOG
from music_recommender.recommender import MusicRecommender
from music_recommender.reliability import (
    check_consistency,
    check_empty_profile_guardrail,
    check_explanation_groundedness,
    check_graceful_degradation,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

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
