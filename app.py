import argparse
import logging
import textwrap

from tabulate import tabulate

from music_recommender.data import TasteProfile
from music_recommender.demo_catalog import CATALOG
from music_recommender.ranking_strategies import STRATEGIES, get_strategy
from music_recommender.recommender import MusicRecommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

EXPLANATION_WRAP_WIDTH = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TuneMatch demo with a chosen ranking mode.")
    parser.add_argument(
        "--mode",
        choices=sorted(STRATEGIES),
        default="balanced",
        help="Ranking strategy to optimize for (default: balanced).",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of recommendations to return (default: 3).")
    return parser.parse_args()


def render_table(recommendations) -> str:
    rows = [
        [
            rec.title,
            rec.artist,
            rec.genre,
            rec.score,
            textwrap.fill(rec.explanation, EXPLANATION_WRAP_WIDTH),
        ]
        for rec in recommendations
    ]
    return tabulate(rows, headers=["Title", "Artist", "Genre", "Score", "Why"], tablefmt="grid")


def main() -> None:
    args = parse_args()

    # A deliberately broad profile (matches several songs on different
    # signals) so switching --mode visibly reorders the results, not just
    # changes their score. See README for a narrower single-match example.
    profile = TasteProfile(
        name="Alex",
        preferred_genres=["rock", "lofi"],
        preferred_moods=["happy", "intense"],
        preferred_themes=[],
        favorite_artists=[],
    )

    recommender = MusicRecommender(ranking_strategy=get_strategy(args.mode))
    recommendations = recommender.recommend(profile, CATALOG, limit=args.limit)

    print(f"Personalized recommendations for {profile.name} - ranking mode: '{args.mode}'\n")
    print(render_table(recommendations))


if __name__ == "__main__":
    main()
