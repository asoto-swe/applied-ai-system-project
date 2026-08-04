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
    parser.add_argument("--name", default="Alex", help="Your name, for the printed header.")
    parser.add_argument(
        "--genres", default="rock,lofi", help="Comma-separated genres you like, e.g. 'hip hop,indie pop'."
    )
    parser.add_argument(
        "--moods", default="happy,intense", help="Comma-separated moods you're after, e.g. 'chill,reflective'."
    )
    parser.add_argument("--themes", default="", help="Comma-separated themes you like, e.g. 'heartbreak,nostalgia'.")
    parser.add_argument("--artists", default="", help="Comma-separated favorite artists, e.g. 'The xx,Phoebe Bridgers'.")
    return parser.parse_args()


def _split_csv(raw: str) -> list:
    return [item.strip() for item in raw.split(",") if item.strip()]


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

    # Defaults reproduce the original demo profile (broad on purpose, so
    # switching --mode visibly reorders results). Pass --genres/--moods/etc.
    # to get recommendations based on your own actual taste instead.
    profile = TasteProfile(
        name=args.name,
        preferred_genres=_split_csv(args.genres),
        preferred_moods=_split_csv(args.moods),
        preferred_themes=_split_csv(args.themes),
        favorite_artists=_split_csv(args.artists),
    )

    recommender = MusicRecommender(ranking_strategy=get_strategy(args.mode))
    recommendations = recommender.recommend(profile, CATALOG, limit=args.limit)

    print(f"Personalized recommendations for {profile.name} - ranking mode: '{args.mode}'\n")
    print(render_table(recommendations))


if __name__ == "__main__":
    main()
