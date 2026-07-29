import logging

from music_recommender.data import Song, TasteProfile
from music_recommender.recommender import recommend_songs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
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

    recommendations = recommend_songs(profile, songs, limit=2)
    print("Personalized recommendations:")
    for recommendation in recommendations:
        print(f"- {recommendation.title} by {recommendation.artist} (score: {recommendation.score})")
        print(f"  {recommendation.explanation}")


if __name__ == "__main__":
    main()
