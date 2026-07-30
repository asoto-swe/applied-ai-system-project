"""A small shared demo catalog used by app.py and scripts/evaluate_reliability.py.

Centralized so the two don't silently diverge into two different "the
demo catalog" datasets over time.
"""
from .data import Song
from .song_attributes import SongAttributesStore

_BASE_CATALOG = [
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
    Song(
        title="Midnight Blue", artist="Vell Roland", genre="jazz", mood="relaxed",
        themes=["late nights", "empty rooms"], lyrics_excerpt="The bar's gone quiet, just the piano and me",
        energy=0.40, valence=0.65, acousticness=0.55,
    ),
    Song(
        title="Quiet Reverie", artist="Elena Voss", genre="classical", mood="serene",
        themes=["stillness", "reflection"], lyrics_excerpt="",
        energy=0.30, valence=0.60, acousticness=0.85,
    ),
    Song(
        title="Floating Room", artist="Halcyon Drift", genre="ambient", mood="dreamy",
        themes=["weightlessness", "dreams"], lyrics_excerpt="No floor, no ceiling, just the hum",
        energy=0.20, valence=0.50, acousticness=0.65,
    ),
    Song(
        title="Blueprint", artist="Marcus Iron", genre="hip hop", mood="confident",
        themes=["ambition", "hustle"], lyrics_excerpt="Every brick I laid, I laid it on my own",
        energy=0.75, valence=0.65, acousticness=0.10,
    ),
    Song(
        title="Palm Shade", artist="Solstice Roots", genre="reggae", mood="relaxed",
        themes=["ease", "sunlight"], lyrics_excerpt="Let the afternoon just roll on by",
        energy=0.55, valence=0.75, acousticness=0.35,
    ),
    Song(
        title="Dust and Highways", artist="June Callahan", genre="country", mood="nostalgic",
        themes=["home", "memory"], lyrics_excerpt="Every mile marker's a memory now",
        energy=0.45, valence=0.60, acousticness=0.55,
    ),
    Song(
        title="Neon Rewind", artist="Chrome Static", genre="synthwave", mood="nostalgic",
        themes=["retro glow", "night drives"], lyrics_excerpt="Dashboard glow, same road, different year",
        energy=0.60, valence=0.55, acousticness=0.15,
    ),
    Song(
        title="Broken Glass", artist="The Discontents", genre="punk", mood="angry",
        themes=["rebellion", "frustration"], lyrics_excerpt="You built the walls, I brought the glass",
        energy=0.90, valence=0.35, acousticness=0.10,
    ),
    Song(
        title="Slow Burn", artist="Nadia Wells", genre="r&b", mood="romantic",
        themes=["intimacy", "longing"], lyrics_excerpt="Take your time, we've got the whole night",
        energy=0.50, valence=0.65, acousticness=0.30,
    ),
    Song(
        title="Rust and Rain", artist="Otis Graves", genre="blues", mood="melancholic",
        themes=["heartbreak", "weariness"], lyrics_excerpt="Rust on the fence, rain on the road",
        energy=0.40, valence=0.35, acousticness=0.45,
    ),
]

# Enriched with AI-generated attributes (popularity, release decade, detailed
# mood tags, vocal style, instrumentation notes) if available — see
# song_attributes.py and scripts/generate_song_attributes.py. Falls back to
# the base catalog's defaults (None/empty) if the generated data isn't present.
CATALOG = SongAttributesStore().enrich(_BASE_CATALOG)
