from dataclasses import dataclass, field
from typing import List


@dataclass
class Song:
    title: str
    artist: str
    genre: str
    mood: str
    themes: List[str] = field(default_factory=list)
    lyrics_excerpt: str = ""
    # Audio features on a 0.0-1.0 scale, compared against the taste-affinity
    # model's predicted target profile (see taste_model.py). Default to a
    # neutral midpoint for songs whose audio features aren't known.
    energy: float = 0.5
    valence: float = 0.5
    acousticness: float = 0.5


@dataclass
class TasteProfile:
    name: str
    preferred_genres: List[str]
    preferred_moods: List[str]
    preferred_themes: List[str]
    favorite_artists: List[str]


@dataclass
class Recommendation:
    title: str
    artist: str
    score: float
    explanation: str
