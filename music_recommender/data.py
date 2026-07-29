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
