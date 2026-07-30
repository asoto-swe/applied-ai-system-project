"""Loads AI-generated song attributes (popularity, release decade, detailed
mood tags, vocal style, instrumentation notes) and enriches Song objects
with them.

The attributes themselves are generated offline by an agentic Claude
workflow (see scripts/generate_song_attributes.py) and persisted to
music_recommender/data/song_attributes.csv, keyed by song title. This
module only *loads and applies* that data at runtime — it never calls
Claude itself, so using it never requires an API key.

If the CSV is missing, unreadable, or doesn't have an entry for a given
song, that song's new attributes are simply left at their dataclass
defaults (None / empty list) — the same graceful-degradation pattern used
throughout this project.
"""
from __future__ import annotations

import csv
import logging
import pathlib
from dataclasses import replace
from typing import Dict, List

from .data import Song

logger = logging.getLogger(__name__)

_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
ATTRIBUTES_PATH = _PACKAGE_DIR / "data" / "song_attributes.csv"


class SongAttributesStore:
    """Looks up AI-generated attributes for a song by title."""

    def __init__(self) -> None:
        self._rows: Dict[str, dict] = self._load_rows()

    def _load_rows(self) -> Dict[str, dict]:
        if not ATTRIBUTES_PATH.exists():
            logger.warning("Song attributes file not found at %s; songs will use default attributes.", ATTRIBUTES_PATH)
            return {}
        try:
            with open(ATTRIBUTES_PATH, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return {row["title"]: row for row in rows}
        except Exception as exc:
            logger.warning("Could not load song attributes, continuing with defaults: %s", exc)
            return {}

    def enrich(self, songs: List[Song]) -> List[Song]:
        """Returns new Song objects with generated attributes applied where
        available. Songs with no matching row are returned unchanged."""
        enriched = []
        for song in songs:
            row = self._rows.get(song.title)
            if row is None:
                enriched.append(song)
                continue
            try:
                enriched.append(
                    replace(
                        song,
                        popularity=int(row["popularity"]) if row.get("popularity") else None,
                        release_decade=row.get("release_decade") or None,
                        detailed_mood_tags=[t.strip() for t in row.get("detailed_mood_tags", "").split(";") if t.strip()],
                        vocal_style=row.get("vocal_style") or None,
                        instrumentation_notes=row.get("instrumentation_notes") or None,
                    )
                )
            except Exception as exc:
                logger.warning("Malformed attribute row for '%s', leaving defaults: %s", song.title, exc)
                enriched.append(song)
        return enriched
