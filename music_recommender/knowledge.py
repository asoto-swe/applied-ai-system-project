"""A second retrieval data source: short genre-background notes.

This is the project's RAG Enhancement — retrieval no longer draws on a
single source (the song catalog) but also on a curated corpus of
genre-background context, so a recommendation's explanation can be grounded
in *why a genre tends to sound and feel the way it does*, not just which
tags happened to match.
"""
from __future__ import annotations

import csv
import logging
import pathlib
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
GENRE_NOTES_PATH = _PACKAGE_DIR / "data" / "genre_notes.csv"


class GenreKnowledgeBase:
    """Looks up a short background note for a genre, if one exists.

    A missing or unreadable notes file degrades gracefully to an empty
    knowledge base (lookups return None) rather than raising — consistent
    with every other AI dependency in this project."""

    def __init__(self) -> None:
        self._notes: Dict[str, str] = self._load_notes()

    def _load_notes(self) -> Dict[str, str]:
        try:
            with open(GENRE_NOTES_PATH, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return {row["genre"].strip().lower(): row["note"].strip() for row in rows}
        except Exception as exc:
            logger.warning("Genre knowledge base unavailable, continuing without genre context: %s", exc)
            return {}

    def lookup(self, genre: str) -> Optional[str]:
        return self._notes.get(genre.strip().lower())
