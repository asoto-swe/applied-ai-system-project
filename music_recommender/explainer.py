"""LLM-generated, evidence-grounded explanations for song recommendations.

This is the "generation" half of the RAG pipeline: retrieval (embeddings.py /
retriever.py) finds evidence, and this module has Claude turn that evidence
into a personalized explanation rather than filling in a hand-written template.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .data import Song, TasteProfile

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"


class ExplanationError(RuntimeError):
    """Raised when an AI-generated explanation cannot be produced."""


class ClaudeExplainer:
    """Asks Claude to explain one recommendation, grounded in retrieved evidence."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise ExplanationError(
                "ANTHROPIC_API_KEY is not set. Set it as an environment variable to enable "
                "AI-generated explanations."
            )

        try:
            import anthropic
        except ImportError as exc:
            raise ExplanationError(
                "The anthropic package is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def explain(self, profile: TasteProfile, song: Song, evidence: dict) -> str:
        client = self._get_client()
        prompt = self._build_prompt(profile, song, evidence)

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=200,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logger.error("Claude explanation request failed: %s", exc)
            raise ExplanationError(f"Failed to generate explanation: {exc}") from exc

        if response.stop_reason == "refusal":
            raise ExplanationError("Claude declined to generate this explanation.")

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise ExplanationError("Claude returned an empty explanation.")
        return text

    @staticmethod
    def _build_prompt(profile: TasteProfile, song: Song, evidence: dict) -> str:
        matches = []
        if evidence.get("matched_genres"):
            matches.append(f"genre match: {', '.join(evidence['matched_genres'])}")
        if evidence.get("matched_moods"):
            matches.append(f"mood match: {', '.join(evidence['matched_moods'])}")
        if evidence.get("matched_themes"):
            matches.append(f"theme match: {', '.join(evidence['matched_themes'])}")
        if evidence.get("matched_artists"):
            matches.append(f"favorite artist match: {', '.join(evidence['matched_artists'])}")
        semantic_score = evidence.get("semantic_score", 0.0)
        audio_fit_score = evidence.get("audio_fit_score", 0.0)
        genre_context = evidence.get("genre_context")
        genre_context_line = (
            f"- Background on {song.genre} as a genre: {genre_context}\n" if genre_context else ""
        )

        return (
            f"Listener name: {profile.name}\n"
            f"Listener's stated taste: genres={profile.preferred_genres}, "
            f"moods={profile.preferred_moods}, themes={profile.preferred_themes}, "
            f"favorite artists={profile.favorite_artists}\n\n"
            f'Recommended song: "{song.title}" by {song.artist}\n'
            f"Song genre: {song.genre}, mood: {song.mood}, themes: {song.themes}\n"
            f'Lyrics excerpt: "{song.lyrics_excerpt}"\n\n'
            "Retrieved evidence for why this song was recommended:\n"
            f"- {'; '.join(matches) if matches else 'no exact tag matches'}\n"
            f"- Semantic similarity between the listener's taste and this song's meaning: "
            f"{semantic_score:.2f} on a 0 to 1 scale\n"
            f"- Predicted audio-feature fit (energy/positivity/acousticness, from a model "
            f"trained on genre and mood): {audio_fit_score:.2f} on a 0 to 1 scale\n"
            f"{genre_context_line}\n"
            "Write a 1-2 sentence, warm, specific explanation of why this song fits the "
            "listener, addressed to them by name. Ground every claim in the evidence above — "
            "do not invent facts about the song or the listener that aren't given. If the "
            "semantic similarity or audio-feature fit is the main driver (few or no tag "
            "matches), explain the connection in terms of feeling, meaning, or overall sound "
            "rather than claiming a tag match that didn't happen. Only mention audio-feature "
            "fit if it's reasonably high (0.7+); otherwise ignore it. You may briefly weave in "
            "the genre background if it's given and genuinely adds something, but keep the "
            "focus on the listener, not a genre lecture."
        )
