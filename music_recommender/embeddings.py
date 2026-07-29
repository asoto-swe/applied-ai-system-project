"""Thin wrapper around the Voyage AI embeddings API.

Anthropic has no first-party embeddings endpoint; Voyage AI (an Anthropic
company) is the recommended provider for text embeddings used alongside
Claude. This module isolates that dependency so the rest of the recommender
never talks to Voyage directly.
"""
from __future__ import annotations

import logging
import math
import os
from typing import List, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "voyage-3.5"


class EmbeddingError(RuntimeError):
    """Raised when text embeddings cannot be generated."""


class VoyageEmbeddingClient:
    """Generates semantic embeddings for taste profiles and songs via Voyage AI."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise EmbeddingError(
                "VOYAGE_API_KEY is not set. Get a free key at https://dash.voyageai.com "
                "and set it as an environment variable before running the recommender."
            )

        try:
            import voyageai
        except ImportError as exc:
            raise EmbeddingError(
                "The voyageai package is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def embed_query(self, text: str) -> List[float]:
        """Embed a single taste-profile description."""
        return self.embed_documents([text], input_type="query")[0]

    def embed_documents(self, texts: Sequence[str], input_type: str = "document") -> List[List[float]]:
        """Embed a batch of song descriptions (lyrics/themes/genre/mood)."""
        client = self._get_client()
        try:
            result = client.embed(list(texts), model=self._model, input_type=input_type)
        except Exception as exc:
            logger.error("Voyage AI embedding request failed: %s", exc)
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc
        return result.embeddings


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
