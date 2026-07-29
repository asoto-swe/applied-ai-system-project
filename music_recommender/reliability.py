"""Reliability and consistency checks for the AI-driven recommendation pipeline.

These functions probe *behavior*, not code correctness: do repeated calls
with identical input produce identical rankings, and does the pipeline
degrade gracefully instead of crashing on adversarial or edge-case taste
profiles. Used by scripts/evaluate_reliability.py to produce a
human-readable report, and by tests/test_reliability.py to keep these
checks in the automated test suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .data import Song, TasteProfile
from .recommender import MusicRecommender


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_consistency(
    recommender: MusicRecommender,
    profile: TasteProfile,
    songs: List[Song],
    runs: int = 3,
    limit: int = 3,
) -> CheckResult:
    """Runs the same query multiple times and checks the ranking (titles +
    scores) is identical every time. Catches nondeterminism in retrieval or
    scoring — the parts of the pipeline that should be fully deterministic.
    Deliberately does NOT compare explanation text, since Claude-generated
    wording naturally varies run to run even when grounded in the same
    evidence; see check_explanation_groundedness for that instead."""
    runs_output = []
    for _ in range(runs):
        recs = recommender.recommend(profile, songs, limit=limit)
        runs_output.append([(r.title, r.score) for r in recs])

    consistent = all(result == runs_output[0] for result in runs_output)
    detail = (
        f"{runs} runs, identical rankings: {runs_output[0]}"
        if consistent
        else f"{runs} runs produced different rankings: {runs_output}"
    )
    return CheckResult(name=f"Consistency for '{profile.name}'", passed=consistent, detail=detail)


def check_graceful_degradation(
    recommender: MusicRecommender,
    profile: TasteProfile,
    songs: List[Song],
    limit: int = 3,
) -> CheckResult:
    """Runs a (possibly adversarial/edge-case) profile through the pipeline
    and checks it doesn't crash, respects the limit, returns scores sorted
    descending, and never returns a non-positive score — i.e. it degrades
    gracefully instead of failing outright."""
    try:
        recs = recommender.recommend(profile, songs, limit=limit)
    except Exception as exc:
        return CheckResult(
            name=f"Graceful degradation for '{profile.name}'",
            passed=False,
            detail=f"Raised unexpectedly: {exc}",
        )

    within_limit = len(recs) <= limit
    sorted_desc = all(recs[i].score >= recs[i + 1].score for i in range(len(recs) - 1))
    all_positive = all(r.score > 0 for r in recs)

    passed = within_limit and sorted_desc and all_positive
    detail = (
        f"{len(recs)} recommendations, within_limit={within_limit}, "
        f"sorted_desc={sorted_desc}, all_positive={all_positive}"
    )
    return CheckResult(name=f"Graceful degradation for '{profile.name}'", passed=passed, detail=detail)


def check_empty_profile_guardrail(recommender: MusicRecommender, songs: List[Song]) -> CheckResult:
    """Confirms the empty-taste-profile guardrail still raises ValueError."""
    empty_profile = TasteProfile(
        name="Nobody", preferred_genres=[], preferred_moods=[], preferred_themes=[], favorite_artists=[]
    )
    try:
        recommender.recommend(empty_profile, songs, limit=3)
    except ValueError:
        return CheckResult(name="Empty-profile guardrail", passed=True, detail="Raised ValueError as expected")
    else:
        return CheckResult(
            name="Empty-profile guardrail", passed=False, detail="Did not raise ValueError for an empty profile"
        )


def check_explanation_groundedness(
    recommender: MusicRecommender, profile: TasteProfile, songs: List[Song]
) -> Optional[CheckResult]:
    """Basic groundedness heuristics on the top recommendation's explanation:
    non-empty, addresses the listener by name, and isn't wildly long. This
    is a light heuristic, not a full factuality check — it catches an empty
    or clearly-broken generation, not subtle hallucination. Returns None
    (skipped) if the profile produces no recommendations to check."""
    recs = recommender.recommend(profile, songs, limit=1)
    if not recs:
        return None

    explanation = recs[0].explanation
    mentions_name = profile.name in explanation
    reasonable_length = 0 < len(explanation) <= 800

    passed = mentions_name and reasonable_length
    detail = f"mentions_name={mentions_name}, length={len(explanation)}"
    return CheckResult(name=f"Explanation groundedness for '{profile.name}'", passed=passed, detail=detail)
