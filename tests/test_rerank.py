from __future__ import annotations

import pandas as pd

from curiosity_reranker.rerank import rerank_candidates
from curiosity_reranker.schema import UserProfile


def test_rerank_adds_curiosity_columns() -> None:
    candidates = pd.DataFrame(
        [
            {
                "item_id": "1",
                "title": "Known Drama",
                "genres": "Drama",
                "overview": "A familiar workplace drama.",
                "baseline_score": 0.9,
            },
            {
                "item_id": "2",
                "title": "Hidden Signal",
                "genres": "Sci-Fi|Mystery",
                "overview": "A scientist uncovers a hidden mystery and must discover why the signal appeared.",
                "baseline_score": 0.7,
            },
        ]
    )
    user = UserProfile(user_id="u1", preferred_genres=("Drama",))
    ranked = rerank_candidates(candidates, user)

    assert "curiosity_score" in ranked.columns
    assert "unexpectedness_score" in ranked.columns
    assert "rerank_score" in ranked.columns
    assert ranked["rerank_score"].is_monotonic_decreasing
