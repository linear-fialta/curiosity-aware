from __future__ import annotations

import pandas as pd

from curiosity_reranker.features import genre_overlap
from curiosity_reranker.schema import UserProfile


def mmr_rerank_candidates(
    candidates: pd.DataFrame,
    relevance_weight: float = 0.75,
    top_k: int | None = None,
) -> pd.DataFrame:
    if top_k is None:
        top_k = len(candidates)
    selected: list[dict] = []
    remaining = candidates.to_dict("records")
    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")
        for idx, row in enumerate(remaining):
            redundancy = _max_genre_similarity(row, selected)
            score = relevance_weight * float(row["baseline_score"]) - (1 - relevance_weight) * redundancy
            if score > best_score:
                best_idx = idx
                best_score = score
        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = best_score
        selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True)


def serendipity_rerank_candidates(
    candidates: pd.DataFrame,
    user: UserProfile,
    relevance_weight: float = 0.65,
) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        genres = tuple(str(row["genres"]).split("|"))
        unexpectedness = 1.0 - genre_overlap(genres, user.preferred_genres)
        score = relevance_weight * float(row["baseline_score"]) + (1 - relevance_weight) * unexpectedness
        rows.append({**row.to_dict(), "unexpectedness_score": unexpectedness, "serendipity_score": score})
    return pd.DataFrame(rows).sort_values("serendipity_score", ascending=False).reset_index(drop=True)


def direct_vlm_score_rerank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    copied = candidates.copy()
    copied["direct_vlm_proxy_score"] = (
        0.7 * copied.get("visual_information_gap_score", 0.0)
        + 0.3 * copied.get("cross_modal_gap_score", 0.0)
    )
    return copied.sort_values(
        ["direct_vlm_proxy_score", "baseline_score"],
        ascending=False,
    ).reset_index(drop=True)


def _max_genre_similarity(row: dict, selected: list[dict]) -> float:
    if not selected:
        return 0.0
    current = _genre_set(str(row["genres"]))
    return max(_jaccard(current, _genre_set(str(item["genres"]))) for item in selected)


def _genre_set(genres: str) -> set[str]:
    return {genre.strip().lower() for genre in genres.split("|") if genre.strip()}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
