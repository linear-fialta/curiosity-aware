from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from curiosity_reranker.features import curiosity_score, genre_overlap, normalize_score
from curiosity_reranker.schema import CandidateItem, UserProfile


@dataclass(frozen=True)
class RerankWeights:
    relevance: float = 0.60
    curiosity: float = 0.25
    unexpectedness: float = 0.15


def _item_from_row(row: pd.Series) -> CandidateItem:
    genres = tuple(str(row["genres"]).split("|")) if pd.notna(row["genres"]) else ()
    return CandidateItem(
        item_id=str(row["item_id"]),
        title=str(row["title"]),
        genres=genres,
        overview=str(row["overview"]),
        baseline_score=float(row["baseline_score"]),
    )


def rerank_candidates(
    candidates: pd.DataFrame,
    user: UserProfile,
    weights: RerankWeights | None = None,
) -> pd.DataFrame:
    weights = weights or RerankWeights()
    records = []

    for _, row in candidates.iterrows():
        item = _item_from_row(row)
        curiosity = curiosity_score(item, user)
        unexpectedness = 1.0 - genre_overlap(item.genres, user.preferred_genres)
        rerank_score = (
            weights.relevance * normalize_score(item.baseline_score)
            + weights.curiosity * curiosity
            + weights.unexpectedness * unexpectedness
        )
        records.append(
            {
                **row.to_dict(),
                "curiosity_score": curiosity,
                "unexpectedness_score": unexpectedness,
                "rerank_score": rerank_score,
                "explanation": _build_explanation(item, curiosity, unexpectedness),
            }
        )

    return pd.DataFrame(records).sort_values("rerank_score", ascending=False).reset_index(drop=True)


def _build_explanation(item: CandidateItem, curiosity: float, unexpectedness: float) -> str:
    if unexpectedness > 0.7:
        angle = "broadens the user's usual taste profile"
    elif curiosity > 0.55:
        angle = "contains unresolved narrative cues"
    else:
        angle = "preserves relevance while adding exploration value"
    return f"{item.title} is recommended because it {angle}."

