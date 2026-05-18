from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from curiosity_reranker.features import (
    genre_overlap,
    moderate_unexpectedness_score,
    normalize_score,
    text_information_gap_score,
)
from curiosity_reranker.schema import CandidateItem, UserProfile


@dataclass(frozen=True)
class RerankWeights:
    relevance: float = 0.55
    visual_gap: float = 0.20
    cross_modal_gap: float = 0.05
    text_gap: float = 0.10
    moderate_unexpectedness: float = 0.10


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
        visual_gap = float(row.get("visual_information_gap_score", 0.0))
        cross_modal_gap = float(row.get("cross_modal_gap_score", 0.0))
        text_gap = text_information_gap_score(item)
        unexpectedness = 1.0 - genre_overlap(item.genres, user.preferred_genres)
        moderate_unexpectedness = moderate_unexpectedness_score(item, user)
        rerank_score = (
            weights.relevance * normalize_score(item.baseline_score)
            + weights.visual_gap * visual_gap
            + weights.cross_modal_gap * cross_modal_gap
            + weights.text_gap * text_gap
            + weights.moderate_unexpectedness * moderate_unexpectedness
        )
        records.append(
            {
                **row.to_dict(),
                "text_information_gap_score": text_gap,
                "cross_modal_gap_score": cross_modal_gap,
                "unexpectedness_score": unexpectedness,
                "moderate_unexpectedness_score": moderate_unexpectedness,
                "rerank_score": rerank_score,
                "explanation": _build_explanation(
                    item,
                    visual_gap,
                    cross_modal_gap,
                    text_gap,
                    moderate_unexpectedness,
                    row,
                ),
            }
        )

    return pd.DataFrame(records).sort_values("rerank_score", ascending=False).reset_index(drop=True)


def _build_explanation(
    item: CandidateItem,
    visual_gap: float,
    cross_modal_gap: float,
    text_gap: float,
    moderate_unexpectedness: float,
    row: pd.Series,
) -> str:
    if visual_gap > 0.65:
        angle = str(row.get("visual_reason", "its image creates a visual information gap"))
    elif text_gap > 0.55:
        angle = "its description leaves unresolved narrative or semantic information"
    elif cross_modal_gap > 0.75:
        angle = "its image and metadata create a cross-modal information gap"
    elif moderate_unexpectedness > 0.75:
        angle = "it is close enough to the user's profile while adding meaningful distance"
    else:
        angle = "it preserves relevance while adding some exploration value"
    ending = "" if angle.endswith((".", "?", "!")) else "."
    return f"{item.title} is reranked higher because {angle}{ending}"
