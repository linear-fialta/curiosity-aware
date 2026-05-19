from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from curiosity_reranker.features import genre_overlap, normalize_score, text_information_gap_score
from curiosity_reranker.rerank import _item_from_row
from curiosity_reranker.schema import CandidateItem, UserProfile


@dataclass(frozen=True)
class VIGRerankConfig:
    relevance_power: float = 1.0
    visual_gap_power: float = 1.0
    text_gap_power: float = 0.3
    cross_modal_gap_power: float = 0.3
    target_distance: float = 0.35
    distance_sigma: float = 0.35
    diversity_penalty: float = 0.08
    coverage_bonus: float = 0.03
    epsilon: float = 1e-6


def vig_rerank_candidates(
    candidates: pd.DataFrame,
    user: UserProfile,
    top_k: int | None = None,
    config: VIGRerankConfig | None = None,
) -> pd.DataFrame:
    """Listwise Visual Information Gap reranking.

    The item score rewards relevance, visual information gap, text/cross-modal
    gap, and the WITS-inspired moderate-distance sweet spot. The listwise step
    then penalizes redundancy against already selected items.
    """
    config = config or VIGRerankConfig()
    scored = _score_candidates(candidates, user, config)
    if top_k is None:
        top_k = len(scored)

    selected: list[dict] = []
    remaining = scored.to_dict("records")
    covered_genres: set[str] = set()

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = -math.inf
        for idx, row in enumerate(remaining):
            redundancy = _max_genre_similarity(row, selected)
            coverage = _new_genre_coverage(row, covered_genres)
            listwise_score = (
                float(row["vig_item_score"])
                - config.diversity_penalty * redundancy
                + config.coverage_bonus * coverage
            )
            if listwise_score > best_score:
                best_idx = idx
                best_score = listwise_score

        chosen = remaining.pop(best_idx)
        chosen["vig_listwise_score"] = best_score
        chosen["vig_rank"] = len(selected) + 1
        selected.append(chosen)
        covered_genres.update(_genre_set(chosen["genres"]))

    return pd.DataFrame(selected).reset_index(drop=True)


def _score_candidates(
    candidates: pd.DataFrame,
    user: UserProfile,
    config: VIGRerankConfig,
) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        item = _item_from_row(row)
        relevance = normalize_score(item.baseline_score)
        visual_gap = normalize_score(float(row.get("visual_information_gap_score", 0.0)))
        text_gap = text_information_gap_score(item)
        cross_modal_gap = normalize_score(float(row.get("cross_modal_gap_score", 0.0)))
        taste_distance = 1.0 - genre_overlap(item.genres, user.preferred_genres)
        sweet_spot = _sweet_spot_score(
            taste_distance,
            target=config.target_distance,
            sigma=config.distance_sigma,
        )

        vig_item_score = (
            (relevance + config.epsilon) ** config.relevance_power
            * (visual_gap + config.epsilon) ** config.visual_gap_power
            * (text_gap + config.epsilon) ** config.text_gap_power
            * (cross_modal_gap + config.epsilon) ** config.cross_modal_gap_power
            * (sweet_spot + config.epsilon)
        )
        rows.append(
            {
                **row.to_dict(),
                "relevance_score": relevance,
                "text_information_gap_score": text_gap,
                "taste_distance_score": taste_distance,
                "sweet_spot_score": sweet_spot,
                "vig_item_score": vig_item_score,
                "vig_explanation": _build_vig_explanation(item, visual_gap, sweet_spot, row),
            }
        )

    return pd.DataFrame(rows).sort_values("vig_item_score", ascending=False).reset_index(drop=True)


def _sweet_spot_score(distance: float, target: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("distance_sigma must be positive.")
    return normalize_score(math.exp(-((distance - target) ** 2) / (2 * sigma**2)))


def _max_genre_similarity(row: dict, selected: list[dict]) -> float:
    if not selected:
        return 0.0
    current = _genre_set(str(row["genres"]))
    return max(_jaccard(current, _genre_set(str(item["genres"]))) for item in selected)


def _new_genre_coverage(row: dict, covered_genres: set[str]) -> float:
    genres = _genre_set(str(row["genres"]))
    if not genres:
        return 0.0
    return len(genres - covered_genres) / len(genres)


def _genre_set(genres: str) -> set[str]:
    return {genre.strip().lower() for genre in genres.split("|") if genre.strip()}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _build_vig_explanation(
    item: CandidateItem,
    visual_gap: float,
    sweet_spot: float,
    row: pd.Series,
) -> str:
    if visual_gap >= 0.7 and sweet_spot >= 0.7:
        reason = str(row.get("visual_reason", "its image creates a visual information gap"))
        return _sentence(f"{item.title} sits near the relevance-novelty sweet spot, and {reason}")
    if visual_gap >= 0.7:
        return f"{item.title} has a strong visual information gap but weaker taste fit."
    if sweet_spot >= 0.7:
        return f"{item.title} is taste-adjacent without being redundant."
    return f"{item.title} is retained mainly for baseline relevance."


def _sentence(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else f"{text}."
