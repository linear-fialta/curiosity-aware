from __future__ import annotations

import itertools

import pandas as pd


def average_novelty(ranked_items: pd.DataFrame) -> float:
    if ranked_items.empty or "unexpectedness_score" not in ranked_items:
        return 0.0
    return float(ranked_items["unexpectedness_score"].mean())


def intra_list_genre_diversity(ranked_items: pd.DataFrame) -> float:
    if len(ranked_items) < 2:
        return 0.0

    genre_sets = [
        set(str(genres).lower().split("|"))
        for genres in ranked_items["genres"].tolist()
    ]
    distances = []
    for left, right in itertools.combinations(genre_sets, 2):
        union = left | right
        if not union:
            distances.append(0.0)
        else:
            distances.append(1.0 - (len(left & right) / len(union)))
    return float(sum(distances) / len(distances))


def hit_rate_at_k(ranked_items: pd.DataFrame, k: int) -> float:
    if ranked_items.empty or "is_relevant" not in ranked_items:
        return 0.0
    return float(ranked_items.head(k)["is_relevant"].max())


def ndcg_at_k(ranked_items: pd.DataFrame, k: int) -> float:
    if ranked_items.empty or "is_relevant" not in ranked_items:
        return 0.0
    gains = ranked_items.head(k)["is_relevant"].tolist()
    dcg = 0.0
    for idx, gain in enumerate(gains, start=1):
        if gain:
            dcg += 1.0 / _log2(idx + 1)
    ideal = 1.0
    return float(dcg / ideal)


def summarize_rankings(ranked_by_user: dict[int, pd.DataFrame], k: int) -> dict[str, float]:
    if not ranked_by_user:
        return {
            "hit_rate": 0.0,
            "ndcg": 0.0,
            "avg_baseline_relevance": 0.0,
            "avg_visual_gap": 0.0,
            "avg_novelty": 0.0,
            "genre_diversity": 0.0,
        }

    rows = []
    for ranked in ranked_by_user.values():
        top = ranked.head(k)
        rows.append(
            {
                "hit_rate": hit_rate_at_k(ranked, k),
                "ndcg": ndcg_at_k(ranked, k),
                "avg_baseline_relevance": float(top["baseline_score"].mean()),
                "avg_visual_gap": float(top.get("visual_information_gap_score", 0).mean()),
                "avg_novelty": average_novelty(top),
                "genre_diversity": intra_list_genre_diversity(top),
            }
        )
    return {key: float(pd.DataFrame(rows)[key].mean()) for key in rows[0]}


def _log2(value: int) -> float:
    import math

    return math.log(value, 2)
