from __future__ import annotations

import itertools

import numpy as np
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
    rows = per_user_metric_rows(ranked_by_user, k)
    if rows.empty:
        return _empty_metric_summary()

    return {key: float(rows[key].mean()) for key in _metric_names()}


def bootstrap_metric_intervals(
    ranked_by_user: dict[int, pd.DataFrame],
    k: int,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rows = per_user_metric_rows(ranked_by_user, k)
    if rows.empty:
        return {
            f"{metric}_{bound}": 0.0
            for metric in _metric_names()
            for bound in ("ci_low", "ci_high")
        }

    n_resamples = max(1, n_resamples)
    rng = np.random.default_rng(seed)
    values = rows[_metric_names()].to_numpy(dtype=float)
    intervals: dict[str, float] = {}
    for metric_idx, metric in enumerate(_metric_names()):
        boot_means = []
        for _ in range(n_resamples):
            sample_idx = rng.integers(0, len(values), size=len(values))
            boot_means.append(float(values[sample_idx, metric_idx].mean()))
        intervals[f"{metric}_ci_low"] = float(np.quantile(boot_means, 0.025))
        intervals[f"{metric}_ci_high"] = float(np.quantile(boot_means, 0.975))
    return intervals


def per_user_metric_rows(ranked_by_user: dict[int, pd.DataFrame], k: int) -> pd.DataFrame:
    rows = []
    for user_id, ranked in ranked_by_user.items():
        top = ranked.head(k)
        rows.append(
            {
                "user_id": user_id,
                "hit_rate": hit_rate_at_k(ranked, k),
                "ndcg": ndcg_at_k(ranked, k),
                "avg_baseline_relevance": float(top["baseline_score"].mean()),
                "avg_visual_gap": _column_mean(top, "visual_information_gap_score"),
                "avg_novelty": average_novelty(top),
                "genre_diversity": intra_list_genre_diversity(top),
                "visual_scene_coverage": _column_mean(top, "visual_scene_available"),
            }
        )
    return pd.DataFrame(rows)


def _metric_names() -> list[str]:
    return [
        "hit_rate",
        "ndcg",
        "avg_baseline_relevance",
        "avg_visual_gap",
        "avg_novelty",
        "genre_diversity",
        "visual_scene_coverage",
    ]


def _empty_metric_summary() -> dict[str, float]:
    return {metric: 0.0 for metric in _metric_names()}


def _column_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    return float(frame[column].mean())


def _log2(value: int) -> float:
    import math

    return math.log(value, 2)
