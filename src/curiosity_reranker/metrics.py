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

