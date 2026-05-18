from __future__ import annotations

import math
import re
from collections.abc import Iterable

from curiosity_reranker.schema import CandidateItem, UserProfile


QUESTION_WORDS = {
    "why",
    "how",
    "secret",
    "mystery",
    "unknown",
    "hidden",
    "discover",
    "unexpected",
    "reveals",
    "uncover",
}


def normalize_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def genre_overlap(item_genres: Iterable[str], preferred_genres: Iterable[str]) -> float:
    item_set = {genre.lower() for genre in item_genres}
    preferred_set = {genre.lower() for genre in preferred_genres}
    if not item_set or not preferred_set:
        return 0.0
    return len(item_set & preferred_set) / len(item_set | preferred_set)


def novelty_score(item: CandidateItem, user: UserProfile) -> float:
    return 1.0 - genre_overlap(item.genres, user.preferred_genres)


def narrative_incompleteness_score(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return 0.0
    signal_count = sum(token in QUESTION_WORDS for token in tokens)
    length_bonus = 1.0 / (1.0 + math.exp(-(len(tokens) - 25) / 20))
    return normalize_score((signal_count / 4.0) + (0.35 * length_bonus))


def semantic_gap_score(item: CandidateItem) -> float:
    title_tokens = set(re.findall(r"[a-zA-Z]+", item.title.lower()))
    overview_tokens = set(re.findall(r"[a-zA-Z]+", item.overview.lower()))
    genre_tokens = {genre.lower() for genre in item.genres}
    if not overview_tokens:
        return 0.0

    explicit_overlap = len((title_tokens | genre_tokens) & overview_tokens)
    gap = 1.0 - (explicit_overlap / max(1, len(title_tokens | genre_tokens)))
    return normalize_score(gap)


def curiosity_score(item: CandidateItem, user: UserProfile) -> float:
    novelty = novelty_score(item, user)
    narrative_gap = narrative_incompleteness_score(item.overview)
    semantic_gap = semantic_gap_score(item)
    return normalize_score((0.4 * novelty) + (0.35 * narrative_gap) + (0.25 * semantic_gap))

