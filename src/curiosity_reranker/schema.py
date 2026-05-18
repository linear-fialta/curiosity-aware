from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateItem:
    item_id: str
    title: str
    genres: tuple[str, ...]
    overview: str
    baseline_score: float


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    preferred_genres: tuple[str, ...]

