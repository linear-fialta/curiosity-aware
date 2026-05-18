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


@dataclass(frozen=True)
class VisualSceneInterpretation:
    item_id: str
    main_objects: tuple[str, ...]
    setting: str
    visible_action: str
    occluded_or_missing_information: tuple[str, ...]
    object_context_incongruity: str
    genre_ambiguity: tuple[str, ...]
    emotional_tension: tuple[str, ...]
    implied_question: str
