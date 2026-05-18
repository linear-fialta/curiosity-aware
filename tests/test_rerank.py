from __future__ import annotations

import pandas as pd

from curiosity_reranker.rerank import rerank_candidates
from curiosity_reranker.schema import UserProfile
from curiosity_reranker.visual import attach_visual_interpretations, visual_information_gap_score
from curiosity_reranker.schema import VisualSceneInterpretation


def test_rerank_adds_curiosity_columns() -> None:
    candidates = pd.DataFrame(
        [
            {
                "item_id": "1",
                "title": "Known Drama",
                "genres": "Drama",
                "overview": "A familiar workplace drama.",
                "baseline_score": 0.9,
            },
            {
                "item_id": "2",
                "title": "Hidden Signal",
                "genres": "Sci-Fi|Mystery",
                "overview": "A scientist uncovers a hidden mystery and must discover why the signal appeared.",
                "baseline_score": 0.7,
            },
        ]
    )
    user = UserProfile(user_id="u1", preferred_genres=("Drama",))
    ranked = rerank_candidates(candidates, user)

    assert "text_information_gap_score" in ranked.columns
    assert "cross_modal_gap_score" in ranked.columns
    assert "unexpectedness_score" in ranked.columns
    assert "moderate_unexpectedness_score" in ranked.columns
    assert "rerank_score" in ranked.columns
    assert ranked["rerank_score"].is_monotonic_decreasing


def test_visual_information_gap_uses_scene_structure() -> None:
    scene = VisualSceneInterpretation(
        item_id="1",
        main_objects=("unknown object", "person"),
        setting="foggy field",
        visible_action="person approaches object",
        occluded_or_missing_information=("what the object is", "why it is there"),
        object_context_incongruity="ordinary person encounters unexplained object",
        genre_ambiguity=("mystery", "science fiction"),
        emotional_tension=("uncertainty", "risk"),
        implied_question="What happens when the person reaches the object?",
    )

    assert visual_information_gap_score(scene) > 0.6


def test_attach_visual_interpretations_adds_gap_columns() -> None:
    candidates = pd.DataFrame(
        [
            {
                "item_id": "1",
                "title": "Mystery Item",
                "genres": "Mystery",
                "overview": "A person discovers something hidden.",
                "baseline_score": 0.8,
            }
        ]
    )
    scene = VisualSceneInterpretation(
        item_id="1",
        main_objects=("person",),
        setting="field",
        visible_action="person waits",
        occluded_or_missing_information=("what they are waiting for",),
        object_context_incongruity="",
        genre_ambiguity=("mystery",),
        emotional_tension=("anticipation",),
        implied_question="What are they waiting for?",
    )

    enriched = attach_visual_interpretations(candidates, {"1": scene})
    assert "visual_information_gap_score" in enriched.columns
    assert "cross_modal_gap_score" in enriched.columns
    assert "visual_reason" in enriched.columns
