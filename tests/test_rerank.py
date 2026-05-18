from __future__ import annotations

import pandas as pd

from curiosity_reranker.rerank import rerank_candidates
from curiosity_reranker.schema import UserProfile
from curiosity_reranker.vig_rerank import vig_rerank_candidates
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


def test_vig_rerank_adds_sweet_spot_and_listwise_scores() -> None:
    candidates = pd.DataFrame(
        [
            {
                "item_id": "1",
                "title": "Safe Drama",
                "genres": "Drama",
                "overview": "A familiar drama.",
                "baseline_score": 0.9,
                "visual_information_gap_score": 0.2,
                "cross_modal_gap_score": 0.2,
                "visual_reason": "the image is familiar",
            },
            {
                "item_id": "2",
                "title": "Taste Adjacent Mystery",
                "genres": "Drama|Mystery",
                "overview": "A hidden event is discovered.",
                "baseline_score": 0.82,
                "visual_information_gap_score": 0.9,
                "cross_modal_gap_score": 0.8,
                "visual_reason": "the image implies an unresolved question",
            },
            {
                "item_id": "3",
                "title": "Too Distant Fantasy",
                "genres": "Fantasy|Adventure",
                "overview": "A magical quest unfolds.",
                "baseline_score": 0.8,
                "visual_information_gap_score": 0.9,
                "cross_modal_gap_score": 0.8,
                "visual_reason": "the image implies an unresolved question",
            },
        ]
    )
    user = UserProfile(user_id="u1", preferred_genres=("Drama",))
    ranked = vig_rerank_candidates(candidates, user, top_k=2)

    assert list(ranked["vig_rank"]) == [1, 2]
    assert "sweet_spot_score" in ranked.columns
    assert "vig_item_score" in ranked.columns
    assert "vig_listwise_score" in ranked.columns
    assert ranked.iloc[0]["title"] == "Taste Adjacent Mystery"
