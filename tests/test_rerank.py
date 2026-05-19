from __future__ import annotations

import pandas as pd

from curiosity_reranker.rerank import rerank_candidates
from curiosity_reranker.schema import UserProfile
from curiosity_reranker.vig_rerank import vig_rerank_candidates
from curiosity_reranker.metrics import bootstrap_metric_intervals, summarize_rankings
from curiosity_reranker.visual import attach_visual_interpretations, visual_information_gap_score
from curiosity_reranker.schema import VisualSceneInterpretation
from curiosity_reranker.baseline import fit_matrix_factorization, generate_mf_candidates, leave_one_out_split


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
    assert "visual_scene_available" in enriched.columns
    assert enriched.iloc[0]["visual_scene_available"] == 1
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


def test_matrix_factorization_generates_candidates() -> None:
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 5.0, "timestamp": 1},
            {"userId": 1, "movieId": 11, "rating": 4.0, "timestamp": 2},
            {"userId": 1, "movieId": 12, "rating": 3.0, "timestamp": 3},
            {"userId": 2, "movieId": 10, "rating": 4.0, "timestamp": 1},
            {"userId": 2, "movieId": 12, "rating": 5.0, "timestamp": 2},
            {"userId": 2, "movieId": 13, "rating": 3.0, "timestamp": 3},
        ]
    )
    movies = pd.DataFrame(
        [
            {"movieId": 10, "title": "A", "genres": "Drama", "overview": "A drama"},
            {"movieId": 11, "title": "B", "genres": "Mystery", "overview": "A mystery"},
            {"movieId": 12, "title": "C", "genres": "Sci-Fi", "overview": "A sci-fi story"},
            {"movieId": 13, "title": "D", "genres": "Comedy", "overview": "A comedy"},
        ]
    )
    train, test = leave_one_out_split(ratings, min_interactions=3)
    model = fit_matrix_factorization(train, n_factors=4, epochs=1, seed=7)
    candidates = generate_mf_candidates(model, train, test, movies, candidates_per_user=2)

    assert not candidates.empty
    assert {"user_id", "item_id", "baseline_score", "is_relevant"}.issubset(candidates.columns)


def test_metric_summary_reports_visual_coverage_and_intervals() -> None:
    ranked = pd.DataFrame(
        [
            {
                "item_id": "1",
                "genres": "Drama",
                "baseline_score": 0.9,
                "unexpectedness_score": 0.2,
                "visual_information_gap_score": 0.8,
                "visual_scene_available": 1,
                "is_relevant": 1,
            },
            {
                "item_id": "2",
                "genres": "Comedy",
                "baseline_score": 0.7,
                "unexpectedness_score": 0.6,
                "visual_information_gap_score": 0.0,
                "visual_scene_available": 0,
                "is_relevant": 0,
            },
        ]
    )

    summary = summarize_rankings({1: ranked}, k=2)
    intervals = bootstrap_metric_intervals({1: ranked}, k=2, n_resamples=5, seed=7)

    assert summary["visual_scene_coverage"] == 0.5
    assert "hit_rate_ci_low" in intervals
    assert "visual_scene_coverage_ci_high" in intervals
