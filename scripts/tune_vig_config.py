from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from curiosity_reranker.baseline import (
    build_user_profile,
    fit_matrix_factorization,
    generate_mf_candidates,
    leave_one_out_split,
    load_movielens_with_optional_metadata,
)
from curiosity_reranker.features import genre_unexpectedness
from curiosity_reranker.metrics import summarize_rankings
from curiosity_reranker.vig_rerank import VIGRerankConfig, vig_rerank_candidates
from curiosity_reranker.visual import attach_visual_interpretations, load_visual_interpretations


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune VIG-Rerank parameters on existing VLM features.")
    parser.add_argument("--movielens-dir", default=str(ROOT / "data" / "raw" / "ml-latest-small"))
    parser.add_argument("--metadata", default=str(ROOT / "data" / "external" / "tmdb_movies.csv"))
    parser.add_argument("--scenes", default=str(ROOT / "data" / "external" / "vlm_scene_interpretations.jsonl"))
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "vig_tuning_results.csv"))
    args = parser.parse_args()

    ratings, movies = load_movielens_with_optional_metadata(args.movielens_dir, args.metadata)
    train, test = leave_one_out_split(ratings)
    model = fit_matrix_factorization(train, n_factors=args.factors, epochs=args.epochs)
    candidates = generate_mf_candidates(
        model,
        train,
        test,
        movies,
        candidates_per_user=args.candidate_k,
        max_users=args.max_users,
    )
    scenes = load_visual_interpretations(Path(args.scenes))
    candidates = attach_visual_interpretations(candidates, scenes)

    user_profiles = {
        int(user_id): build_user_profile(int(user_id), train, movies)
        for user_id in candidates["user_id"].unique()
    }
    grouped_candidates = {
        int(user_id): user_candidates.sort_values("baseline_score", ascending=False).reset_index(drop=True)
        for user_id, user_candidates in candidates.groupby("user_id")
    }

    configs = _config_grid()
    rows = []
    for idx, config in enumerate(configs, start=1):
        ranked_by_user = {}
        for user_id, user_candidates in grouped_candidates.items():
            user_candidates = _attach_unexpectedness(user_candidates, user_profiles[user_id])
            ranked_by_user[user_id] = vig_rerank_candidates(
                user_candidates,
                user_profiles[user_id],
                top_k=args.candidate_k,
                config=config,
            )
        metrics = summarize_rankings(ranked_by_user, args.top_k)
        rows.append(
            {
                "config_id": idx,
                **config.__dict__,
                **metrics,
                "selection_score": _selection_score(metrics),
            }
        )
        print(
            f"[{idx:03d}/{len(configs)}] "
            f"hit={metrics['hit_rate']:.3f} ndcg={metrics['ndcg']:.3f} "
            f"div={metrics['genre_diversity']:.3f} visual={metrics['avg_visual_gap']:.3f}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "hit_rate", "ndcg"],
        ascending=False,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print("\nTop configurations")
    display_cols = [
        "config_id",
        "hit_rate",
        "ndcg",
        "avg_baseline_relevance",
        "avg_visual_gap",
        "avg_novelty",
        "genre_diversity",
        "selection_score",
        "relevance_power",
        "visual_gap_power",
        "target_distance",
        "distance_sigma",
        "diversity_penalty",
        "coverage_bonus",
    ]
    print(results[display_cols].head(12).to_string(index=False))
    print(f"Wrote {output_path}")


def _config_grid() -> list[VIGRerankConfig]:
    configs = []
    for (
        relevance_power,
        visual_gap_power,
        target_distance,
        distance_sigma,
        diversity_penalty,
        coverage_bonus,
    ) in itertools.product(
        [1.0],
        [0.4, 0.6, 0.8, 1.0],
        [0.35, 0.5],
        [0.35],
        [0.0, 0.08, 0.16],
        [0.0, 0.03],
    ):
        configs.append(
            VIGRerankConfig(
                relevance_power=relevance_power,
                visual_gap_power=visual_gap_power,
                text_gap_power=0.3,
                cross_modal_gap_power=0.3,
                target_distance=target_distance,
                distance_sigma=distance_sigma,
                diversity_penalty=diversity_penalty,
                coverage_bonus=coverage_bonus,
            )
        )
    return configs


def _selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["ndcg"]
        + 0.5 * metrics["hit_rate"]
        + 0.04 * metrics["genre_diversity"]
        - 0.02 * max(0.0, 0.84 - metrics["avg_baseline_relevance"])
    )


def _attach_unexpectedness(candidates: pd.DataFrame, user_profile) -> pd.DataFrame:
    copied = candidates.copy()
    values = []
    for genres in copied["genres"].tolist():
        values.append(genre_unexpectedness(str(genres).split("|"), user_profile.preferred_genres))
    copied["unexpectedness_score"] = values
    return copied


if __name__ == "__main__":
    main()
