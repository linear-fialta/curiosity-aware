from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from curiosity_reranker.baseline import (
    build_user_profile,
    fit_matrix_factorization,
    generate_mf_candidates,
    leave_one_out_split,
    load_movielens_with_optional_metadata,
)
from curiosity_reranker.comparison import (
    direct_vlm_score_rerank_candidates,
    mmr_rerank_candidates,
    serendipity_rerank_candidates,
)
from curiosity_reranker.metrics import summarize_rankings
from curiosity_reranker.rerank import rerank_candidates
from curiosity_reranker.vig_rerank import VIGRerankConfig, vig_rerank_candidates
from curiosity_reranker.visual import attach_visual_interpretations, load_visual_interpretations


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MovieLens VIG-Rerank experiment.")
    parser.add_argument("--movielens-dir", default=str(ROOT / "data" / "raw" / "ml-latest-small"))
    parser.add_argument("--metadata", default=None, help="Optional TMDb metadata CSV.")
    parser.add_argument("--scenes", default=None, help="Optional VLM scene interpretation JSONL.")
    parser.add_argument("--max-users", type=int, default=50)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed" / "movielens_experiment"))
    args = parser.parse_args()

    ratings, movies = load_movielens_with_optional_metadata(args.movielens_dir, args.metadata)
    train, test = leave_one_out_split(ratings)
    model = fit_matrix_factorization(
        train,
        n_factors=args.factors,
        epochs=args.epochs,
    )
    candidates = generate_mf_candidates(
        model,
        train,
        test,
        movies,
        candidates_per_user=args.candidate_k,
        max_users=args.max_users,
    )

    if args.scenes:
        scenes = load_visual_interpretations(Path(args.scenes))
        candidates = attach_visual_interpretations(candidates, scenes)
    else:
        candidates = _attach_zero_visual_features(candidates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "mf_candidates.csv", index=False)

    rankings = {name: {} for name in _variant_names()}
    for user_id, user_candidates in candidates.groupby("user_id"):
        user_profile = build_user_profile(int(user_id), train, movies)
        user_candidates = user_candidates.sort_values("baseline_score", ascending=False).reset_index(drop=True)
        user_candidates = _attach_unexpectedness(user_candidates, user_profile)

        rankings["mf_relevance"][int(user_id)] = user_candidates
        rankings["mmr"][int(user_id)] = mmr_rerank_candidates(user_candidates, top_k=args.top_k)
        rankings["serendipity"][int(user_id)] = serendipity_rerank_candidates(user_candidates, user_profile)
        rankings["linear_gap"][int(user_id)] = rerank_candidates(user_candidates, user_profile)
        rankings["direct_vlm_proxy"][int(user_id)] = direct_vlm_score_rerank_candidates(user_candidates)
        rankings["vig_rerank"][int(user_id)] = vig_rerank_candidates(
            user_candidates,
            user_profile,
            top_k=args.candidate_k,
        )
        rankings["vig_no_visual"][int(user_id)] = vig_rerank_candidates(
            _zero_visual_gap(user_candidates),
            user_profile,
            top_k=args.candidate_k,
        )
        rankings["vig_no_listwise"][int(user_id)] = vig_rerank_candidates(
            user_candidates,
            user_profile,
            top_k=args.candidate_k,
            config=VIGRerankConfig(diversity_penalty=0.0, coverage_bonus=0.0),
        )

    summary_rows = []
    for variant, ranked_by_user in rankings.items():
        row = {"variant": variant, **summarize_rankings(ranked_by_user, args.top_k)}
        summary_rows.append(row)
        _save_rankings(output_dir / f"{variant}_rankings.csv", ranked_by_user)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote experiment outputs to {output_dir}")


def _variant_names() -> list[str]:
    return [
        "mf_relevance",
        "mmr",
        "serendipity",
        "linear_gap",
        "direct_vlm_proxy",
        "vig_rerank",
        "vig_no_visual",
        "vig_no_listwise",
    ]


def _attach_zero_visual_features(candidates: pd.DataFrame) -> pd.DataFrame:
    copied = candidates.copy()
    copied["visual_information_gap_score"] = 0.0
    copied["cross_modal_gap_score"] = 0.0
    copied["visual_reason"] = "no VLM scene interpretation is available"
    return copied


def _zero_visual_gap(candidates: pd.DataFrame) -> pd.DataFrame:
    copied = candidates.copy()
    copied["visual_information_gap_score"] = 0.0
    copied["cross_modal_gap_score"] = 0.0
    copied["visual_reason"] = "visual gap ablated"
    return copied


def _attach_unexpectedness(candidates: pd.DataFrame, user_profile) -> pd.DataFrame:
    copied = candidates.copy()
    preferred = {genre.lower() for genre in user_profile.preferred_genres}
    values = []
    for genres in copied["genres"].tolist():
        current = {genre.lower() for genre in str(genres).split("|") if genre}
        union = current | preferred
        values.append(0.0 if not union else 1.0 - len(current & preferred) / len(union))
    copied["unexpectedness_score"] = values
    return copied


def _save_rankings(path: Path, ranked_by_user: dict[int, pd.DataFrame]) -> None:
    if not ranked_by_user:
        return
    rows = []
    for user_id, ranked in ranked_by_user.items():
        copied = ranked.copy()
        copied["user_id"] = user_id
        copied["rank"] = range(1, len(copied) + 1)
        rows.append(copied)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    main()
