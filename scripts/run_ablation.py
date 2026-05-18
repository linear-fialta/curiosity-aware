from __future__ import annotations

from pathlib import Path

import pandas as pd

from curiosity_reranker.metrics import average_novelty, intra_list_genre_diversity
from curiosity_reranker.rerank import rerank_candidates
from curiosity_reranker.schema import UserProfile
from curiosity_reranker.vig_rerank import VIGRerankConfig, vig_rerank_candidates
from curiosity_reranker.visual import attach_visual_interpretations, load_visual_interpretations


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    candidates = pd.read_csv(ROOT / "data" / "sample" / "candidate_items.csv")
    scenes = load_visual_interpretations(ROOT / "data" / "sample" / "vlm_scene_interpretations.jsonl")
    candidates = attach_visual_interpretations(candidates, scenes)
    user = UserProfile(user_id="demo_user", preferred_genres=("Drama", "Biography", "Sci-Fi"))
    top_k = 3

    variants = {
        "relevance_baseline": candidates.sort_values("baseline_score", ascending=False),
        "linear_gap_rerank": rerank_candidates(candidates, user),
        "vig_rerank": vig_rerank_candidates(candidates, user),
        "vig_no_listwise": vig_rerank_candidates(
            candidates,
            user,
            config=VIGRerankConfig(diversity_penalty=0.0, coverage_bonus=0.0),
        ),
        "vig_no_visual": vig_rerank_candidates(_zero_visual_gap(candidates), user),
    }

    rows = []
    for name, ranked in variants.items():
        top = ranked.head(top_k).copy()
        rows.append(
            {
                "variant": name,
                "avg_baseline_relevance": float(top["baseline_score"].mean()),
                "avg_visual_gap": float(top.get("visual_information_gap_score", 0).mean()),
                "avg_novelty": average_novelty(_ensure_unexpectedness(top, user)),
                "genre_diversity": intra_list_genre_diversity(top),
                "top_titles": " | ".join(top["title"].tolist()),
            }
        )

    output = pd.DataFrame(rows)
    output_path = ROOT / "data" / "sample" / "ablation_results.csv"
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    print(f"Wrote {output_path}")


def _zero_visual_gap(candidates: pd.DataFrame) -> pd.DataFrame:
    copied = candidates.copy()
    copied["visual_information_gap_score"] = 0.0
    copied["cross_modal_gap_score"] = 0.0
    return copied


def _ensure_unexpectedness(ranked: pd.DataFrame, user: UserProfile) -> pd.DataFrame:
    if "unexpectedness_score" in ranked:
        return ranked
    copied = ranked.copy()
    preferred = {genre.lower() for genre in user.preferred_genres}
    values = []
    for genres in copied["genres"].tolist():
        current = {genre.lower() for genre in str(genres).split("|")}
        union = current | preferred
        values.append(0.0 if not union else 1.0 - len(current & preferred) / len(union))
    copied["unexpectedness_score"] = values
    return copied


if __name__ == "__main__":
    main()
