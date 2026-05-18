from __future__ import annotations

from pathlib import Path

import pandas as pd

from curiosity_reranker.metrics import average_novelty, intra_list_genre_diversity
from curiosity_reranker.rerank import rerank_candidates
from curiosity_reranker.schema import UserProfile
from curiosity_reranker.vig_rerank import vig_rerank_candidates
from curiosity_reranker.visual import attach_visual_interpretations, load_visual_interpretations


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    candidates = pd.read_csv(ROOT / "data" / "sample" / "candidate_items.csv")
    scenes = load_visual_interpretations(ROOT / "data" / "sample" / "vlm_scene_interpretations.jsonl")
    candidates = attach_visual_interpretations(candidates, scenes)
    user = UserProfile(user_id="demo_user", preferred_genres=("Drama", "Biography", "Sci-Fi"))
    ranked = rerank_candidates(candidates, user)
    vig_ranked = vig_rerank_candidates(candidates, user)

    output_path = ROOT / "data" / "sample" / "reranked_items.csv"
    vig_output_path = ROOT / "data" / "sample" / "vig_reranked_items.csv"
    ranked.to_csv(output_path, index=False)
    vig_ranked.to_csv(vig_output_path, index=False)

    display_cols = [
        "title",
        "baseline_score",
        "visual_information_gap_score",
        "cross_modal_gap_score",
        "text_information_gap_score",
        "unexpectedness_score",
        "moderate_unexpectedness_score",
        "rerank_score",
        "explanation",
    ]
    print(ranked[display_cols].to_string(index=False))
    print()
    vig_display_cols = [
        "vig_rank",
        "title",
        "relevance_score",
        "visual_information_gap_score",
        "cross_modal_gap_score",
        "text_information_gap_score",
        "taste_distance_score",
        "sweet_spot_score",
        "vig_item_score",
        "vig_listwise_score",
        "vig_explanation",
    ]
    print("VIG-Rerank")
    print(vig_ranked[vig_display_cols].to_string(index=False))
    print()
    print(f"Average novelty: {average_novelty(ranked):.3f}")
    print(f"Intra-list genre diversity: {intra_list_genre_diversity(ranked):.3f}")
    print(f"Wrote {output_path}")
    print(f"Wrote {vig_output_path}")


if __name__ == "__main__":
    main()
