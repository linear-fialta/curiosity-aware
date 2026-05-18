from __future__ import annotations

from pathlib import Path

import pandas as pd

from curiosity_reranker.metrics import average_novelty, intra_list_genre_diversity
from curiosity_reranker.rerank import rerank_candidates
from curiosity_reranker.schema import UserProfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    candidates = pd.read_csv(ROOT / "data" / "sample" / "candidate_items.csv")
    user = UserProfile(user_id="demo_user", preferred_genres=("Drama", "Biography", "Sci-Fi"))
    ranked = rerank_candidates(candidates, user)

    output_path = ROOT / "data" / "sample" / "reranked_items.csv"
    ranked.to_csv(output_path, index=False)

    display_cols = [
        "title",
        "baseline_score",
        "curiosity_score",
        "unexpectedness_score",
        "rerank_score",
        "explanation",
    ]
    print(ranked[display_cols].to_string(index=False))
    print()
    print(f"Average novelty: {average_novelty(ranked):.3f}")
    print(f"Intra-list genre diversity: {intra_list_genre_diversity(ranked):.3f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

