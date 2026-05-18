from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(
        [
            {
                "item_id": "m001",
                "title": "Arrival",
                "genres": "Sci-Fi|Drama|Mystery",
                "overview": "A linguist uncovers a hidden structure behind alien messages and must discover why time itself seems to bend.",
                "baseline_score": 0.84,
            },
            {
                "item_id": "m002",
                "title": "The Social Network",
                "genres": "Drama|Biography",
                "overview": "A young programmer builds a social platform that transforms friendship, status, and power.",
                "baseline_score": 0.89,
            },
            {
                "item_id": "m003",
                "title": "Spirited Away",
                "genres": "Animation|Fantasy|Adventure",
                "overview": "A girl enters a hidden spirit world where every rule is unknown and every bargain reveals a new mystery.",
                "baseline_score": 0.77,
            },
            {
                "item_id": "m004",
                "title": "Moneyball",
                "genres": "Drama|Sport|Biography",
                "overview": "A baseball manager uses data and unconventional reasoning to challenge the logic of professional scouting.",
                "baseline_score": 0.81,
            },
            {
                "item_id": "m005",
                "title": "Ex Machina",
                "genres": "Sci-Fi|Thriller|Drama",
                "overview": "A programmer evaluates an artificial intelligence experiment and uncovers hidden motives inside a sealed research facility.",
                "baseline_score": 0.80,
            },
        ]
    )
    candidates.to_csv(SAMPLE_DIR / "candidate_items.csv", index=False)
    print(f"Wrote {SAMPLE_DIR / 'candidate_items.csv'}")


if __name__ == "__main__":
    main()

