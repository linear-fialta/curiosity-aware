from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from curiosity_reranker.visual import (
    cross_modal_gap_score,
    load_visual_interpretations,
    visual_information_gap_score,
    visual_reason,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample posters for human validation.")
    parser.add_argument("--metadata", default=str(ROOT / "data" / "external" / "tmdb_movies.csv"))
    parser.add_argument("--scenes", default=str(ROOT / "data" / "external" / "vlm_scene_interpretations.jsonl"))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--output-dir", default=str(ROOT / "annotation"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    scenes = load_visual_interpretations(Path(args.scenes))
    frame = _build_annotation_frame(metadata, scenes)
    sample = _stratified_sample(frame, n=args.n, seed=args.seed)

    csv_path = output_dir / "visual_gap_annotation_50.csv"
    html_path = output_dir / "visual_gap_annotation_50.html"
    guide_path = output_dir / "README.md"

    sample.to_csv(csv_path, index=False)
    html_path.write_text(_render_gallery(sample), encoding="utf-8")
    guide_path.write_text(_render_guide(), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {guide_path}")


def _build_annotation_frame(
    metadata: pd.DataFrame,
    scenes: dict[str, object],
) -> pd.DataFrame:
    rows = []
    for scene in scenes.values():
        movie = metadata[metadata["movieId"].astype(str) == scene.item_id]
        if movie.empty:
            continue
        row = movie.iloc[0]
        poster_url = str(row.get("poster_url", ""))
        if not poster_url or poster_url == "nan":
            continue
        genres = tuple(str(row.get("genres", "")).split("|"))
        visual_gap = visual_information_gap_score(scene)
        rows.append(
            {
                "item_id": scene.item_id,
                "movieId": int(row["movieId"]),
                "title": str(row.get("title", "")),
                "genres": str(row.get("genres", "")),
                "release_date": str(row.get("release_date", "")),
                "poster_url": poster_url,
                "overview": str(row.get("overview", "")),
                "vlm_visual_information_gap_score": visual_gap,
                "vlm_cross_modal_gap_score": cross_modal_gap_score(
                    scene,
                    title=str(row.get("title", "")),
                    genres=genres,
                    overview=str(row.get("overview", "")),
                ),
                "vlm_reason": visual_reason(scene),
                "vlm_main_objects": " | ".join(scene.main_objects),
                "vlm_setting": scene.setting,
                "vlm_visible_action": scene.visible_action,
                "vlm_missing_information": " | ".join(scene.occluded_or_missing_information),
                "vlm_genre_ambiguity": " | ".join(scene.genre_ambiguity),
                "vlm_emotional_tension": " | ".join(scene.emotional_tension),
                "vlm_implied_question": scene.implied_question,
            }
        )
    return pd.DataFrame(rows)


def _stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("No parsed poster scenes are available for sampling.")

    frame = frame.copy()
    score = frame["vlm_visual_information_gap_score"]
    frame["score_bin"] = "middle"
    frame.loc[score < 0.8, "score_bin"] = "low"
    frame.loc[score > 0.8, "score_bin"] = "high"
    bins = [label for label in ["low", "middle", "high"] if label in set(frame["score_bin"])]
    base = n // len(bins)
    remainder = n % len(bins)

    sampled = []
    for idx, label in enumerate(bins):
        target = base + int(idx < remainder)
        group = frame[frame["score_bin"] == label]
        sampled.append(group.sample(n=min(target, len(group)), random_state=seed + idx))

    output = pd.concat(sampled, ignore_index=True)
    if len(output) < n:
        remaining = frame[~frame["item_id"].isin(output["item_id"])]
        output = pd.concat(
            [
                output,
                remaining.sample(n=n - len(output), random_state=seed + 99),
            ],
            ignore_index=True,
        )

    output = output.sample(frac=1.0, random_state=seed + 123).reset_index(drop=True)
    output.insert(0, "annotation_id", [f"A{i:03d}" for i in range(1, len(output) + 1)])
    for column in [
        "human_missing_context_1_5",
        "human_visual_ambiguity_1_5",
        "human_emotional_tension_1_5",
        "human_curiosity_1_5",
        "human_notes",
    ]:
        output[column] = ""
    return output


def _render_gallery(sample: pd.DataFrame) -> str:
    rows = []
    for row in sample.itertuples(index=False):
        rows.append(
            f"""
            <section class="card">
              <div class="poster"><img src="{html.escape(row.poster_url)}" alt="{html.escape(row.title)} poster"></div>
              <div class="meta">
                <h2>{html.escape(row.annotation_id)} · {html.escape(row.title)}</h2>
                <p><strong>Genres:</strong> {html.escape(row.genres)}</p>
                <p><strong>VLM visual gap:</strong> {row.vlm_visual_information_gap_score:.3f}
                   · <strong>bin:</strong> {html.escape(str(row.score_bin))}</p>
                <p><strong>VLM reason:</strong> {html.escape(row.vlm_reason)}</p>
                <p><strong>Missing/context withheld:</strong> {html.escape(row.vlm_missing_information)}</p>
                <p><strong>Implied question:</strong> {html.escape(row.vlm_implied_question)}</p>
                <p class="overview">{html.escape(row.overview)}</p>
                <table>
                  <tr><th>missing context</th><th>visual ambiguity</th><th>emotional tension</th><th>curiosity</th></tr>
                  <tr><td>1 2 3 4 5</td><td>1 2 3 4 5</td><td>1 2 3 4 5</td><td>1 2 3 4 5</td></tr>
                </table>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Visual Gap Annotation Set</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2933; }}
    h1 {{ font-size: 26px; margin-bottom: 4px; }}
    .note {{ color: #52606d; max-width: 880px; line-height: 1.5; }}
    .card {{ display: grid; grid-template-columns: 170px 1fr; gap: 20px; padding: 18px 0; border-top: 1px solid #d9e2ec; }}
    .poster img {{ width: 170px; border: 1px solid #bcccdc; }}
    h2 {{ font-size: 18px; margin: 0 0 8px; }}
    p {{ margin: 5px 0; line-height: 1.45; }}
    .overview {{ color: #52606d; }}
    table {{ margin-top: 10px; border-collapse: collapse; }}
    th, td {{ border: 1px solid #bcccdc; padding: 6px 10px; font-size: 13px; text-align: center; }}
  </style>
</head>
<body>
  <h1>Visual Gap Annotation Set</h1>
  <p class="note">Use this page to look at the posters. Enter your scores in <code>visual_gap_annotation_50.csv</code>. Score each construct from 1 to 5, where 1 means very low and 5 means very high.</p>
  {''.join(rows)}
</body>
</html>
"""


def _render_guide() -> str:
    return """# Visual Gap Human Annotation

Use `visual_gap_annotation_50.html` to view the posters and `visual_gap_annotation_50.csv` to enter scores.

Score each item from 1 to 5:

- `human_missing_context_1_5`: how much important visual/contextual information is withheld.
- `human_visual_ambiguity_1_5`: how uncertain or multi-interpretable the image is.
- `human_emotional_tension_1_5`: how much unresolved emotional tension the image conveys.
- `human_curiosity_1_5`: how strongly the poster makes you want to know what happens.
- `human_notes`: short notes for obvious VLM errors or unusual cases.

Do not try to match the VLM score. Use your own judgment from the poster first, then glance at the VLM fields only if useful.
"""


if __name__ == "__main__":
    main()
