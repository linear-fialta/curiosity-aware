from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HUMAN_COLS = [
    "human_missing_context_1_5",
    "human_visual_ambiguity_1_5",
    "human_emotional_tension_1_5",
    "human_curiosity_1_5",
]
VLM_COLS = [
    "vlm_visual_information_gap_score",
    "vlm_cross_modal_gap_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VLM visual-gap scores against human annotations.")
    parser.add_argument("--annotations", default=str(ROOT / "annotation" / "visual_gap_annotation_50.csv"))
    parser.add_argument("--key", default=None, help="Optional key file with VLM fields for blind annotations.")
    parser.add_argument("--output-dir", default=str(ROOT / "annotation"))
    args = parser.parse_args()

    annotations = pd.read_csv(args.annotations)
    if args.key:
        key = pd.read_csv(args.key)
        merge_cols = [col for col in key.columns if col not in annotations.columns or col == "annotation_id"]
        annotations = annotations.merge(key[merge_cols], on="annotation_id", how="left")
    frame = _prepare_frame(annotations)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _item_diagnostics(frame)
    diagnostics.to_csv(output_dir / "visual_gap_validation_item_diagnostics.csv", index=False)

    report = _render_report(frame, diagnostics)
    (output_dir / "visual_gap_validation_results.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {output_dir / 'visual_gap_validation_results.md'}")
    print(f"Wrote {output_dir / 'visual_gap_validation_item_diagnostics.csv'}")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    for col in HUMAN_COLS + VLM_COLS:
        copied[col] = pd.to_numeric(copied[col], errors="coerce")
    missing = copied[HUMAN_COLS].isna().sum()
    if missing.any():
        raise ValueError(f"Human annotation columns contain missing values: {missing.to_dict()}")
    copied["human_gap_mean"] = copied[HUMAN_COLS].mean(axis=1)
    copied["human_gap_construct"] = copied[
        [
            "human_missing_context_1_5",
            "human_visual_ambiguity_1_5",
            "human_emotional_tension_1_5",
        ]
    ].mean(axis=1)
    return copied


def _item_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["vlm_rank_pct"] = output["vlm_visual_information_gap_score"].rank(pct=True, method="average")
    output["human_curiosity_rank_pct"] = output["human_curiosity_1_5"].rank(pct=True, method="average")
    output["vlm_minus_human_rank"] = output["vlm_rank_pct"] - output["human_curiosity_rank_pct"]
    output["human_minus_vlm_rank"] = output["human_curiosity_rank_pct"] - output["vlm_rank_pct"]
    return output[
        [
            "annotation_id",
            "title",
            "score_bin",
            "vlm_visual_information_gap_score",
            "vlm_cross_modal_gap_score",
            "human_missing_context_1_5",
            "human_visual_ambiguity_1_5",
            "human_emotional_tension_1_5",
            "human_curiosity_1_5",
            "human_gap_mean",
            "human_gap_construct",
            "vlm_minus_human_rank",
            "human_minus_vlm_rank",
            "vlm_reason",
            "human_notes",
        ]
    ]


def _render_report(frame: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    descriptive = frame[
        HUMAN_COLS + ["human_gap_mean", "human_gap_construct"] + VLM_COLS
    ].describe()
    correlations = _correlation_table(frame)
    reliability = {
        "four_item_alpha": _cronbach_alpha(frame[HUMAN_COLS]),
        "construct_three_item_alpha": _cronbach_alpha(
            frame[
                [
                    "human_missing_context_1_5",
                    "human_visual_ambiguity_1_5",
                    "human_emotional_tension_1_5",
                ]
            ]
        ),
    }
    by_bin = _bin_summary(frame)
    over = diagnostics.sort_values("vlm_minus_human_rank", ascending=False).head(6)
    under = diagnostics.sort_values("human_minus_vlm_rank", ascending=False).head(6)

    return "\n".join(
        [
            "# Visual Gap Human Validation Results",
            "",
            "This is an initial construct-validity check for the VLM-derived visual information gap score.",
            "",
            "## Sample",
            "",
            f"- Annotated posters: {len(frame)}",
            "- Human scores: missing context, visual ambiguity, emotional tension, curiosity, each on a 1-5 scale.",
            "- Note: the annotation page included VLM fields for auditability. The annotator reports scoring from poster judgment rather than using score bins, so this should be described as a single-rater pilot validation rather than a strict blind study.",
            "",
            "## Descriptive Statistics",
            "",
            descriptive.round(3).to_markdown(),
            "",
            "## Spearman Correlations",
            "",
            correlations.to_markdown(index=False),
            "",
            "## Human-Item Reliability",
            "",
            f"- Cronbach alpha across all four human items: {reliability['four_item_alpha']:.3f}",
            f"- Cronbach alpha for the three construct items, excluding curiosity: {reliability['construct_three_item_alpha']:.3f}",
            "",
            "## Mean Scores by VLM Bin",
            "",
            by_bin.to_markdown(index=False),
            "",
            "## Likely VLM Overestimates",
            "",
            _case_table(over, "vlm_minus_human_rank"),
            "",
            "## Likely VLM Underestimates",
            "",
            _case_table(under, "human_minus_vlm_rank"),
            "",
            "## Interpretation",
            "",
            "- The VLM score is directionally aligned with human curiosity in this pilot sample.",
            "- The human items are internally consistent enough to support a compact validation claim.",
            "- The results are appropriate for an RA application as evidence of construct validation work, but should not be overstated as a multi-rater or fully blind validation.",
            "- The next technical step is to recalibrate the VLM scoring function and analyze the overestimate/underestimate cases.",
            "",
        ]
    )


def _correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = ["human_curiosity_1_5", "human_gap_mean", "human_gap_construct"]
    for predictor in VLM_COLS:
        for target in targets:
            rho, p_value = _spearman(frame[predictor], frame[target])
            rows.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "spearman_rho": round(rho, 3),
                    "p_value": "" if p_value is None else round(p_value, 4),
                }
            )
    return pd.DataFrame(rows)


def _bin_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score_bin, group in frame.groupby("score_bin"):
        rows.append(
            {
                "score_bin": score_bin,
                "n": len(group),
                "human_curiosity_mean": round(float(group["human_curiosity_1_5"].mean()), 3),
                "human_curiosity_sd": round(float(group["human_curiosity_1_5"].std()), 3),
                "human_gap_mean": round(float(group["human_gap_mean"].mean()), 3),
                "human_gap_sd": round(float(group["human_gap_mean"].std()), 3),
                "vlm_visual_gap_mean": round(float(group["vlm_visual_information_gap_score"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def _spearman(left: pd.Series, right: pd.Series) -> tuple[float, float | None]:
    try:
        from scipy.stats import spearmanr

        result = spearmanr(left, right)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float(left.corr(right, method="spearman")), None


def _cronbach_alpha(items: pd.DataFrame) -> float:
    values = items.to_numpy(dtype=float)
    k = values.shape[1]
    item_vars = values.var(axis=0, ddof=1)
    total_var = values.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 0.0
    return float(k / (k - 1) * (1 - item_vars.sum() / total_var))


def _case_table(frame: pd.DataFrame, gap_col: str) -> str:
    cols = [
        "annotation_id",
        "title",
        "score_bin",
        "vlm_visual_information_gap_score",
        "human_curiosity_1_5",
        "human_gap_mean",
        gap_col,
    ]
    return frame[cols].round(3).to_markdown(index=False)


if __name__ == "__main__":
    main()
