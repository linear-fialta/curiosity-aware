# Experiment Plan

## Goal

The goal is to show that visual information gaps improve exploratory recommendation quality while preserving relevance.

## Data

Start with MovieLens + TMDb posters:

1. Use MovieLens ratings to construct users, histories, and candidate items.
2. Use TMDb metadata to attach posters and overviews.
3. Use Qwen2.5-VL to parse posters into scene interpretations.
4. Compute visual information gap features.

## Baselines

The first experiment should compare:

- `relevance_baseline`: sort by baseline recommender score.
- `linear_gap_rerank`: simple weighted reranking.
- `vig_rerank`: nonlinear sweet-spot + listwise VIG-Rerank.
- `vig_no_listwise`: removes redundancy penalty and coverage bonus.
- `vig_no_visual`: removes visual and cross-modal gap features.

Run the sample ablation with:

```bash
python scripts/run_ablation.py
```

## Metrics

Offline metrics:

- relevance retention: average baseline relevance in top-k
- novelty: distance from user profile
- diversity: intra-list genre diversity
- visual gap intensity: average visual information gap in top-k

Human evaluation:

- perceived curiosity
- willingness to click
- perceived relevance
- perceived serendipity after viewing details

## Key Ablation Logic

If `vig_rerank` beats `vig_no_visual` on perceived curiosity while preserving relevance, the visual information gap mechanism has evidence.

If `vig_rerank` beats `vig_no_listwise` on diversity without much relevance loss, the listwise optimization has evidence.

If `vig_rerank` beats direct VLM curiosity scoring, the structured-measurement approach is stronger than prompt-only scoring.

