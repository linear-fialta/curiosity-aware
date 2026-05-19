# Experiment Plan

## Objective

The experiment evaluates whether visual information gaps provide useful exploratory recommendation signals beyond relevance and generic novelty.

## Pipeline

1. Train a matrix-factorization model on MovieLens ratings.
2. Use leave-one-out evaluation: the most recent rating per user is held out.
3. Generate candidate items with matrix factorization.
4. Attach TMDb metadata and VLM-parsed poster fields.
5. Compute visual information gap and cross-modal gap features.
6. Compare reranking methods on top-k metrics.

## Compared Methods

- `mf_relevance`: sort candidates by matrix-factorization score.
- `mmr`: greedy relevance-diversity reranking.
- `serendipity`: relevance plus genre unexpectedness.
- `linear_gap`: linear weighted gap reranking.
- `direct_vlm_proxy`: direct ranking by visual-gap features.
- `vig_rerank`: nonlinear VIG-Rerank with sweet-spot and listwise terms.
- `vig_no_visual`: remove visual and cross-modal gap features.
- `vig_no_listwise`: remove redundancy penalty and coverage bonus.

## Metrics

- `HitRate@K`: whether the held-out item appears in the top-k list.
- `NDCG@K`: rank-sensitive held-out item recovery.
- `avg_baseline_relevance`: average matrix-factorization score in top-k.
- `avg_visual_gap`: average visual information gap in top-k.
- `avg_novelty`: average genre distance from the user profile.
- `genre_diversity`: intra-list genre diversity.

## Run Commands

Without VLM scenes:

```bash
python scripts/download_movielens.py
PYTHONPATH=src python scripts/run_movielens_experiment.py \
  --max-users 50 \
  --candidate-k 100 \
  --top-k 10 \
  --epochs 8 \
  --factors 32
```

With VLM scenes:

```bash
PYTHONPATH=src python scripts/run_movielens_experiment.py \
  --metadata data/external/tmdb_movies.csv \
  --scenes data/external/vlm_scene_interpretations.jsonl \
  --max-users 100 \
  --candidate-k 100 \
  --top-k 10 \
  --epochs 8 \
  --factors 32
```

Tune VIG-Rerank:

```bash
PYTHONPATH=src python scripts/tune_vig_config.py \
  --metadata data/external/tmdb_movies.csv \
  --scenes data/external/vlm_scene_interpretations.jsonl \
  --max-users 100 \
  --candidate-k 100 \
  --top-k 10
```

## Interpretation Logic

The main ablation is `vig_rerank` versus `vig_no_visual`. If VIG-Rerank drops sharply when visual features are removed, the visual information gap mechanism carries incremental signal.

The comparison with `direct_vlm_proxy` tests whether the structured reranking objective improves over directly sorting by VLM-derived visual fields.

The comparison with `serendipity` tests whether generic novelty is sufficient. In the current pilot it is not: high novelty can move recommendations too far from the user's interpretable preference region.

## Current Status

The current pilot is an offline evaluation. A human-subject evaluation is still needed to test perceived curiosity, willingness to click, and perceived serendipity after item inspection.

