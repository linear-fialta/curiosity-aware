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

- `mf_relevance`: matrix-factorization baseline score.
- `mmr`: relevance-diversity reranking.
- `serendipity`: relevance + unexpectedness.
- `linear_gap`: simple weighted visual-gap reranking.
- `direct_vlm_proxy`: direct visual-gap score baseline.
- `vig_rerank`: nonlinear sweet-spot + listwise VIG-Rerank.
- `vig_no_listwise`: removes redundancy penalty and coverage bonus.
- `vig_no_visual`: removes visual and cross-modal gap features.

Run the sample ablation with:

```bash
python scripts/run_ablation.py
```

Run the MovieLens experiment without VLM scenes:

```bash
python scripts/download_movielens.py
PYTHONPATH=src python scripts/run_movielens_experiment.py \
  --max-users 50 \
  --candidate-k 100 \
  --top-k 10 \
  --epochs 8 \
  --factors 32
```

This mode is useful for verifying the recommender baseline, listwise reranking, and evaluation code before running any VLM model.

Run the full VLM-assisted version:

```bash
python scripts/enrich_tmdb_metadata.py
pip install -e ".[vlm]"
python scripts/generate_vlm_scene_interpretations.py \
  --input data/external/tmdb_movies.csv \
  --output data/external/vlm_scene_interpretations.jsonl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --limit 200
PYTHONPATH=src python scripts/run_movielens_experiment.py \
  --metadata data/external/tmdb_movies.csv \
  --scenes data/external/vlm_scene_interpretations.jsonl \
  --max-users 100 \
  --candidate-k 100 \
  --top-k 10
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

## Current Implementation Status

The repository now includes a runnable matrix-factorization baseline implemented in numpy, not through a recommender-system package. It also includes end-to-end candidate generation and ranking evaluation. The VLM scene generation script is ready, but running it requires a machine with enough memory for Qwen2.5-VL.
