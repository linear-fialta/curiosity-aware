# Visual Information Gap Reranker

This repository implements a reproducible pilot study for exploratory recommendation. It extends my WITS 2025 paper, **"Trailers or Thieves? How Content Similarity Affects Continuance and Exploration for Condensed Clips in Algorithmic Feeds"**, from text-based narrative gaps in algorithmic feeds to image-based information gaps in recommender-system design.

The project asks whether movie posters contain structured visual cues that help explain why a user may explore an item that is relevant, but not merely redundant. The current implementation uses MovieLens ratings, TMDb metadata, Qwen2.5-VL scene parsing, a matrix-factorization candidate generator, and several reranking baselines.

## Research Question

Can visual information gaps extracted from item images improve exploratory recommendation beyond relevance, generic novelty, and direct VLM scoring?

## Theoretical Link to the WITS Paper

The WITS paper studies a "trailer versus thief" problem in algorithmic feeds. Its central mechanism is that engagement peaks when content balances processing fluency and novelty, while narrative gaps can redirect users from passive continuance toward active exploration.

This repository carries the same logic into a design artifact:

```text
moderate similarity + narrative incompleteness in condensed clips
        -> visual information gaps in posters/keyframes
        -> exploration-aware reranking
```

The construct boundary is important:

- Curiosity is treated as a pre-click mechanism: the motivation to resolve an information gap.
- Serendipity is treated as a post-consumption outcome: an unexpected recommendation that turns out to be useful.
- The VLM is used as a measurement layer, not as the recommender.

## Method Overview

```text
MovieLens ratings
        -> matrix-factorization candidate generator
        -> candidate list per user

TMDb posters
        -> Qwen2.5-VL scene parsing
        -> structured fields: objects, actions, missing information, genre ambiguity, emotional tension
        -> visual information gap score

Candidates + visual features
        -> relevance, visual gap, cross-modal gap, moderate-distance sweet spot
        -> VIG-Rerank and comparison baselines
        -> HitRate@K, NDCG@K, novelty, diversity, visual-gap intensity
```

## Repository Structure

```text
.
├── docs/
│   ├── algorithm_design.md
│   ├── construct_definition.md
│   ├── data_plan.md
│   ├── experiment_plan.md
│   ├── project_proposal.md
│   └── vlm_image_interpretation.md
├── scripts/
│   ├── download_movielens.py
│   ├── enrich_tmdb_metadata.py
│   ├── generate_vlm_scene_interpretations.py
│   ├── run_movielens_experiment.py
│   └── tune_vig_config.py
├── src/curiosity_reranker/
│   ├── baseline.py
│   ├── comparison.py
│   ├── features.py
│   ├── metrics.py
│   ├── vig_rerank.py
│   └── visual.py
├── tests/
│   └── test_rerank.py
└── pyproject.toml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional VLM dependencies:

```bash
pip install -e ".[vlm]"
```

## Data Preparation

Download MovieLens:

```bash
python scripts/download_movielens.py
```

Enrich MovieLens movies with TMDb metadata and poster URLs:

```bash
export TMDB_API_KEY="your_tmdb_v3_api_key"
python scripts/enrich_tmdb_metadata.py
```

The TMDb script is resumable. It writes:

```text
data/external/tmdb_movie_cache.csv
data/external/tmdb_movies.csv
```

Generate visual scene interpretations:

```bash
python scripts/generate_vlm_scene_interpretations.py \
  --input data/external/tmdb_movies.csv \
  --output data/external/vlm_scene_interpretations.jsonl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --limit 3000
```

The VLM script is also resumable. Re-running the same command skips completed `item_id`s. Use `--overwrite` only when intentionally regenerating the file.

## Run Experiments

Run the full MovieLens experiment with VLM scene fields:

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

Outputs:

```text
data/processed/movielens_experiment/
├── mf_candidates.csv
├── summary_metrics.csv
└── *_rankings.csv
```

Tune VIG-Rerank parameters without rerunning VLM inference:

```bash
PYTHONPATH=src python scripts/tune_vig_config.py \
  --metadata data/external/tmdb_movies.csv \
  --scenes data/external/vlm_scene_interpretations.jsonl \
  --max-users 100 \
  --candidate-k 100 \
  --top-k 10
```

Run tests:

```bash
python -m pytest
```

## Baselines

The experiment compares:

- `mf_relevance`: matrix-factorization relevance ranking.
- `mmr`: relevance-diversity reranking.
- `serendipity`: relevance plus genre unexpectedness.
- `linear_gap`: linear weighted reranking using visual/text gaps.
- `direct_vlm_proxy`: direct ranking by visual-gap features.
- `vig_rerank`: nonlinear VIG-Rerank with sweet-spot and listwise terms.
- `vig_no_visual`: VIG-Rerank with visual and cross-modal gap removed.
- `vig_no_listwise`: VIG-Rerank without redundancy penalty or coverage bonus.

## Current Pilot Result

The current local pilot uses:

- MovieLens `ml-latest-small`
- 9,622 TMDb-enriched movies
- 2,998 Qwen2.5-VL parsed posters
- 100 users, 100 candidates per user, top-10 evaluation

Latest metrics:

| variant | HitRate@10 | NDCG@10 | avg visual gap | avg novelty | genre diversity |
|---|---:|---:|---:|---:|---:|
| `mf_relevance` | 0.02 | 0.0069 | 0.6788 | 0.7834 | 0.7388 |
| `mmr` | 0.02 | 0.0100 | 0.6698 | 0.8475 | 0.9526 |
| `serendipity` | 0.00 | 0.0000 | 0.6412 | 0.9921 | 0.7965 |
| `linear_gap` | 0.03 | 0.0125 | 0.8350 | 0.5905 | 0.7016 |
| `direct_vlm_proxy` | 0.09 | 0.0408 | 0.8680 | 0.7987 | 0.8533 |
| `vig_rerank` | 0.09 | 0.0367 | 0.8179 | 0.5320 | 0.6752 |
| `vig_no_visual` | 0.06 | 0.0273 | 0.0000 | 0.8420 | 0.9621 |
| `vig_no_listwise` | 0.08 | 0.0355 | 0.8168 | 0.5243 | 0.6436 |

Interpretation:

- Visual-gap-aware methods outperform MF, MMR, serendipity, and no-visual ablations in this pilot.
- Direct visual-gap ranking is a strong baseline; VIG-Rerank matches its HitRate@10 while imposing a theoretically motivated sweet-spot structure.
- Generic novelty performs poorly: the serendipity baseline has the highest novelty but zero HitRate@10.
- The listwise component improves conceptual control over redundancy, but its relevance-diversity trade-off needs further tuning.

## Current Limitations

- The VLM coverage is partial: 2,998 posters, not the full TMDb-enriched set.
- Poster-level visual interpretation is noisy and should be validated by human coding on a sample.
- Offline HitRate/NDCG evaluate held-out MovieLens behavior, not perceived curiosity directly.
- The next step is a small human evaluation comparing baseline and VIG-ranked lists.

