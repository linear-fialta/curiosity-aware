# Visual Information Gap Reranker

This is my implementation for a research pilot on visual information gaps in exploratory recommender systems.

The project extends my WITS 2025 paper, **"Trailers or Thieves? How Content Similarity Affects Continuance and Exploration for Condensed Clips in Algorithmic Feeds"**, from text-based narrative gaps in algorithmic feeds to image-based information gaps in recommender-system design.

The project asks whether movie posters contain structured visual cues that help explain why a user may explore an item that is relevant, but not merely redundant. The current implementation uses MovieLens ratings, TMDb metadata, Qwen2.5-VL scene parsing, a matrix-factorization candidate generator, and several reranking baselines.

Author: Yachen Guo

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
│   ├── pilot_results.md
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
├── pyproject.toml
└── train.py
```

## Environment Settings

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional VLM dependencies:

```bash
pip install -e ".[vlm]"
```

The pilot uses Python 3.10+ and Qwen2.5-VL for poster parsing.

## Dataset

Dataset:

- [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/)
- TMDb metadata and poster URLs, collected with a local TMDb API key
- Qwen2.5-VL poster scene interpretations generated from the TMDb posters

The generated TMDb and VLM files are not committed to the repository. The scripts below reproduce the local files used by the pilot.

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
python train.py \
  --metadata data/external/tmdb_movies.csv \
  --scenes data/external/vlm_scene_interpretations.jsonl \
  --max-users 100 \
  --candidate-k 100 \
  --top-k 10 \
  --epochs 8 \
  --factors 32 \
  --seed 42
```

The same experiment can also be run through `scripts/run_movielens_experiment.py`; the command-line arguments are defined in its `parse_args` block.

Outputs:

```text
data/processed/movielens_experiment/
├── mf_candidates.csv
├── run_config.json
├── summary_metrics.csv
├── summary_metric_intervals.csv
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

## Current Pilot

The current local pilot uses:

- MovieLens `ml-latest-small`
- 9,622 TMDb-enriched movies
- 2,998 Qwen2.5-VL parsed posters
- 100 users, 100 candidates per user, top-10 evaluation
- seed 42, with bootstrap intervals saved for the ranking metrics

The latest run shows that visual-gap-aware methods outperform MF, MMR, serendipity, and no-visual ablations on HitRate@10. Direct visual-gap ranking remains a strong baseline; VIG-Rerank matches its HitRate@10 while imposing a theoretically motivated sweet-spot structure.

See [docs/pilot_results.md](docs/pilot_results.md) for the full table and interpretation.

## Current Limitations

- The VLM coverage is partial: 2,998 posters, not the full TMDb-enriched set.
- Poster-level visual interpretation is noisy and should be validated by human coding on a sample.
- Offline HitRate/NDCG evaluate held-out MovieLens behavior, not perceived curiosity directly.
- The next step is a small human evaluation comparing baseline and VIG-ranked lists.

## Acknowledgement

This project is partly motivated by prior work on unexpected recommendation, serendipity, and cross-domain recommendation. It is designed as a compact research artifact for testing whether VLM-derived visual information gaps can support exploration-aware reranking.

Last Update: 2026/05/20
