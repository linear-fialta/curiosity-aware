# Curiosity-Aware Multimodal Reranker

This repository is an early-stage design science project on multimodal recommender systems. The core idea is to design and evaluate a lightweight reranking artifact that helps recommender systems support exploratory consumption, not only relevance optimization.

## Research Question

Can multimodal curiosity signals improve exploratory recommendation quality beyond relevance-based ranking?

## Motivation

Many recommender systems are optimized for short-term relevance and engagement. In exploratory domains such as movies, short videos, books, courses, and products, users may also value recommendations that are surprising, curiosity-inducing, and meaningfully different from what they already know.

This project studies whether a VLM/LLM-enhanced reranker can identify curiosity-triggering features from item metadata, images, trailers, reviews, or short-video keyframes, and use those signals to improve top-k recommendation lists.

## Artifact

The initial artifact is a modular reranking pipeline:

```text
User history + candidate items
        |
        v
Baseline relevance score
        |
        v
Curiosity feature extraction
        |
        v
Curiosity-aware reranking
        |
        v
Top-k recommendations + explanations
```

The first runnable version uses structured and text metadata only. The next version will add poster/keyframe-based VLM features.

## Repository Structure

```text
.
├── docs/
│   ├── data_plan.md
│   └── project_proposal.md
├── scripts/
│   ├── build_sample_data.py
│   └── run_demo.py
├── src/
│   └── curiosity_reranker/
│       ├── __init__.py
│       ├── features.py
│       ├── metrics.py
│       ├── rerank.py
│       └── schema.py
├── tests/
│   └── test_rerank.py
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/build_sample_data.py
python scripts/run_demo.py
pytest
```

## Data Setup

For the full project, start with MovieLens and then enrich items with TMDb metadata:

```bash
python scripts/download_movielens.py
export TMDB_API_KEY="your_tmdb_key"
python scripts/enrich_tmdb_metadata.py
```

## Planned Data Sources

The recommended first dataset is MovieLens plus movie posters and metadata from TMDb. This combination is small enough to implement quickly, but rich enough to support multimodal recommendation experiments.

See [docs/data_plan.md](docs/data_plan.md) for details.

## Current Status

- [x] Research framing
- [x] Runnable sample pipeline
- [x] Baseline reranking logic
- [x] Initial unit tests
- [ ] MovieLens ingestion
- [ ] TMDb metadata/poster enrichment
- [ ] VLM-based visual curiosity extraction
- [ ] Offline evaluation
- [ ] Human evaluation design
