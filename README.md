# Visual Information Gap Reranker

This repository is an early-stage design science project on exploratory recommender systems. The core idea is to design and evaluate a lightweight reranking artifact that uses visual information gaps to encourage exploration, not only relevance optimization.

## Research Question

Can visual information gaps extracted from item images help recommender systems generate recommendations that are relevant enough to understand, but unresolved enough to invite exploration?

## Motivation

Many recommender systems optimize for relevance, similarity, and short-term engagement. However, exploratory consumption often begins when an item creates a productive information gap: the user recognizes enough to care, but sees something unresolved enough to want to know more.

This project treats curiosity as a pre-click mechanism and serendipity as a post-consumption outcome. The artifact uses VLMs as visual scene parsers, not as black-box judges. A VLM converts posters or video keyframes into auditable scene elements such as objects, actions, missing information, incongruity, and implied questions. The reranker then computes a visual information gap score from those elements.

## Artifact

The initial artifact is a modular reranking pipeline:

```text
User history + candidate items
        |
        v
Baseline relevance score
        |
        v
VLM scene interpretation
        |
        v
Visual information gap scoring
        |
        v
Exploration-aware reranking
        |
        v
Top-k recommendations + explanations
```

The first runnable version uses sample VLM-style scene interpretations. The next version will generate those interpretations from TMDb posters or short-video keyframes.

## Repository Structure

```text
.
├── docs/
│   ├── data_plan.md
│   ├── construct_definition.md
│   ├── algorithm_design.md
│   ├── experiment_plan.md
│   ├── vlm_image_interpretation.md
│   └── project_proposal.md
├── scripts/
│   ├── build_sample_data.py
│   ├── download_movielens.py
│   ├── enrich_tmdb_metadata.py
│   ├── generate_vlm_scene_interpretations.py
│   ├── run_ablation.py
│   └── run_demo.py
├── src/
│   └── curiosity_reranker/
│       ├── __init__.py
│       ├── features.py
│       ├── metrics.py
│       ├── rerank.py
│       ├── schema.py
│       ├── vig_rerank.py
│       └── visual.py
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
python scripts/run_ablation.py
pytest
```

## Data Setup

For the full project, start with MovieLens and then enrich items with TMDb metadata:

```bash
python scripts/download_movielens.py
export TMDB_API_KEY="your_tmdb_key"
python scripts/enrich_tmdb_metadata.py
```

To generate real visual scene interpretations with Qwen2.5-VL:

```bash
pip install -e ".[vlm]"
python scripts/generate_vlm_scene_interpretations.py \
  --input data/external/tmdb_movies.csv \
  --output data/external/vlm_scene_interpretations.jsonl \
  --model Qwen/Qwen2.5-VL-7B-Instruct
```

## Planned Data Sources

The recommended first dataset is MovieLens plus movie posters and metadata from TMDb. This combination is small enough to implement quickly, but rich enough to support multimodal recommendation experiments.

See [docs/data_plan.md](docs/data_plan.md) for details.

## Current Status

- [x] Construct framing: curiosity mechanism vs. serendipity outcome
- [x] Runnable sample pipeline
- [x] Visual information gap scoring logic
- [x] Nonlinear VIG-Rerank algorithm
- [x] Ablation script
- [x] Initial unit tests
- [x] MovieLens ingestion script
- [x] TMDb metadata/poster enrichment script
- [ ] VLM-based scene interpretation from real posters
- [ ] Offline evaluation
- [ ] Human evaluation design
