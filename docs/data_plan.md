# Data Plan

This project should start with a dataset that is public, easy to reproduce, and naturally multimodal. The goal is not to build the largest possible system, but to show a complete design science workflow: artifact design, implementation, evaluation, and research contribution.

## Recommended Dataset: MovieLens + TMDb

### 1. MovieLens Ratings

Use MovieLens as the behavioral backbone.

Required files:

- `ratings.csv`: user-item interactions
- `movies.csv`: movie titles and genres
- `links.csv`: mapping from MovieLens IDs to TMDb IDs

Suggested starting version:

- `ml-latest-small` for fast iteration and local experiments
- `ml-25m` later if larger-scale evaluation is needed

Repository command:

```bash
python scripts/download_movielens.py
```

Why it works:

- Standard recommender-system benchmark
- Easy to reproduce
- Good enough for offline evaluation
- Links to external movie metadata and posters

### 2. TMDb Movie Metadata

Use TMDb to enrich each movie with multimodal and semantic features.

Useful fields:

- `overview`: plot description
- `poster_path`: movie poster image
- `tagline`: short promotional text
- `genres`: semantic categories
- `release_date`: temporal context
- `popularity`: market-level attention signal

Why it matters:

- Posters allow VLM-based visual analysis.
- Overviews and taglines allow LLM-based narrative and curiosity analysis.
- Movie recommendation is easy to explain to nontechnical readers.

Repository command:

```bash
export TMDB_API_KEY="your_tmdb_key"
python scripts/enrich_tmdb_metadata.py
```

The enrichment script reads `data/raw/ml-latest-small/links.csv` and writes `data/external/tmdb_movies.csv`.

### 3. Optional Trailer or Keyframe Data

If time allows, add trailer keyframes.

Possible features:

- visual novelty
- scene diversity
- unresolved narrative cue
- emotional contrast
- semantic gap between title, poster, and overview

This connects strongly to prior work on condensed clips and narrative completeness.

## Minimal Data Schema

The first version only needs one candidate-item table.

```text
item_id
title
genres
overview
poster_url
baseline_score
```

A later version can add user histories:

```text
user_id
item_id
rating
timestamp
```

## Curiosity Signals

Initial text-only proxies:

- `novelty_score`: how different an item is from the user's watched genres
- `semantic_gap_score`: whether title, genre, and overview create an information gap
- `narrative_incompleteness_score`: whether the overview leaves unresolved questions
- `unexpectedness_score`: how far the item is from the user's usual profile while staying relevant

Later VLM features:

- poster visual complexity
- unusual visual objects
- emotion/style contrast
- mismatch between poster and genre
- visual ambiguity

## Evaluation Plan

### Offline Evaluation

Compare baseline top-k recommendation with curiosity-aware reranking.

Metrics:

- relevance: NDCG, hit rate, recall
- exploration: novelty, intra-list diversity, unexpectedness
- balance: relevance-diversity frontier

### Human Evaluation

Show participants two recommendation lists and ask:

- Which list makes you more interested in exploring?
- Which list feels more surprising but still relevant?
- Which explanation makes you more likely to click?

### Design Science Contribution

The contribution is not only a better metric. It is an artifact that operationalizes curiosity in recommender systems and demonstrates how multimodal generative AI can support exploratory consumption.
