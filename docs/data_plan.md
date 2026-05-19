# Data Plan

The empirical setting is intentionally modest: a public recommender-system benchmark enriched with poster-level visual information. The objective is not to maximize scale, but to keep the design artifact reproducible and easy to inspect.

## Primary Data Sources

### MovieLens

MovieLens provides the behavioral backbone:

- `ratings.csv`: user-item interactions
- `movies.csv`: titles and genres
- `links.csv`: mapping from MovieLens IDs to TMDb IDs

The current implementation uses `ml-latest-small` for fast iteration. Larger MovieLens releases can be substituted once the pipeline is stable.

```bash
python scripts/download_movielens.py
```

### TMDb

TMDb supplies item metadata and poster URLs:

- `overview`
- `tagline`
- `release_date`
- `popularity`
- `poster_url`

The enrichment script is resumable and writes a cache every 25 records.

```bash
export TMDB_API_KEY="your_tmdb_v3_api_key"
python scripts/enrich_tmdb_metadata.py
```

Outputs:

```text
data/external/tmdb_movie_cache.csv
data/external/tmdb_movies.csv
```

### VLM Scene Interpretations

Poster URLs are parsed with Qwen2.5-VL into structured visual fields. The model is not asked to recommend movies or score curiosity directly. It only extracts scene information that is later converted into visual-gap variables.

Required fields:

```text
item_id
main_objects
setting
visible_action
occluded_or_missing_information
object_context_incongruity
genre_ambiguity
emotional_tension
implied_question
```

Generation command:

```bash
python scripts/generate_vlm_scene_interpretations.py \
  --input data/external/tmdb_movies.csv \
  --output data/external/vlm_scene_interpretations.jsonl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --limit 3000
```

The script supports resume-by-default behavior.

## Derived Tables

`scripts/run_movielens_experiment.py` creates:

```text
data/processed/movielens_experiment/mf_candidates.csv
data/processed/movielens_experiment/summary_metrics.csv
data/processed/movielens_experiment/*_rankings.csv
```

The candidate table contains:

```text
user_id
item_id
movieId
title
genres
overview
baseline_score
predicted_rating
heldout_item
is_relevant
visual_information_gap_score
cross_modal_gap_score
visual_reason
```

## Feature Construction

The core visual-gap score uses five components:

- missing or occluded information
- unresolved visible action
- object-context incongruity
- genre ambiguity
- emotional tension

The cross-modal gap compares VLM-derived visual terms with title, genre, and overview terms. This is a coarse proxy rather than a semantic entailment model; it is intentionally transparent for a first artifact.

## Validation Plan

The next validation step is human coding on a random poster sample:

- whether the VLM extracted visible objects correctly
- whether the implied question is grounded in the image
- whether missing information and emotional tension are plausible
- whether visual-gap scores correlate with perceived curiosity

