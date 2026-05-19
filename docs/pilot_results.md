# Pilot Results

This note records the current offline pilot. It should be read as an implementation check and an early signal, not as a final empirical claim.

## Setup

- Dataset: MovieLens `ml-latest-small`
- Metadata: 9,622 TMDb-enriched movies
- Visual fields: 2,998 Qwen2.5-VL parsed posters
- Users: 100
- Candidates per user: 100
- Evaluation: top-10 leave-one-out ranking
- Matrix-factorization seed: 42
- Bootstrap samples for saved intervals: 500

Run command:

```bash
PYTHONPATH=src python scripts/run_movielens_experiment.py \
  --metadata data/external/tmdb_movies.csv \
  --scenes data/external/vlm_scene_interpretations.jsonl \
  --max-users 100 \
  --candidate-k 100 \
  --top-k 10 \
  --epochs 8 \
  --factors 32 \
  --seed 42 \
  --bootstrap-samples 500
```

## Metrics

| variant | HitRate@10 | NDCG@10 | avg visual gap | avg novelty | genre diversity | visual coverage |
|---|---:|---:|---:|---:|---:|---:|
| `mf_relevance` | 0.02 | 0.0069 | 0.6788 | 0.7834 | 0.7388 | 0.842 |
| `mmr` | 0.02 | 0.0100 | 0.6698 | 0.8475 | 0.9526 | 0.828 |
| `serendipity` | 0.00 | 0.0000 | 0.6412 | 0.9921 | 0.7965 | 0.781 |
| `linear_gap` | 0.03 | 0.0125 | 0.8350 | 0.5905 | 0.7016 | 1.000 |
| `direct_vlm_proxy` | 0.09 | 0.0408 | 0.8680 | 0.7987 | 0.8533 | 1.000 |
| `vig_rerank` | 0.09 | 0.0367 | 0.8179 | 0.5320 | 0.6752 | 1.000 |
| `vig_no_visual` | 0.06 | 0.0273 | 0.0000 | 0.8420 | 0.9621 | 0.000 |
| `vig_no_listwise` | 0.08 | 0.0355 | 0.8168 | 0.5243 | 0.6436 | 1.000 |

Bootstrap intervals are written to:

```text
data/processed/movielens_experiment/summary_metric_intervals.csv
```

## Reading

Visual features carry signal: visual-gap-aware methods outperform MF, MMR, serendipity, and the no-visual ablation on HitRate@10.

Direct visual-gap ranking remains a strong baseline. VIG-Rerank matches its HitRate@10 but slightly trails on NDCG@10, which suggests that the current sweet-spot and listwise terms need further tuning rather than simple removal.

Generic novelty is insufficient. The serendipity baseline has the highest novelty but zero HitRate@10 in this pilot, consistent with the WITS argument that exploration is not monotonic in novelty.

Coverage matters for interpretation. The main visual methods rank only items with parsed poster scenes in this pilot, while non-visual baselines naturally include a mix of parsed and unparsed items. This should be reported as a measurement constraint, not hidden as a preprocessing detail.

## Next Checks

- Repeat the run on more users after expanding VLM coverage.
- Hand-code a small poster sample to validate VLM scene fields.
- Report confidence intervals or bootstrap variation across users.
- Add a human evaluation for perceived curiosity and click intention.
