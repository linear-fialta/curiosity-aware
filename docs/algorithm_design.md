# Algorithm Design

## VIG-Rerank

The main algorithm is `VIG-Rerank`, a Visual Information Gap-aware reranker for exploratory recommendation.

The VLM is not the recommender. It is a measurement layer that parses item images into structured scene fields. The algorithmic contribution is the reranking objective and listwise selection procedure.

## Item-Level Objective

For user `u` and candidate item `i`, the item score is:

```text
VIGItemScore(u, i)
= Rel(u, i)^a
  * VIG(i)^b
  * TextGap(i)^c
  * CrossModalGap(i)^d
  * SweetSpot(Distance(u, i))
```

The sweet-spot term is:

```text
SweetSpot(d) = exp(-((d - d*)^2) / (2 * sigma^2))
```

This implements the idea that exploratory recommendation should not maximize novelty monotonically. Items that are too familiar are repetitive, while items that are too distant may be uninterpretable. The best candidates are taste-adjacent.

## Listwise Selection

After computing item-level scores, the algorithm greedily selects items into the final list:

```text
argmax_i VIGItemScore(u, i)
         - lambda * redundancy(i, selected_items)
         + eta * new_category_coverage(i, selected_items)
```

This step prevents the reranker from filling the list with visually intriguing but redundant items.

## Baselines

The repository supports these comparison conditions:

- matrix-factorization relevance baseline
- MMR relevance-diversity baseline
- serendipity-style relevance + unexpectedness baseline
- direct VLM proxy baseline
- linear visual-gap reranker
- VIG-Rerank
- VIG-Rerank without listwise diversity
- VIG-Rerank without visual gap

## Why This Is More Than Calling a VLM

Calling a VLM only produces structured scene observations. The scoring function, sweet-spot transformation, and listwise optimization are implemented in the repository.

The project can therefore evaluate whether visual information gaps improve exploratory recommendation beyond relevance, diversity, or generic serendipity heuristics.
