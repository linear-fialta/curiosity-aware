# Project Proposal

## Title

Visual Information Gaps in Exploratory Recommendation: A VLM-Assisted Reranking Artifact

## Research Question

Can visual information gaps extracted from item images help recommender systems generate recommendations that are relevant enough to understand, but unresolved enough to invite exploration?

## Problem

Recommender systems often prioritize relevance, similarity, and short-term engagement. However, exploration often begins when a user encounters an item that is related to their interests but contains unresolved information. The user recognizes enough to care, but still wants to know more.

Existing serendipity-oriented recommenders usually evaluate whether a recommendation is unexpected and useful after consumption. They less directly explain the pre-click mechanism that motivates a user to explore an unexpected item. This project focuses on that mechanism by operationalizing visual information gaps in posters or keyframes.

The project follows directly from my WITS 2025 paper on condensed clips in algorithmic feeds. That paper studies how dynamic similarity and narrative completeness shape continuance and exploration. This project translates the same theoretical logic into a design artifact: if narrative gaps can motivate exploration in short-video feeds, visual gaps may serve a similar role in poster- or keyframe-based recommendation.

## Proposed Artifact

This project designs a VLM-assisted reranking artifact. Given a set of candidate items from an existing recommender model, a VLM parses each image into structured scene observations such as objects, actions, missing information, incongruity, genre ambiguity, and implied questions. The artifact then computes a visual information gap score and applies VIG-Rerank, a nonlinear listwise reranking algorithm that jointly considers relevance, visual information gap, taste-adjacent novelty, cross-modal gap, and list redundancy.

The VLM does not directly score curiosity. It only produces auditable visual interpretations. The construct score is computed by the artifact.

## Method

1. Build a baseline candidate generator using user-item interaction data.
2. Enrich items with metadata such as overview, genre, poster, and tagline.
3. Use a VLM to parse posters or keyframes into structured visual observations.
4. Compute visual information gap from the parsed observations.
5. Rerank candidate items using VIG-Rerank.
6. Compare against relevance-only, linear reranking, no-visual-gap, and no-listwise baselines.

## Expected Evaluation

The project will use both offline and human evaluation.

Offline metrics include HitRate@K, NDCG@K, baseline relevance retention, novelty, diversity, and visual-gap intensity. Human evaluation will compare whether users perceive visual-gap-aware lists as more exploration-inducing while remaining relevant.

## Expected Contribution

The project contributes a design artifact for exploratory recommendation. The conceptual contribution is to distinguish curiosity as an exploration mechanism from serendipity as a post-consumption outcome. The technical contribution is VIG-Rerank, a transparent VLM-assisted method that operationalizes visual information gaps and optimizes a relevance-novelty sweet spot rather than maximizing novelty monotonically.

## Fit With Prior Experience

This project builds on prior work on condensed clips, dynamic similarity, narrative completeness, and exploratory intention. It extends that work from measuring narrative gaps in algorithmic feeds to designing a multimodal artifact that uses visual gaps for exploratory recommendation.
