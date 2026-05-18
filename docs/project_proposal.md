# Project Proposal

## Title

Designing Curiosity-Aware Multimodal Recommenders: A Lightweight VLM-Enhanced Reranking Framework for Exploratory Consumption

## Research Question

Can multimodal curiosity signals improve exploratory recommendation quality beyond relevance-based ranking?

## Problem

Recommender systems often prioritize relevance, similarity, and short-term engagement. However, in many consumption settings, users do not only want more of what they already like. They may also value recommendations that are unexpected, curiosity-inducing, and meaningfully different from their past choices.

Existing systems often struggle to operationalize this type of exploratory value, especially when curiosity is triggered by multimodal cues such as posters, visual style, narrative incompleteness, or a semantic gap between what an item appears to be and what it promises.

## Proposed Artifact

This project designs a curiosity-aware reranking artifact. Given a set of candidate items from an existing recommender model, the artifact extracts curiosity-related signals from text and visual metadata, then reranks candidates by jointly considering relevance, curiosity, unexpectedness, and redundancy.

The initial artifact uses text metadata. The next version will use VLMs to extract visual curiosity cues from posters or keyframes.

## Method

1. Build a baseline candidate generator using user-item interaction data.
2. Enrich items with metadata such as overview, genre, poster, and tagline.
3. Extract curiosity signals from item metadata.
4. Rerank candidate items using a weighted objective.
5. Compare the baseline list with the curiosity-aware list.

## Expected Evaluation

The project will use both offline and human evaluation.

Offline metrics include relevance, novelty, diversity, and unexpectedness. Human evaluation will compare whether users perceive curiosity-aware lists as more exploration-inducing while remaining relevant.

## Expected Contribution

The project contributes a design artifact for exploratory recommendation. It also demonstrates a design science pathway from problem formulation to artifact design, implementation, and evaluation.

## Fit With Prior Experience

This project builds on prior experience with recommender systems, multimodal feature extraction, short-video content analysis, and econometric evaluation. It extends that background from measuring digital phenomena to designing and evaluating a digital artifact.

