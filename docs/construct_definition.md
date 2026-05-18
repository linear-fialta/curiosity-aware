# Construct Definition

## Core Distinction

This project distinguishes curiosity from serendipity.

Curiosity is a pre-click psychological mechanism. It explains why a user becomes motivated to explore an item after seeing a recommendation.

Serendipity is a post-consumption recommendation outcome. It describes a recommendation that turns out to be unexpected and useful after the user engages with it.

In short:

```text
curiosity = mechanism that motivates exploration
serendipity = outcome of a successful unexpected discovery
```

This project does not try to replace serendipity metrics. Instead, it asks whether visual information gaps can be designed into a reranking artifact as a mechanism that increases exploratory behavior.

## Definition of Curiosity

Curiosity is defined as the motivation to resolve a perceived information gap.

In recommender systems, this gap is productive only when the item is neither too familiar nor too alien. The user must recognize enough to care, but still perceive something unresolved.

Therefore:

```text
curiosity is high when an item is:
1. relevant enough to be interpretable,
2. unfamiliar enough to avoid repetition,
3. unresolved enough to invite exploration.
```

## Visual Information Gap

Visual information gap is the image-level manifestation of curiosity. It captures unresolved, incomplete, or incongruent visual information that may motivate a user to click, inspect, or search for more.

This project operationalizes visual information gap through five components:

- `missing_information`: important visual context is absent or occluded.
- `unresolved_action`: the image shows an action whose outcome is unclear.
- `object_context_incongruity`: objects or characters appear in an unexpected setting.
- `genre_ambiguity`: the image makes the item's category uncertain.
- `emotional_tension`: facial expressions, composition, color, or scene structure imply conflict.

## Relationship to Prior Work

The project builds on the logic of the prior WITS paper on condensed clips:

- Moderate similarity creates a balance between processing fluency and novelty.
- Narrative incompleteness creates gaps that may motivate exploration.
- Exploratory intention is distinct from passive continuance.

This project extends that logic from text/narrative sequence to multimodal recommendation design:

```text
narrative gap in condensed clips
        |
        v
visual information gap in posters/keyframes
        |
        v
exploration-aware reranking artifact
```

## Design Proposition

For exploratory recommendation, items should not be reranked only by relevance. They should be reranked by whether they sit in a productive zone:

```text
interpretable enough + visually unresolved enough + not too redundant
```

The expected contribution is a design artifact that uses VLM-assisted visual interpretation to operationalize this zone.

