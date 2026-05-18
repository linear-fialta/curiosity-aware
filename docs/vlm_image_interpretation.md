# VLM Image Interpretation

## Principle

The VLM should not directly assign a curiosity score. Direct scoring is difficult to audit and easy to dismiss as prompt engineering.

Instead, the VLM is used as a visual scene parser. It converts an image into structured observations. The project code then computes visual information gap from those observations.

## VLM Output Schema

For each poster or keyframe, the VLM should return JSON in this format:

```json
{
  "item_id": "m001",
  "main_objects": ["large unknown structure", "small human figures"],
  "setting": "foggy open field",
  "visible_action": "humans approach an unexplained object",
  "occluded_or_missing_information": [
    "what the structure is",
    "why the humans are approaching it"
  ],
  "object_context_incongruity": "ordinary human figures appear before a nonhuman object",
  "genre_ambiguity": ["science fiction", "mystery", "drama"],
  "emotional_tension": ["uncertainty", "awe", "risk"],
  "implied_question": "What is the object and what will happen when humans interact with it?"
}
```

## Recommended Prompt

```text
You are helping build an exploratory recommender system.

Analyze the image as a visual scene. Do not rate curiosity. Do not recommend the item.

Return only valid JSON with these fields:
- item_id
- main_objects
- setting
- visible_action
- occluded_or_missing_information
- object_context_incongruity
- genre_ambiguity
- emotional_tension
- implied_question

Focus on what a viewer can infer from the image and what remains unresolved.
```

## Why This Is More Than Prompting

Prompting is only the data extraction step. The construct is computed outside the VLM.

```text
image
  -> VLM scene parsing
  -> structured visual observations
  -> deterministic visual information gap function
  -> reranking objective
  -> offline and human evaluation
```

This makes the method more transparent:

- Researchers can inspect the extracted scene fields.
- The scoring rule can be ablated and changed.
- Human coders can validate VLM scene interpretations.
- The artifact can be compared against relevance-only and serendipity-oriented baselines.

