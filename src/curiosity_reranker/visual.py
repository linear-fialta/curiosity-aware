from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from curiosity_reranker.features import genre_overlap, normalize_score
from curiosity_reranker.schema import VisualSceneInterpretation


def load_visual_interpretations(path: Path) -> dict[str, VisualSceneInterpretation]:
    interpretations: dict[str, VisualSceneInterpretation] = {}
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            scene = VisualSceneInterpretation(
                item_id=str(payload["item_id"]),
                main_objects=tuple(_as_text_list(payload.get("main_objects", []))),
                setting=str(payload.get("setting", "")),
                visible_action=str(payload.get("visible_action", "")),
                occluded_or_missing_information=tuple(
                    _as_text_list(payload.get("occluded_or_missing_information", []))
                ),
                object_context_incongruity=str(payload.get("object_context_incongruity", "")),
                genre_ambiguity=tuple(_as_text_list(payload.get("genre_ambiguity", []))),
                emotional_tension=tuple(_as_text_list(payload.get("emotional_tension", []))),
                implied_question=str(payload.get("implied_question", "")),
            )
            interpretations[scene.item_id] = scene
    return interpretations


def visual_information_gap_score(scene: VisualSceneInterpretation | None) -> float:
    if scene is None:
        return 0.0

    missing_info = normalize_score(len(scene.occluded_or_missing_information) / 3.0)
    unresolved_action = 1.0 if scene.visible_action and scene.implied_question else 0.0
    incongruity = 1.0 if scene.object_context_incongruity else 0.0
    genre_ambiguity = normalize_score(len(scene.genre_ambiguity) / 3.0)
    emotional_tension = normalize_score(len(scene.emotional_tension) / 3.0)

    return normalize_score(
        0.30 * missing_info
        + 0.20 * unresolved_action
        + 0.20 * incongruity
        + 0.15 * genre_ambiguity
        + 0.15 * emotional_tension
    )


def visual_reason(scene: VisualSceneInterpretation | None) -> str:
    if scene is None:
        return "no visual scene interpretation is available"
    if scene.implied_question:
        return f"the image implies an unresolved question: {scene.implied_question}"
    if scene.occluded_or_missing_information:
        return "the image withholds important contextual information"
    if scene.object_context_incongruity:
        return "the image places familiar elements in an unexpected visual context"
    return "the image provides limited visual information gap"


def cross_modal_gap_score(
    scene: VisualSceneInterpretation | None,
    title: str,
    genres: tuple[str, ...],
    overview: str,
) -> float:
    if scene is None:
        return 0.0

    image_terms = _tokens(
        " ".join(
            [
                *scene.main_objects,
                scene.setting,
                scene.visible_action,
                scene.object_context_incongruity,
                *scene.genre_ambiguity,
                *scene.emotional_tension,
                scene.implied_question,
            ]
        )
    )
    text_terms = _tokens(" ".join([title, " ".join(genres), overview]))
    term_gap = 1.0
    if image_terms and text_terms:
        term_gap = 1.0 - (len(image_terms & text_terms) / len(image_terms | text_terms))

    genre_gap = 1.0 - genre_overlap(scene.genre_ambiguity, genres)
    return normalize_score((0.55 * term_gap) + (0.45 * genre_gap))


def attach_visual_interpretations(
    candidates: pd.DataFrame,
    interpretations: dict[str, VisualSceneInterpretation],
) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        item_id = str(row["item_id"])
        scene = interpretations.get(item_id)
        genres = tuple(str(row["genres"]).split("|")) if pd.notna(row["genres"]) else ()
        rows.append(
            {
                **row.to_dict(),
                "visual_information_gap_score": visual_information_gap_score(scene),
                "cross_modal_gap_score": cross_modal_gap_score(
                    scene,
                    title=str(row["title"]),
                    genres=genres,
                    overview=str(row["overview"]),
                ),
                "visual_reason": visual_reason(scene),
            }
        )
    return pd.DataFrame(rows)


def _tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "who",
        "why",
        "will",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _as_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [f"{key}: {val}" for key, val in value.items()]
    if isinstance(value, list | tuple):
        flattened = []
        for item in value:
            if isinstance(item, dict):
                flattened.extend(f"{key}: {val}" for key, val in item.items())
            elif item is not None:
                flattened.append(str(item))
        return flattened
    return [str(value)]
