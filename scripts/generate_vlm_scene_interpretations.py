from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCENE_PROMPT = """You are helping construct variables for an exploratory recommender system.

Analyze the image as a visual scene. Do not rate curiosity. Do not recommend the item.

Return only valid JSON with these fields:
- main_objects: list of up to 5 visible objects, characters, or symbols
- setting: short description of the visual setting
- visible_action: what action or relation is visible
- occluded_or_missing_information: list of up to 5 important unresolved visual information items
- object_context_incongruity: one sentence if objects appear in an unexpected context, otherwise empty string
- genre_ambiguity: list of up to 3 plausible genres suggested by the image
- emotional_tension: list of up to 3 emotions or tensions implied by the image
- implied_question: one question a viewer may want answered after seeing the image

Do not repeat list items. Do not wrap the JSON in markdown fences. Focus on what is visible in the image and what remains unresolved."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate VLM scene interpretations from poster URLs or image paths."
    )
    parser.add_argument("--input", required=True, help="CSV with item_id/movieId and poster_url/image_path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output instead of resuming from it.",
    )
    args = parser.parse_args()

    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Install VLM dependencies first: pip install -e '.[vlm]'."
        ) from exc

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = pd.read_csv(input_path)
    if args.limit is not None:
        rows = rows.head(args.limit)
    completed_item_ids = set() if args.overwrite else _load_completed_item_ids(output_path)
    if completed_item_ids:
        print(f"Resuming from {output_path}; skipping {len(completed_item_ids)} completed items.")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.overwrite else "a"
    with output_path.open(mode) as file:
        for _, row in rows.iterrows():
            item_id = _get_item_id(row)
            if item_id in completed_item_ids:
                continue

            image_ref = str(row.get("poster_url") or row.get("image_path"))
            if not image_ref or image_ref == "nan":
                continue

            try:
                payload = interpret_image(
                    item_id=item_id,
                    image_ref=image_ref,
                    model=model,
                    processor=processor,
                    process_vision_info=process_vision_info,
                )
            except Exception as exc:
                print(f"Skipping {item_id} after VLM error: {exc}")
                continue
            file.write(json.dumps(payload, ensure_ascii=True) + "\n")
            file.flush()
            completed_item_ids.add(item_id)
            print(f"Wrote scene interpretation for {item_id}")


def interpret_image(
    item_id: str,
    image_ref: str,
    model,
    processor,
    process_vision_info,
) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": SCENE_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=False,
        repetition_penalty=1.08,
    )
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    payload = _extract_json(output_text)
    payload["item_id"] = item_id
    return payload


def _extract_json(text: str) -> dict:
    text = _strip_markdown_fence(text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON: {text[:200]}")
    payload = json.loads(text[start : end + 1])
    return _normalize_payload(payload)


def _strip_markdown_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _get_item_id(row: pd.Series) -> str:
    if "item_id" in row and pd.notna(row["item_id"]):
        return str(row["item_id"])
    if "movieId" in row and pd.notna(row["movieId"]):
        return str(row["movieId"])
    raise KeyError("Input CSV must include item_id or movieId.")


def _load_completed_item_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = payload.get("item_id")
            if item_id is not None:
                completed.add(str(item_id))
    return completed


def _normalize_payload(payload: dict) -> dict:
    list_fields = [
        "main_objects",
        "occluded_or_missing_information",
        "genre_ambiguity",
        "emotional_tension",
    ]
    text_fields = [
        "setting",
        "visible_action",
        "object_context_incongruity",
        "implied_question",
    ]
    normalized = {}
    for field in list_fields:
        value = payload.get(field, [])
        normalized[field] = value if isinstance(value, list) else [str(value)]
    for field in text_fields:
        normalized[field] = str(payload.get(field, ""))
    return normalized


if __name__ == "__main__":
    main()
