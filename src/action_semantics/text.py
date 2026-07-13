from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from .models import ClipRecord, StepRecord, TextSegment

_SPACE_RE = re.compile(r"\s+")
_NON_NARRATIVE_INDEXED_VIDEO_METADATA_PREFIXES = (
    # These fields are preserved for provenance and later structured matching,
    # but treating labels and inventories as sentences creates false actions
    # such as "clean" from a category or "scrub" from a brush name.
    "gemini_metadata.source_video",
    # Canonical clip name/description/goal already live in the typed top-level
    # fields. Everything under this metadata node is provenance, duplicate
    # variants, timestamps, or associated inventory; parsing it again would
    # duplicate actions and turn inventory labels into fake sentences.
    "gemini_metadata.clip",
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0000", " ")
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def normalize_term(value: str | None) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9+.#/ -]", " ", text)
    return _SPACE_RE.sub(" ", text).strip()


def flatten_text_values(value: Any, *, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Yield readable text leaves from a nested metadata structure."""
    if value is None:
        return
    if isinstance(value, str):
        text = normalize_text(value)
        if text:
            yield prefix, text
    elif isinstance(value, dict):
        for key, sub_value in value.items():
            key_text = normalize_term(str(key)).replace(" ", "_") or "field"
            new_prefix = f"{prefix}.{key_text}" if prefix else key_text
            yield from flatten_text_values(sub_value, prefix=new_prefix)
    elif isinstance(value, list):
        for index, sub_value in enumerate(value):
            new_prefix = f"{prefix}[{index}]" if prefix else f"item[{index}]"
            yield from flatten_text_values(sub_value, prefix=new_prefix)
    else:
        text = normalize_text(value)
        if text and not text.isnumeric():
            yield prefix, text


def clip_text_segments(clip: ClipRecord, min_length: int = 3) -> list[TextSegment]:
    fields = {
        "title": clip.title,
        "description": clip.description,
        "summary": clip.summary,
        "transcript": clip.transcript,
        "automatic_captions": clip.automatic_captions,
        "whisper_transcript": clip.whisper_transcript,
    }
    segments: list[TextSegment] = []
    seen: set[tuple[str, str]] = set()
    for field, value in fields.items():
        text = normalize_text(value)
        if len(text) >= min_length and (field, text) not in seen:
            segments.append(TextSegment(record_type="clip", record_id=clip.clip_id, source_field=field, text=text))
            seen.add((field, text))
    for field, text in flatten_text_values(clip.gemini_metadata, prefix="gemini_metadata"):
        if field.startswith(_NON_NARRATIVE_INDEXED_VIDEO_METADATA_PREFIXES):
            continue
        if len(text) >= min_length and (field, text) not in seen:
            confidence = 0.9 if "frames" in field or "steps" in field else 0.85
            segments.append(
                TextSegment(
                    record_type="clip",
                    record_id=clip.clip_id,
                    source_field=field,
                    text=text,
                    source_confidence=confidence,
                )
            )
            seen.add((field, text))
    return segments


def step_text_segments(step: StepRecord, min_length: int = 3) -> list[TextSegment]:
    fields: dict[str, Any] = {
        "title": step.title,
        "description": step.description,
        "tools": "; ".join(step.tools),
        "materials": "; ".join(step.materials),
        "techniques": "; ".join(step.techniques),
    }
    segments: list[TextSegment] = []
    for field, value in fields.items():
        text = normalize_text(value)
        if len(text) >= min_length:
            segments.append(TextSegment(record_type="step", record_id=step.step_id, source_field=field, text=text))
    return segments
