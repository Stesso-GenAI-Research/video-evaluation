"""Create measurable extraction diagnostics and a worksheet for human review."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import read_clips, read_jsonl_model, write_csv
from .models import ActionTriple, FrameNetMapping, VerbNetMapping


_YES_LABELS = {"yes", "y", "true", "1"}
_NO_LABELS = {"no", "n", "false", "0"}


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sample(rows: list[Any], count: int, rng: random.Random) -> list[Any]:
    if len(rows) <= count:
        return list(rows)
    return rng.sample(rows, count)


def _contains_human_labels(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        return any(
            (row.get(column) or "").strip()
            for row in csv.DictReader(handle)
            for column in ("action_correct", "object_correct", "tool_correct")
        )


def build_quality_review(
    *,
    clips_jsonl: Path,
    month1_dir: Path,
    month2_dir: Path,
    output_dir: Path,
    random_seed: int = 1729,
    examples_per_group: int = 15,
) -> dict[str, Path]:
    """Write an automatic quality summary and a stratified manual-review CSV."""
    clips = read_clips(clips_jsonl)
    triples = read_jsonl_model(month1_dir / "action_object_tool_triples.jsonl", ActionTriple)
    verbnet = read_jsonl_model(month1_dir / "verbnet_mappings.jsonl", VerbNetMapping)
    framenet = read_jsonl_model(month2_dir / "framenet_mappings.jsonl", FrameNetMapping)
    clips_by_id = {clip.clip_id: clip for clip in clips}
    triple_clip_ids = {row.record_id for row in triples if row.record_type == "clip"}

    direct_tools = [row for row in triples if row.tool_lemmas]
    context_tools = [row for row in triples if row.context_tool_lemmas]
    fallback_tools = [
        row for row in triples if not row.tool_lemmas and row.context_tool_lemmas
    ]
    with_objects = [row for row in triples if row.object_lemmas]
    complete = [row for row in triples if row.object_lemmas and row.tool_lemmas]
    missing_object = [row for row in triples if not row.object_lemmas]
    low_confidence = [row for row in triples if row.confidence < 0.6]

    source_fields = Counter(row.source_field for row in triples)
    summary = {
        "clip_count": len(clips),
        "clips_with_extracted_action": len(triple_clip_ids),
        "clips_without_extracted_action": len(clips) - len(triple_clip_ids),
        "clip_action_coverage": _rate(len(triple_clip_ids), len(clips)),
        "triple_count": len(triples),
        "triple_object_coverage": _rate(len(with_objects), len(triples)),
        "direct_tool_coverage": _rate(len(direct_tools), len(triples)),
        "record_tool_context_coverage": _rate(len(context_tools), len(triples)),
        "tool_fallback_available": _rate(len(fallback_tools), len(triples)),
        "complete_direct_triples": len(complete),
        "low_confidence_triples": len(low_confidence),
        "verbnet_mapping_coverage": _rate(
            sum(row.has_mapping for row in verbnet), len(verbnet)
        ),
        "framenet_mapping_coverage": _rate(
            sum(row.has_mapping for row in framenet), len(framenet)
        ),
        "triple_source_field_counts": dict(source_fields.most_common()),
        "interpretation": [
            "Direct tool coverage measures tools grammatically attached to an action.",
            "Record tool context is a separate fallback from clip or step metadata.",
            "The fallback is useful for scoring, but it does not prove which tool belongs to which action.",
            "The manual review worksheet must be labeled before claiming extraction precision.",
        ],
    }

    rng = random.Random(random_seed)
    review_groups: list[tuple[str, list[ActionTriple]]] = [
        ("complete_direct", complete),
        ("metadata_tool_fallback", fallback_tools),
        ("missing_object", missing_object),
        ("low_confidence", low_confidence),
    ]
    review_rows: list[dict[str, Any]] = []
    for group_name, candidates in review_groups:
        for row in _sample(candidates, examples_per_group, rng):
            clip = clips_by_id.get(row.record_id)
            review_rows.append(
                {
                    "review_group": group_name,
                    "clip_id": row.record_id,
                    "clip_title": clip.title if clip else None,
                    "source_field": row.source_field,
                    "sentence": row.sentence,
                    "action_lemma": row.action_lemma,
                    "object_lemmas": ";".join(row.object_lemmas),
                    "direct_tool_lemmas": ";".join(row.tool_lemmas),
                    "record_tool_context": ";".join(row.context_tool_lemmas),
                    "confidence": row.confidence,
                    "action_correct": "",
                    "object_correct": "",
                    "tool_correct": "",
                    "notes": "",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "extraction_quality_summary.json"
    review_path = output_dir / "manual_review_sample.csv"
    preserved_labeled_review = _contains_human_labels(review_path)
    if preserved_labeled_review:
        review_path = output_dir / "manual_review_sample.generated.csv"
    summary["manual_review_rows"] = len(review_rows)
    summary["manual_review_group_counts"] = dict(
        Counter(row["review_group"] for row in review_rows)
    )
    summary["preserved_existing_labeled_review"] = preserved_labeled_review
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(review_path, review_rows)
    return {"quality_summary": summary_path, "manual_review": review_path}


def summarize_manual_review(review_csv: Path, output_json: Path) -> dict[str, Any]:
    """Calculate precision from the completed yes/no review columns."""
    with review_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manual review file has no rows: {review_csv}")

    dimensions = ("action_correct", "object_correct", "tool_correct")

    def summarize(rows_to_score: list[dict[str, str]]) -> dict[str, Any]:
        result: dict[str, Any] = {"row_count": len(rows_to_score)}
        for dimension in dimensions:
            labels: list[bool] = []
            for row_number, row in enumerate(rows_to_score, start=2):
                value = (row.get(dimension) or "").strip().lower()
                if not value:
                    continue
                if value in _YES_LABELS:
                    labels.append(True)
                elif value in _NO_LABELS:
                    labels.append(False)
                else:
                    raise ValueError(
                        f"Invalid {dimension} label {value!r} near CSV row {row_number}; "
                        "use yes, no, or leave it blank."
                    )
            result[dimension] = {
                "labeled": len(labels),
                "correct": sum(labels),
                "precision": _rate(sum(labels), len(labels)) if labels else None,
            }
        return result

    groups = sorted({row.get("review_group", "") for row in rows})
    summary = {
        "source_review_csv": str(review_csv),
        "overall": summarize(rows),
        "by_review_group": {
            group: summarize([row for row in rows if row.get("review_group", "") == group])
            for group in groups
        },
        "note": "Precision is calculated only from nonblank human labels.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
