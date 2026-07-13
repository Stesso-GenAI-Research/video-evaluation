from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .extraction.triples import extract_triples, triple_dict_for_analysis
from .extraction.verbnet import map_triple_verbs
from .io_utils import read_clips, read_steps, write_csv, write_jsonl
from .models import ActionTriple, ClipRecord, StepRecord, TextSegment
from .provenance import build_manifest, write_manifest
from .text import clip_text_segments, normalize_term, step_text_segments


_INVENTORY_STOPWORDS = {"a", "an", "and", "or", "the", "with"}


def _inventory_terms(values: list[str]) -> list[str]:
    """Keep both full inventory phrases and useful individual words."""
    terms: set[str] = set()
    for value in values:
        normalized = normalize_term(value)
        if not normalized:
            continue
        terms.add(normalized)
        terms.update(
            token
            for token in normalized.split()
            if len(token) > 1 and token not in _INVENTORY_STOPWORDS
        )
    return sorted(terms)


def _clip_inventory(clip: ClipRecord) -> tuple[list[str], list[str]]:
    metadata = clip.gemini_metadata.get("clip", {})
    if not isinstance(metadata, dict):
        return [], []
    tools = metadata.get("tools", [])
    supplies = metadata.get("supplies", [])

    def with_alternatives(values: Any, item_key: str) -> list[str]:
        output = list(values) if isinstance(values, list) else []
        items = metadata.get(item_key, [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("name"), str):
                    output.append(item["name"])
                alternatives = item.get("alternatives", [])
                if isinstance(alternatives, list):
                    output.extend(
                        value for value in alternatives if isinstance(value, str)
                    )
        return output

    return (
        _inventory_terms(with_alternatives(tools, "tool_items")),
        _inventory_terms(with_alternatives(supplies, "supply_items")),
    )


def add_record_inventories(
    triples: list[ActionTriple],
    clips: list[ClipRecord],
    steps: list[StepRecord],
) -> list[ActionTriple]:
    """Attach record-level tools/materials without calling them direct dependencies."""
    inventories: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for clip in clips:
        inventories[("clip", clip.clip_id)] = _clip_inventory(clip)
    for step in steps:
        inventories[("step", step.step_id)] = (
            _inventory_terms(step.tools),
            _inventory_terms(step.materials),
        )
    return [
        triple.model_copy(
            update={
                "context_tool_lemmas": inventories.get(
                    (triple.record_type, triple.record_id), ([], [])
                )[0],
                "context_material_lemmas": inventories.get(
                    (triple.record_type, triple.record_id), ([], [])
                )[1],
            }
        )
        for triple in triples
    ]


def build_segments(
    clips: list[ClipRecord],
    steps: list[StepRecord],
    min_text_length: int,
) -> list[TextSegment]:
    rows: list[TextSegment] = []
    for clip in clips:
        rows.extend(clip_text_segments(clip, min_text_length))
    for step in steps:
        rows.extend(step_text_segments(step, min_text_length))
    return rows


def month1_summary(triples: list[ActionTriple]) -> dict[str, Any]:
    verbs = Counter(triple.action_lemma for triple in triples)
    by_type = Counter(triple.record_type for triple in triples)
    with_object = sum(1 for triple in triples if triple.object_lemmas)
    with_tool = sum(1 for triple in triples if triple.tool_lemmas)
    return {
        "triple_count": len(triples),
        "record_type_counts": dict(by_type),
        "unique_action_lemmas": len(verbs),
        "top_action_lemmas": verbs.most_common(50),
        "triples_with_object": with_object,
        "triples_with_tool": with_tool,
        "object_coverage": with_object / len(triples) if triples else 0.0,
        "tool_coverage": with_tool / len(triples) if triples else 0.0,
    }


def run_month1(
    *,
    clips_jsonl: Path,
    steps_jsonl: Path | None,
    config: PipelineConfig,
) -> dict[str, Path]:
    month_dir = config.ensure_output_dir() / "month1"
    month_dir.mkdir(parents=True, exist_ok=True)

    clips = read_clips(clips_jsonl)
    if config.clip_limit is not None:
        clips = clips[: config.clip_limit]
    steps = read_steps(steps_jsonl) if steps_jsonl is not None else []

    segments = build_segments(clips, steps, config.min_text_length)
    triples = add_record_inventories(
        extract_triples(segments, config.spacy_model), clips, steps
    )
    verbnet_rows = map_triple_verbs(triples)

    segments_path = month_dir / "text_segments.jsonl"
    triples_path = month_dir / "action_object_tool_triples.jsonl"
    triples_csv_path = month_dir / "action_object_tool_triples.csv"
    verbnet_path = month_dir / "verbnet_mappings.jsonl"
    summary_path = month_dir / "month1_summary.json"
    manifest_path = month_dir / "manifest.json"

    write_jsonl(segments_path, segments)
    write_jsonl(triples_path, triples)
    write_csv(
        triples_csv_path,
        [triple_dict_for_analysis(triple) for triple in triples],
        fieldnames=[
            *ActionTriple.model_fields.keys(),
            "object_lemmas_joined",
            "tool_lemmas_joined",
            "context_tool_lemmas_joined",
            "material_lemmas_joined",
            "context_material_lemmas_joined",
        ],
    )
    write_jsonl(verbnet_path, verbnet_rows)
    summary_path.write_text(
        json.dumps(month1_summary(triples), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    output_files = [segments_path, triples_path, triples_csv_path, verbnet_path, summary_path]
    write_manifest(
        manifest_path,
        build_manifest(
            command="run-month1",
            input_files=[path for path in [clips_jsonl, steps_jsonl] if path is not None],
            output_files=output_files,
            parameters={
                "spacy_model": config.spacy_model,
                "min_text_length": config.min_text_length,
                "clip_limit": config.clip_limit,
                "clip_rows_used": len(clips),
                "step_rows_used": len(steps),
            },
        ),
    )
    return {
        "month_dir": month_dir,
        "segments": segments_path,
        "triples": triples_path,
        "triples_csv": triples_csv_path,
        "verbnet": verbnet_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }
