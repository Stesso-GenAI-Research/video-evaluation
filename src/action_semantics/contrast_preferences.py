"""Generate a large, score-independent controlled preference corpus."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from action_semantics.config import DEFAULT_RANDOM_SEED
from action_semantics.io_utils import read_clips, write_csv, write_jsonl
from action_semantics.models import ActionTriple, ClipRecord
from action_semantics.provenance import build_manifest
from action_semantics.retrieval.clip_resolution import clip_interval
from action_semantics.retrieval.preference_evaluation import prepare_preference_index
from action_semantics.retrieval.scorers import StructuredResources, resources_from_files
from action_semantics.synthetic_preferences import (
    SYNTHETIC_CONFIDENCE,
    _category_name,
    _clip_audit_fields,
    _inventory,
    _timestamp_reference,
)
from action_semantics.text import normalize_text


CONTRAST_GENERATOR_VERSION: Final[str] = "metadata-contrast-preferences-v1"
DEFAULT_CONTRAST_STEP_COUNT: Final[int] = 800
_GENERIC_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "analyze",
        "assess",
        "be",
        "demonstrate",
        "ensure",
        "have",
        "identify",
        "include",
        "introduce",
        "provide",
        "review",
        "show",
        "understand",
        "use",
    }
)
_GENERIC_OBJECTS: Final[frozenset[str]] = frozenset(
    {
        "area",
        "benefit",
        "component",
        "end",
        "example",
        "feature",
        "foot",
        "inch",
        "information",
        "introduction",
        "material",
        "method",
        "need",
        "option",
        "overview",
        "part",
        "piece",
        "process",
        "project",
        "result",
        "side",
        "step",
        "system",
        "thing",
        "time",
        "tool",
        "video",
        "way",
    }
)


@dataclass(frozen=True)
class ContrastTarget:
    clip: ClipRecord
    triple: ActionTriple
    objects: tuple[str, ...]


@dataclass(frozen=True)
class ContrastIndexes:
    targets: tuple[ContrastTarget, ...]
    by_object: dict[str, tuple[ContrastTarget, ...]]
    by_action: dict[str, tuple[ContrastTarget, ...]]


def _reference_eligible(clip: ClipRecord) -> bool:
    start, end = clip_interval(clip)
    return bool(
        clip.video_id is not None
        and normalize_text(clip.title)
        and start is not None
        and end is not None
        and end > start
    )


def _informative_objects(triple: ActionTriple) -> tuple[str, ...]:
    output: list[str] = []
    for value in triple.object_lemmas:
        normalized = normalize_text(value).casefold()
        if (
            not normalized
            or normalized in _GENERIC_OBJECTS
            or len(normalized) < 3
            or re.fullmatch(r"[\d.\-/]+", normalized)
        ):
            continue
        if normalized not in output:
            output.append(normalized)
    return tuple(output)


def _primary_title_target(
    clip: ClipRecord,
    resources: StructuredResources,
) -> ContrastTarget | None:
    candidates: list[tuple[ActionTriple, tuple[str, ...]]] = []
    for triple in resources.triples_for("clip", clip.clip_id):
        objects = _informative_objects(triple)
        if (
            triple.source_field != "title"
            or triple.action_lemma in _GENERIC_ACTIONS
            or not objects
        ):
            continue
        candidates.append((triple, objects))
    if not candidates:
        return None
    triple, objects = max(
        candidates,
        key=lambda item: (
            item[0].confidence,
            len(item[1]),
            bool(item[0].tool_lemmas),
            item[0].action_lemma,
        ),
    )
    return ContrastTarget(clip=clip, triple=triple, objects=objects)


def _build_indexes(
    clips: list[ClipRecord],
    resources: StructuredResources,
) -> ContrastIndexes:
    targets = tuple(
        target
        for clip in clips
        if _reference_eligible(clip)
        if (target := _primary_title_target(clip, resources)) is not None
    )
    by_object_lists: defaultdict[str, list[ContrastTarget]] = defaultdict(list)
    by_action_lists: defaultdict[str, list[ContrastTarget]] = defaultdict(list)
    for target in targets:
        for object_lemma in target.objects:
            by_object_lists[object_lemma].append(target)
        by_action_lists[target.triple.action_lemma].append(target)
    return ContrastIndexes(
        targets=targets,
        by_object={key: tuple(value) for key, value in by_object_lists.items()},
        by_action={key: tuple(value) for key, value in by_action_lists.items()},
    )


def _context(target: ContrastTarget) -> tuple[str, str] | None:
    tools = _inventory(target.clip, "tools")
    if tools:
        return "tool", min(tools, key=lambda value: (len(value), value.casefold()))
    materials = _inventory(target.clip, "supplies")
    if materials:
        return "material", min(
            materials, key=lambda value: (len(value), value.casefold())
        )
    return None


def _all_context_values(target: ContrastTarget) -> set[str]:
    return {
        value.casefold()
        for value in (
            *_inventory(target.clip, "tools"),
            *_inventory(target.clip, "supplies"),
        )
    }


def _deduplicate_targets(values: list[ContrastTarget]) -> list[ContrastTarget]:
    output: list[ContrastTarget] = []
    seen: set[str] = set()
    for value in values:
        if value.clip.clip_id in seen:
            continue
        output.append(value)
        seen.add(value.clip.clip_id)
    return output


def _candidate_pools(
    target: ContrastTarget,
    indexes: ContrastIndexes,
) -> dict[str, list[ContrastTarget]]:
    target_id = target.clip.clip_id
    target_objects = set(target.objects)
    same_object = _deduplicate_targets(
        [candidate for obj in target.objects for candidate in indexes.by_object[obj]]
    )
    action_contrast = [
        candidate
        for candidate in same_object
        if candidate.clip.clip_id != target_id
        and candidate.triple.action_lemma != target.triple.action_lemma
    ]
    object_contrast = [
        candidate
        for candidate in indexes.by_action[target.triple.action_lemma]
        if candidate.clip.clip_id != target_id
        and not target_objects.intersection(candidate.objects)
    ]
    within_video = [
        candidate
        for candidate in indexes.targets
        if candidate.clip.clip_id != target_id
        and candidate.clip.video_id == target.clip.video_id
    ]
    target_context = _context(target)
    context_contrast: list[ContrastTarget] = []
    if target_context is not None:
        context_value = target_context[1].casefold()
        context_contrast = [
            candidate
            for candidate in same_object
            if candidate.clip.clip_id != target_id
            and candidate.triple.action_lemma == target.triple.action_lemma
            and context_value not in _all_context_values(candidate)
        ]
    return {
        "same-object-different-action": action_contrast,
        "same-action-different-object": object_contrast,
        "within-video-adjacent": within_video,
        "context-contrast": context_contrast,
    }


def _candidate_sort_key(
    target: ContrastTarget,
    candidate: ContrastTarget,
    *,
    random_order: dict[str, float],
) -> tuple[Any, ...]:
    target_objects = set(target.objects)
    shared_objects = len(target_objects.intersection(candidate.objects))
    return (
        int(candidate.clip.video_id == target.clip.video_id),
        -int(_category_name(candidate.clip) == _category_name(target.clip)),
        -shared_objects,
        random_order[candidate.clip.clip_id],
        candidate.clip.clip_id,
    )


def _closest_within_video(
    target: ContrastTarget,
    candidates: list[ContrastTarget],
) -> ContrastTarget | None:
    target_start, target_end = clip_interval(target.clip)
    if target_start is None or target_end is None:
        return None
    midpoint = (target_start + target_end) / 2

    def distance(candidate: ContrastTarget) -> tuple[float, str]:
        start, end = clip_interval(candidate.clip)
        if start is None or end is None:
            return float("inf"), candidate.clip.clip_id
        return abs(((start + end) / 2) - midpoint), candidate.clip.clip_id

    return min(candidates, key=distance, default=None)


def _selected_specs(
    target: ContrastTarget,
    indexes: ContrastIndexes,
    *,
    random_order: dict[str, float],
) -> list[tuple[str, ContrastTarget]]:
    pools = _candidate_pools(target, indexes)
    selected: list[tuple[str, ContrastTarget]] = []
    used: set[str] = set()
    for pair_type in (
        "same-object-different-action",
        "same-action-different-object",
        "within-video-adjacent",
        "context-contrast",
    ):
        candidates = [
            candidate
            for candidate in pools[pair_type]
            if candidate.clip.clip_id not in used
        ]
        if pair_type == "within-video-adjacent":
            chosen = _closest_within_video(target, candidates)
        else:
            candidates.sort(
                key=lambda candidate: _candidate_sort_key(
                    target,
                    candidate,
                    random_order=random_order,
                )
            )
            chosen = candidates[0] if candidates else None
        if chosen is not None:
            selected.append((pair_type, chosen))
            used.add(chosen.clip.clip_id)
    return selected


def _select_targets(
    indexes: ContrastIndexes,
    *,
    step_count: int,
    seed: int,
    random_order: dict[str, float],
) -> list[ContrastTarget]:
    eligible = [
        target
        for target in indexes.targets
        if len(_selected_specs(target, indexes, random_order=random_order)) >= 2
    ]
    by_category: defaultdict[str, list[ContrastTarget]] = defaultdict(list)
    for target in eligible:
        by_category[_category_name(target.clip) or "Uncategorized"].append(target)
    rng = random.Random(seed)
    categories = sorted(by_category)
    rng.shuffle(categories)
    for targets in by_category.values():
        targets.sort(key=lambda value: value.clip.clip_id)
        rng.shuffle(targets)

    selected: list[ContrastTarget] = []
    used_videos: set[str] = set()
    deferred: defaultdict[str, list[ContrastTarget]] = defaultdict(list)
    while len(selected) < step_count:
        progressed = False
        for category in categories:
            while by_category[category]:
                candidate = by_category[category].pop()
                video_id = str(candidate.clip.video_id)
                if video_id in used_videos:
                    deferred[category].append(candidate)
                    continue
                selected.append(candidate)
                used_videos.add(video_id)
                progressed = True
                break
            if len(selected) == step_count:
                break
        if not progressed:
            break
    while len(selected) < step_count:
        progressed = False
        for category in categories:
            if not deferred[category]:
                continue
            selected.append(deferred[category].pop())
            progressed = True
            if len(selected) == step_count:
                break
        if not progressed:
            break
    if len(selected) < step_count:
        raise ValueError(
            f"Requested {step_count} contrast steps, but only {len(selected)} clips "
            "support at least two independent contrast types"
        )
    return selected


def _query_title(target: ContrastTarget) -> str:
    action = normalize_text(target.triple.action_text) or target.triple.action_lemma
    object_text = normalize_text(target.triple.object_text)
    if not object_text:
        object_text = " ".join(target.objects)
    return normalize_text(f"{action} {object_text}")


def _write_notice(
    path: Path,
    *,
    step_count: int,
    pair_count: int,
    seed: int,
) -> None:
    path.write_text(
        f"""# Metadata-contrast pairwise development corpus

Generator: `{CONTRAST_GENERATOR_VERSION}`

This directory contains {step_count} constructed action/object steps and
{pair_count} fabricated A/B labels (seed {seed}). Clip references point to the
real canonical sample. Steps and preference labels are not human observations.

Pair construction is independent of lexical, structured, and hybrid scores.
Candidates are selected from parsed title metadata using four declared rules:
same object with a different action, same action with a different object,
adjacent clips in one source video, and same action/object with different
tool/material context when available. The clip that supplied the step action,
object, and optional context is assigned as the winner.

Use this corpus for development benchmarks, diagnostic case sampling, and
annotation-protocol preparation. Do not merge it with authentic W25 judgments,
describe it as human data, or use its aggregate agreement rates as an unbiased
estimate of production performance.
""",
        encoding="utf-8",
    )


def generate_contrast_preferences(
    *,
    index: Path,
    output_dir: Path,
    step_count: int = DEFAULT_CONTRAST_STEP_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Generate a large score-independent contrast corpus."""
    if step_count < 2:
        raise ValueError("step_count must be at least 2")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "steps": output_dir / "steps.jsonl",
        "pairs": output_dir / "pairwise.jsonl",
        "audit": output_dir / "pair_audit.csv",
        "notice": output_dir / "DATA_NOTICE.md",
        "manifest": output_dir / "generation_manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "Refusing to overwrite contrast preference artifacts: "
            + ", ".join(existing)
        )

    index_artifacts = prepare_preference_index(index, output_dir)
    clips = read_clips(index_artifacts.clips_jsonl)
    resources = resources_from_files(index_artifacts.month1_dir, index_artifacts.month2_dir)
    indexes = _build_indexes(clips, resources)
    rng = random.Random(seed)
    random_order = {
        target.clip.clip_id: rng.random()
        for target in sorted(indexes.targets, key=lambda value: value.clip.clip_id)
    }
    targets = _select_targets(
        indexes,
        step_count=step_count,
        seed=seed,
        random_order=random_order,
    )

    step_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    pair_type_occurrences: Counter[str] = Counter()
    pair_number = 0
    for step_number, target in enumerate(targets, start=1):
        step_id = f"metadata-contrast-step-{step_number:04d}"
        context = _context(target)
        specs = _selected_specs(target, indexes, random_order=random_order)
        has_context_contrast = any(pair_type == "context-contrast" for pair_type, _ in specs)
        tools = [context[1]] if context and context[0] == "tool" and has_context_contrast else []
        materials = (
            [context[1]]
            if context and context[0] == "material" and has_context_contrast
            else []
        )
        title = _query_title(target)
        step_rows.append(
            {
                "id": step_id,
                "title": title,
                "description": None,
                "tools": tools,
                "materials": materials,
                "synthetic": True,
                "development_only": True,
                "synthetic_generator": CONTRAST_GENERATOR_VERSION,
                "synthetic_target_clip_id": target.clip.clip_id,
                "synthetic_source_category": _category_name(target.clip),
                "controlled_action": target.triple.action_lemma,
                "controlled_objects": list(target.objects),
                "controlled_context": context[1] if context and has_context_contrast else None,
                "pair_selection_uses_retrieval_scores": False,
            }
        )
        for pair_type, distractor in specs:
            pair_number += 1
            pair_type_occurrences[pair_type] += 1
            winner_position = "A" if pair_type_occurrences[pair_type] % 2 else "B"
            target_reference = (
                _timestamp_reference(target.clip)
                if pair_number % 2
                else {"clip_id": target.clip.clip_id}
            )
            distractor_reference = (
                {"clip_id": distractor.clip.clip_id}
                if pair_number % 2
                else _timestamp_reference(distractor.clip)
            )
            if winner_position == "A":
                clip_a_reference, clip_b_reference = target_reference, distractor_reference
                clip_a, clip_b = target.clip, distractor.clip
            else:
                clip_a_reference, clip_b_reference = distractor_reference, target_reference
                clip_a, clip_b = distractor.clip, target.clip
            comparison_id = f"metadata-contrast-pair-{pair_number:05d}"
            provenance = f"synthetic_metadata_{pair_type.replace('-', '_')}_v1"
            rationale = (
                "Constructed label: the winner supplied the query action, object, "
                "and requested context; the alternative differs on the declared "
                "contrast dimension."
            )
            pair_rows.append(
                {
                    "comparison_id": comparison_id,
                    "step_id": step_id,
                    "clip_a": clip_a_reference,
                    "clip_b": clip_b_reference,
                    "winner_position": winner_position,
                    "winner_clip_id": target.clip.clip_id,
                    "judge_provenance": provenance,
                    "judge_confidence": SYNTHETIC_CONFIDENCE,
                    "judge_id": CONTRAST_GENERATOR_VERSION,
                    "synthetic": True,
                    "development_only": True,
                    "synthetic_pair_type": pair_type,
                    "pair_selection_uses_retrieval_scores": False,
                    "adjudication_basis": "metadata contrast rule; no human review",
                    "synthetic_label_rationale": rationale,
                }
            )
            audit_rows.append(
                {
                    "comparison_id": comparison_id,
                    "step_id": step_id,
                    "step_title": title,
                    "controlled_action": target.triple.action_lemma,
                    "controlled_objects": "; ".join(target.objects),
                    "controlled_context": (
                        context[1] if context and has_context_contrast else None
                    ),
                    "pair_type": pair_type,
                    **_clip_audit_fields("clip_a", clip_a),
                    **_clip_audit_fields("clip_b", clip_b),
                    "winner_position": winner_position,
                    "winner_clip_id": target.clip.clip_id,
                    "judge_provenance": provenance,
                    "judge_confidence": SYNTHETIC_CONFIDENCE,
                    "adjudication_basis": "metadata contrast rule; no human review",
                    "synthetic_label_rationale": rationale,
                }
            )

    write_jsonl(paths["steps"], step_rows)
    write_jsonl(paths["pairs"], pair_rows)
    write_csv(paths["audit"], audit_rows)
    _write_notice(
        paths["notice"],
        step_count=len(step_rows),
        pair_count=len(pair_rows),
        seed=seed,
    )
    manifest = build_manifest(
        command="generate-contrast-pairs",
        input_files=[
            index_artifacts.clips_jsonl,
            index_artifacts.month1_dir / "action_object_tool_triples.jsonl",
        ],
        output_files=[paths["steps"], paths["pairs"], paths["audit"], paths["notice"]],
        parameters={
            "generator_version": CONTRAST_GENERATOR_VERSION,
            "synthetic": True,
            "development_only": True,
            "human_judgments": False,
            "research_evidence": False,
            "seed": seed,
            "step_count": len(step_rows),
            "pair_count": len(pair_rows),
            "pair_type_counts": dict(sorted(pair_type_occurrences.items())),
            "target_video_count": len(
                {str(target.clip.video_id) for target in targets}
            ),
            "target_category_count": len(
                {_category_name(target.clip) for target in targets}
            ),
            "selection_uses_retrieval_scores": False,
            "winner_position_policy": "alternating within each pair type",
            "label_rule": "clip supplying step metadata wins declared contrast",
            "judge_confidence": SYNTHETIC_CONFIDENCE,
            "index_root": str(index_artifacts.root),
        },
    )
    manifest["schema_version"] = "metadata-contrast-preference-generation.v1"
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**paths, "index_root": index_artifacts.root}
