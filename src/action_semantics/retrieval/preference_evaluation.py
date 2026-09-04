"""Frozen evaluation of already-judged pairwise clip preferences.

This path deliberately scores the two supplied clips directly.  It does not
retrieve challengers, construct a top-k list, or tune any retrieval setting.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from action_semantics.config import DEFAULT_RANDOM_SEED, DEFAULT_SPACY_MODEL
from action_semantics.io_utils import iter_jsonl, read_clips, sha256_file, write_jsonl
from action_semantics.index_freshness import index_staleness_reasons
from action_semantics.models import ActionTriple, StepRecord
from action_semantics.month1 import add_record_inventories
from action_semantics.provenance import build_manifest, write_manifest
from action_semantics.retrieval.clip_resolution import (
    DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    ClipReference,
    ClipResolutionResult,
    ClipResolver,
)
from action_semantics.retrieval.evaluation import clustered_bootstrap_mean_ci
from action_semantics.retrieval.lexical import PRODUCTION_TFIDF_SETTINGS, TfidfIndex
from action_semantics.retrieval.provenance import build_retrieval_provenance
from action_semantics.retrieval.scorers import (
    STRUCTURED_SCORER_VERSION,
    StructuredResources,
    StructuredWeights,
    resources_from_files,
    score_query_clip_ids,
)
from action_semantics.retrieval.search import parse_query_triples
from action_semantics.sample_analysis import run_indexed_video_analysis
from action_semantics.text import normalize_text


PREFERENCE_SCHEMA_VERSION = "preference-pair-scores.v1"
SUMMARY_SCHEMA_VERSION = "preference-evaluation-summary.v1"
VALIDATION_SCHEMA_VERSION = "preference-evaluation-validation.v1"
STEP_QUERY_RULE_VERSION = "step-title-description-tools-materials-v1"
FROZEN_HYBRID_ALPHA = 0.5
FROZEN_MIN_TAXONOMY_SUPPORT = 2
DEFAULT_BOOTSTRAP_ITERATIONS = 5000
METHODS = ("lexical", "structured", "hybrid")


class PreferenceStepInput(BaseModel):
    """Typed W25 step view without changing the legacy ``StepRecord`` contract."""

    model_config = ConfigDict(extra="allow", strict=True)

    step_id: str = Field(validation_alias=AliasChoices("step_id", "id"))
    title: str | None
    description: str | None
    tools: list[str]
    materials: list[str]
    source_row_sha256: str | None = None

    @model_validator(mode="before")
    @classmethod
    def identifier_aliases_agree(cls, value: Any) -> Any:
        if (
            isinstance(value, dict)
            and "id" in value
            and "step_id" in value
            and value["id"] != value["step_id"]
        ):
            raise ValueError("id and step_id must agree when both are supplied")
        return value

    @field_validator("step_id")
    @classmethod
    def step_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("step_id/id must not be blank")
        return value.strip()

    @field_validator("title", "description")
    @classmethod
    def optional_text_is_normalized(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_text(value)
        return normalized or None

    @field_validator("tools", "materials")
    @classmethod
    def inventory_values_are_not_blank(cls, values: list[str]) -> list[str]:
        normalized = [normalize_text(value) for value in values]
        if any(not value for value in normalized):
            raise ValueError("inventory values must be nonblank strings")
        return normalized


class PreferencePairInput(BaseModel):
    """One already-judged comparison in the current preference contract."""

    model_config = ConfigDict(extra="allow", strict=True)

    comparison_id: str = Field(validation_alias=AliasChoices("comparison_id", "id"))
    step_id: str
    clip_a: ClipReference
    clip_b: ClipReference
    winner_position: Literal["A", "B"] = Field(
        validation_alias=AliasChoices("winner_position", "winner")
    )
    winner_clip_id: str | None = None
    judge_provenance: str
    judge_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_id: str | None = None
    source_row_sha256: str | None = None

    @model_validator(mode="before")
    @classmethod
    def aliases_agree(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if (
            "id" in value
            and "comparison_id" in value
            and value["id"] != value["comparison_id"]
        ):
            raise ValueError(
                "id and comparison_id must agree when both are supplied"
            )
        if (
            "winner" in value
            and "winner_position" in value
            and value["winner"] != value["winner_position"]
        ):
            raise ValueError(
                "winner and winner_position must agree when both are supplied"
            )
        return value

    @field_validator(
        "comparison_id",
        "step_id",
        "judge_provenance",
        "winner_clip_id",
        "judge_id",
    )
    @classmethod
    def strings_are_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("value must not be blank")
        return value.strip()

    @field_validator("judge_confidence", mode="before")
    @classmethod
    def confidence_is_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("judge_confidence must be a number from 0 to 1")
        return float(value)

    @model_validator(mode="after")
    def clips_are_distinct(self) -> "PreferencePairInput":
        if self.clip_a.reference_key() == self.clip_b.reference_key():
            raise ValueError("clip_a and clip_b must reference different clips")
        return self


@dataclass(frozen=True)
class IndexArtifacts:
    root: Path
    clips_jsonl: Path
    month1_dir: Path
    month2_dir: Path
    source_kind: Literal["nested_jsonl", "built_index_directory"]
    built_for_evaluation: bool
    rebuilt_this_run: bool


@dataclass(frozen=True)
class ParsedStepQuery:
    text: str
    components: dict[str, Any]
    triples: list[ActionTriple]
    diagnostics: dict[str, Any]


def build_step_query(step: PreferenceStepInput) -> tuple[str, dict[str, Any]]:
    """Apply the one frozen Step-1 query construction rule."""
    title = normalize_text(step.title)
    description = normalize_text(step.description)
    tools = [normalize_text(value) for value in step.tools if normalize_text(value)]
    materials = [
        normalize_text(value) for value in step.materials if normalize_text(value)
    ]
    ordered_parts = [
        value
        for value in (
            title,
            description,
            f"Tools: {', '.join(tools)}" if tools else "",
            f"Materials: {', '.join(materials)}" if materials else "",
        )
        if value
    ]
    query_text = " ".join(ordered_parts)
    if not query_text:
        raise ValueError(
            f"step {step.step_id!r} has no title, description, tools, or materials"
        )
    return query_text, {
        "rule_version": STEP_QUERY_RULE_VERSION,
        "field_order": ["title", "description", "tools", "materials"],
        "title": title or None,
        "description": description or None,
        "tools": tools,
        "materials": materials,
        "join": "single spaces; inventory prefixes are 'Tools:' and 'Materials:'",
    }


def exact_pair_outcome(
    score_a: float,
    score_b: float,
    winner_position: Literal["A", "B"],
) -> tuple[Literal["A", "B", "tie"], float]:
    """Return the exact score winner and Step-1 agreement credit."""
    if score_a == score_b:
        return "tie", 0.5
    predicted: Literal["A", "B"] = "A" if score_a > score_b else "B"
    return predicted, 1.0 if predicted == winner_position else 0.0


def _read_typed_rows(
    path: Path,
    model_type: type[PreferenceStepInput] | type[PreferencePairInput],
    *,
    input_name: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    rows: list[Any] = []
    errors: list[dict[str, Any]] = []
    try:
        for record_number, value in enumerate(iter_jsonl(path), start=1):
            try:
                rows.append(model_type.model_validate(value))
            except ValidationError as exc:
                errors.append(
                    {
                        "code": "schema_validation_error",
                        "input": input_name,
                        "record_number": record_number,
                        "message": str(exc),
                    }
                )
    except (OSError, ValueError) as exc:
        errors.append(
            {
                "code": "jsonl_read_error",
                "input": input_name,
                "message": str(exc),
            }
        )
    if not rows and not errors:
        errors.append(
            {
                "code": "empty_input",
                "input": input_name,
                "message": f"{path} contains no JSON records",
            }
        )
    return rows, errors


def _duplicate_errors(
    values: list[str],
    *,
    input_name: str,
    field_name: str,
) -> list[dict[str, Any]]:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if not duplicates:
        return []
    return [
        {
            "code": "duplicate_identifier",
            "input": input_name,
            "field": field_name,
            "values": duplicates,
            "message": f"duplicate {field_name} values: {duplicates}",
        }
    ]


def _load_and_validate_inputs(
    steps_jsonl: Path,
    pairs_jsonl: Path,
) -> tuple[list[PreferenceStepInput], list[PreferencePairInput], list[dict[str, Any]]]:
    steps, step_errors = _read_typed_rows(
        steps_jsonl, PreferenceStepInput, input_name="steps"
    )
    pairs, pair_errors = _read_typed_rows(
        pairs_jsonl, PreferencePairInput, input_name="pairs"
    )
    errors = [*step_errors, *pair_errors]
    errors.extend(
        _duplicate_errors(
            [row.step_id for row in steps],
            input_name="steps",
            field_name="step_id",
        )
    )
    errors.extend(
        _duplicate_errors(
            [row.comparison_id for row in pairs],
            input_name="pairs",
            field_name="comparison_id",
        )
    )
    step_ids = {row.step_id for row in steps}
    for pair in pairs:
        if pair.step_id not in step_ids:
            errors.append(
                {
                    "code": "unknown_step_id",
                    "input": "pairs",
                    "comparison_id": pair.comparison_id,
                    "step_id": pair.step_id,
                    "message": f"unknown step_id {pair.step_id!r}",
                }
            )
    for step in steps:
        try:
            build_step_query(step)
        except ValueError as exc:
            errors.append(
                {
                    "code": "empty_step_query",
                    "input": "steps",
                    "step_id": step.step_id,
                    "message": str(exc),
                }
            )
    return steps, pairs, errors


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _required_index_paths(index_root: Path) -> tuple[Path, Path, Path]:
    clips_jsonl = index_root / "input" / "indexed_video_clips.jsonl"
    month1_dir = index_root / "month1"
    month2_dir = index_root / "month2"
    required = [
        clips_jsonl,
        month1_dir / "action_object_tool_triples.jsonl",
        month1_dir / "verbnet_mappings.jsonl",
        month2_dir / "framenet_mappings.jsonl",
        month2_dir / "diy_actionnet_v1.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "Built index directory is missing required artifacts: " + ", ".join(missing)
        )
    return clips_jsonl, month1_dir, month2_dir


def prepare_preference_index(index: Path, output_dir: Path) -> IndexArtifacts:
    """Use a built production index or build one from nested IndexedVideo JSONL."""
    if index.is_dir():
        clips_jsonl, month1_dir, month2_dir = _required_index_paths(index)
        return IndexArtifacts(
            root=index,
            clips_jsonl=clips_jsonl,
            month1_dir=month1_dir,
            month2_dir=month2_dir,
            source_kind="built_index_directory",
            built_for_evaluation=False,
            rebuilt_this_run=False,
        )
    if not index.is_file():
        raise ValueError(f"Index input does not exist: {index}")
    index_root = output_dir / "index"
    rebuild_reasons = index_staleness_reasons(
        source_jsonl=index,
        output_dir=index_root,
        spacy_model=DEFAULT_SPACY_MODEL,
    )
    if rebuild_reasons:
        run_indexed_video_analysis(
            indexed_videos_jsonl=index,
            output_dir=index_root,
            spacy_model=DEFAULT_SPACY_MODEL,
            random_seed=DEFAULT_RANDOM_SEED,
            min_taxonomy_support=FROZEN_MIN_TAXONOMY_SUPPORT,
        )
    clips_jsonl, month1_dir, month2_dir = _required_index_paths(index_root)
    return IndexArtifacts(
        root=index_root,
        clips_jsonl=clips_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        source_kind="nested_jsonl",
        built_for_evaluation=True,
        rebuilt_this_run=bool(rebuild_reasons),
    )


def _parse_step_query(
    step: PreferenceStepInput,
    resources: StructuredResources,
) -> ParsedStepQuery:
    query_text, components = build_step_query(step)
    known_verbs = {row.action_lemma for row in resources.verbnet if row.has_mapping}
    triples, fallback_text = parse_query_triples(
        query_text,
        DEFAULT_SPACY_MODEL,
        known_verbs=known_verbs,
    )
    # The production parser uses a synthetic query ID.  Relabel it and attach
    # the step's existing inventory context with the same Month 1 helper used
    # by the legacy typed-step pipeline.
    triples = [triple.model_copy(update={"record_id": step.step_id}) for triple in triples]
    inventory_step = StepRecord(
        step_id=step.step_id,
        tools=step.tools,
        materials=step.materials,
    )
    triples = add_record_inventories(triples, [], [inventory_step])
    diagnostics = {
        "initial_action_parse_succeeded": bool(triples) and fallback_text is None,
        "imperative_fallback_applied": fallback_text is not None,
        "imperative_fallback_text": fallback_text,
        "action_parse_succeeded": bool(triples),
        "missing_action_parse": not triples,
        "parsed_action_count": len(triples),
        "actions": sorted({triple.action_lemma for triple in triples}),
        "objects": sorted(
            {term for triple in triples for term in triple.object_lemmas}
        ),
        "tool_context": sorted(
            {
                term
                for triple in triples
                for term in (triple.tool_lemmas or triple.context_tool_lemmas)
            }
        ),
        "supply_context": sorted(
            {
                term
                for triple in triples
                for term in triple.context_material_lemmas
            }
        ),
        "negated_action_count": sum(triple.negated for triple in triples),
    }
    return ParsedStepQuery(
        text=query_text,
        components=components,
        triples=triples,
        diagnostics=diagnostics,
    )


def _resolution_dict(result: ClipResolutionResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def _pair_resolution(
    pair: PreferencePairInput,
    resolution_a: ClipResolutionResult,
    resolution_b: ClipResolutionResult,
) -> tuple[str, list[str], str | None]:
    messages = [
        message
        for message in (resolution_a.message, resolution_b.message)
        if message is not None
    ]
    if not resolution_a.is_resolved or not resolution_b.is_resolved:
        return "unresolved_clip_reference", messages, None
    clip_a_id = resolution_a.canonical_clip_id
    clip_b_id = resolution_b.canonical_clip_id
    if clip_a_id == clip_b_id:
        messages.append("clip_a and clip_b resolve to the same canonical clip")
        return "same_canonical_clip", messages, None
    selected_id = clip_a_id if pair.winner_position == "A" else clip_b_id
    if pair.winner_clip_id is not None:
        if pair.winner_clip_id not in {clip_a_id, clip_b_id}:
            messages.append(
                f"winner_clip_id {pair.winner_clip_id!r} refers to neither resolved clip"
            )
            return "invalid_winner_clip_id", messages, None
        if pair.winner_clip_id != selected_id:
            messages.append("winner_clip_id conflicts with winner_position")
            return "winner_position_conflict", messages, None
    return "resolved", messages, selected_id


def _candidate_diagnostics(
    clip_id: str,
    resources: StructuredResources,
    structured_signals: dict[str, float],
) -> dict[str, Any]:
    triples = resources.triples_for("clip", clip_id)
    return {
        **structured_signals,
        "missing_structured_evidence": not triples,
        "candidate_action_count": len(triples),
        "candidate_negated_action_count": sum(triple.negated for triple in triples),
    }


def _method_summary(
    rows: list[dict[str, Any]],
    method: str,
    *,
    seed: int,
    bootstrap_iterations: int,
    include_ci: bool = True,
) -> dict[str, Any]:
    scored = [row for row in rows if row[f"{method}_agreement_credit"] is not None]
    credits = [float(row[f"{method}_agreement_credit"]) for row in scored]
    summary: dict[str, Any] = {
        "resolved_pairs": len(scored),
        "mean_agreement_credit": sum(credits) / len(credits) if credits else None,
        "wins": sum(value == 1.0 for value in credits),
        "ties": sum(value == 0.5 for value in credits),
        "losses": sum(value == 0.0 for value in credits),
    }
    if include_ci:
        summary["clustered_bootstrap_95_ci"] = clustered_bootstrap_mean_ci(
            credits,
            [str(row["step_id"]) for row in scored],
            seed=seed,
            draws=bootstrap_iterations,
            confidence=0.95,
            cluster_unit="step_id",
        )
    return summary


def _confidence_summary(pairs: list[PreferencePairInput]) -> dict[str, Any]:
    values = [pair.judge_confidence for pair in pairs if pair.judge_confidence is not None]
    return {
        "present": len(values),
        "missing": len(pairs) - len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Frozen Step 1 pairwise-preference evaluation",
        "",
        (
            f"Resolved {summary['counts']['resolved_comparisons']} of "
            f"{summary['counts']['total_comparisons']} comparisons "
            f"({summary['counts']['resolution_percentage']:.1f}%)."
        ),
        "",
        "| method | resolved pairs | agreement | 95% clustered CI | wins | ties | losses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = summary["methods"][method]
        agreement = values["mean_agreement_credit"]
        ci = values["clustered_bootstrap_95_ci"]
        agreement_text = "n/a" if agreement is None else f"{agreement:.3f}"
        ci_text = (
            "n/a"
            if ci["lower"] is None
            else f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
        )
        lines.append(
            f"| {method} | {values['resolved_pairs']} | {agreement_text} | "
            f"{ci_text} | {values['wins']} | {values['ties']} | {values['losses']} |"
        )
    configuration = summary["configuration"]
    lines.extend(
        [
            "",
            "## Frozen configuration",
            "",
            f"- Query rule: `{configuration['query_rule_version']}`",
            f"- Hybrid lexical alpha: `{configuration['hybrid_alpha_lexical']}`",
            (
                "- Structured weights (action/object/context): "
                f"`{configuration['structured_weights']['action']}/"
                f"{configuration['structured_weights']['object']}/"
                f"{configuration['structured_weights']['context']}`"
            ),
            (
                "- Timestamp tolerance: "
                f"`{configuration['timestamp_tolerance_seconds']}` seconds"
            ),
            (
                "- Bootstrap: "
                f"`{configuration['bootstrap_iterations']}` draws, clustered by "
                f"`{configuration['bootstrap_cluster_unit']}`, seed "
                f"`{configuration['bootstrap_seed']}`"
            ),
            "",
            (
                "Agreement measures direct A/B preference agreement. It is not top-k "
                "corpus retrieval accuracy. Unresolved comparisons are retained in "
                "`pair_scores.jsonl` and excluded from method rates."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_preference_pairs(
    *,
    index_artifacts: IndexArtifacts,
    steps_jsonl: Path,
    pairs_jsonl: Path,
    output_dir: Path,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, Path]:
    """Evaluate valid preference pairs against an existing canonical index."""
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "validation_report.json"
    steps, pairs, fatal_errors = _load_and_validate_inputs(steps_jsonl, pairs_jsonl)
    if fatal_errors:
        report = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "failed",
            "fatal_error_count": len(fatal_errors),
            "fatal_errors": fatal_errors,
            "input_counts": {"steps_loaded": len(steps), "comparisons_loaded": len(pairs)},
            "comparisons_silently_dropped": 0,
        }
        _write_json(validation_path, report)
        raise ValueError(
            f"Preference input validation failed; see {validation_path}"
        )

    try:
        clips = read_clips(index_artifacts.clips_jsonl)
        if not clips:
            raise ValueError("canonical clip index is empty")
        resolver = ClipResolver(
            clips, tolerance_seconds=DEFAULT_TIMESTAMP_TOLERANCE_SECONDS
        )
        resources = resources_from_files(
            index_artifacts.month1_dir, index_artifacts.month2_dir
        )
        tfidf = TfidfIndex.from_clips(clips)
    except (OSError, ValidationError, ValueError) as exc:
        report = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "failed",
            "fatal_error_count": 1,
            "fatal_errors": [
                {
                    "code": "invalid_index",
                    "input": "index",
                    "message": str(exc),
                }
            ],
            "input_counts": {"steps_loaded": len(steps), "comparisons_loaded": len(pairs)},
            "comparisons_silently_dropped": 0,
        }
        _write_json(validation_path, report)
        raise ValueError(f"Preference index validation failed; see {validation_path}") from exc

    steps_by_id = {step.step_id: step for step in steps}
    referenced_step_ids = sorted({pair.step_id for pair in pairs})
    parsed_queries = {
        step_id: _parse_step_query(steps_by_id[step_id], resources)
        for step_id in referenced_step_ids
    }
    lexical_by_step = {
        step_id: tfidf.scores(parsed_queries[step_id].text)
        for step_id in referenced_step_ids
    }

    resolutions: list[
        tuple[
            PreferencePairInput,
            ClipResolutionResult,
            ClipResolutionResult,
            str,
            list[str],
            str | None,
        ]
    ] = []
    needed_by_step: defaultdict[str, set[str]] = defaultdict(set)
    validation_issues: list[dict[str, Any]] = []
    for pair in pairs:
        resolution_a = resolver.resolve(pair.clip_a)
        resolution_b = resolver.resolve(pair.clip_b)
        pair_status, messages, judged_winner_clip_id = _pair_resolution(
            pair, resolution_a, resolution_b
        )
        resolutions.append(
            (
                pair,
                resolution_a,
                resolution_b,
                pair_status,
                messages,
                judged_winner_clip_id,
            )
        )
        if pair_status == "resolved":
            needed_by_step[pair.step_id].update(
                {
                    str(resolution_a.canonical_clip_id),
                    str(resolution_b.canonical_clip_id),
                }
            )
        else:
            validation_issues.append(
                {
                    "comparison_id": pair.comparison_id,
                    "step_id": pair.step_id,
                    "code": pair_status,
                    "messages": messages,
                    "clip_a_status": resolution_a.status,
                    "clip_b_status": resolution_b.status,
                }
            )

    scores_by_step: dict[str, dict[str, dict[str, Any]]] = {}
    for step_id, clip_ids in needed_by_step.items():
        query = parsed_queries[step_id]
        scores_by_step[step_id] = score_query_clip_ids(
            query_triples=query.triples,
            clip_ids=sorted(clip_ids),
            lexical_scores=lexical_by_step[step_id],
            resources=resources,
            hybrid_alpha=FROZEN_HYBRID_ALPHA,
        )

    detail_rows: list[dict[str, Any]] = []
    for (
        pair,
        resolution_a,
        resolution_b,
        pair_status,
        resolution_messages,
        judged_winner_clip_id,
    ) in resolutions:
        query = parsed_queries[pair.step_id]
        row: dict[str, Any] = {
            "schema_version": PREFERENCE_SCHEMA_VERSION,
            "comparison_id": pair.comparison_id,
            "step_id": pair.step_id,
            "query_text": query.text,
            "query_construction": query.components,
            "query_diagnostics": query.diagnostics,
            "original_clip_a_reference": pair.clip_a.reference_dict(),
            "original_clip_b_reference": pair.clip_b.reference_dict(),
            "clip_a_resolution": _resolution_dict(resolution_a),
            "clip_b_resolution": _resolution_dict(resolution_b),
            "resolved_canonical_clip_a_id": resolution_a.canonical_clip_id,
            "resolved_canonical_clip_b_id": resolution_b.canonical_clip_id,
            "pair_resolution_status": pair_status,
            "resolution_messages": resolution_messages,
            "judged_winner": pair.winner_position,
            "winner_position": pair.winner_position,
            "winner_clip_id_input": pair.winner_clip_id,
            "judged_winner_canonical_clip_id": judged_winner_clip_id,
            "judge_provenance": pair.judge_provenance,
            "judge_confidence": pair.judge_confidence,
            "judge_id": pair.judge_id,
            "source_row_sha256": pair.source_row_sha256,
            "effective_hybrid_alpha_lexical": None,
        }
        for method in METHODS:
            row[f"{method}_a_score"] = None
            row[f"{method}_b_score"] = None
            row[f"{method}_predicted_winner"] = None
            row[f"{method}_agreement_credit"] = None
        row["structured_a_diagnostics"] = None
        row["structured_b_diagnostics"] = None

        if pair_status == "resolved":
            clip_a_id = str(resolution_a.canonical_clip_id)
            clip_b_id = str(resolution_b.canonical_clip_id)
            scores_a = scores_by_step[pair.step_id][clip_a_id]
            scores_b = scores_by_step[pair.step_id][clip_b_id]
            for method in METHODS:
                score_a = float(scores_a[f"{method}_score"])
                score_b = float(scores_b[f"{method}_score"])
                predicted, credit = exact_pair_outcome(
                    score_a, score_b, pair.winner_position
                )
                row[f"{method}_a_score"] = score_a
                row[f"{method}_b_score"] = score_b
                row[f"{method}_predicted_winner"] = predicted
                row[f"{method}_agreement_credit"] = credit
            row["effective_hybrid_alpha_lexical"] = scores_a[
                "effective_hybrid_alpha_lexical"
            ]
            row["structured_a_diagnostics"] = _candidate_diagnostics(
                clip_a_id, resources, scores_a["structured_signals"]
            )
            row["structured_b_diagnostics"] = _candidate_diagnostics(
                clip_b_id, resources, scores_b["structured_signals"]
            )
        detail_rows.append(row)

    pair_scores_path = output_dir / "pair_scores.jsonl"
    write_jsonl(pair_scores_path, detail_rows)
    resolved_count = sum(row["pair_resolution_status"] == "resolved" for row in detail_rows)
    unresolved_count = len(detail_rows) - resolved_count
    resolution_percentage = 100.0 * resolved_count / len(detail_rows) if detail_rows else 0.0
    weights = StructuredWeights()
    methods = {
        method: _method_summary(
            detail_rows,
            method,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
        )
        for method in METHODS
    }
    provenance_slices: dict[str, Any] = {}
    for provenance in sorted({pair.judge_provenance for pair in pairs}):
        sliced = [row for row in detail_rows if row["judge_provenance"] == provenance]
        provenance_slices[provenance] = {
            "comparison_count": len(sliced),
            "resolved_comparison_count": sum(
                row["pair_resolution_status"] == "resolved" for row in sliced
            ),
            "methods": {
                method: _method_summary(
                    sliced,
                    method,
                    seed=seed,
                    bootstrap_iterations=bootstrap_iterations,
                    include_ci=False,
                )
                for method in METHODS
            },
            "interpretation": "descriptive_only",
        }

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "project1_step1_pairwise_preference_agreement",
        "frozen_no_tuning_evaluation": True,
        "counts": {
            "total_comparisons": len(detail_rows),
            "resolved_comparisons": resolved_count,
            "unresolved_comparisons": unresolved_count,
            "resolution_percentage": resolution_percentage,
            "resolution_target_percentage": 90.0,
            "resolution_target_met": resolution_percentage >= 90.0,
            "referenced_steps": len(referenced_step_ids),
        },
        "methods": methods,
        "descriptive_counts": {
            "pair_resolution_status": dict(
                sorted(Counter(row["pair_resolution_status"] for row in detail_rows).items())
            ),
            "clip_resolution_status": dict(
                sorted(
                    Counter(
                        resolution.status
                        for row in resolutions
                        for resolution in (row[1], row[2])
                    ).items()
                )
            ),
            "winner_position": dict(
                sorted(Counter(pair.winner_position for pair in pairs).items())
            ),
            "judge_provenance": dict(
                sorted(Counter(pair.judge_provenance for pair in pairs).items())
            ),
            "judge_confidence": _confidence_summary(pairs),
            "query_parse_failures_unique_steps": sum(
                query.diagnostics["missing_action_parse"]
                for query in parsed_queries.values()
            ),
            "query_parse_failures_comparisons": sum(
                parsed_queries[pair.step_id].diagnostics["missing_action_parse"]
                for pair in pairs
            ),
            "imperative_fallbacks_unique_steps": sum(
                query.diagnostics["imperative_fallback_applied"]
                for query in parsed_queries.values()
            ),
        },
        "agreement_by_judge_provenance": provenance_slices,
        "configuration": {
            "query_rule_version": STEP_QUERY_RULE_VERSION,
            "query_fields_in_order": ["title", "description", "tools", "materials"],
            "structured_query_inventory_context": (
                "existing Month 1 step tools/materials attachment"
            ),
            "lexical_index_scope": "full canonical clip corpus; fitted once",
            "lexical_tfidf_settings": PRODUCTION_TFIDF_SETTINGS,
            "hybrid_alpha_lexical": FROZEN_HYBRID_ALPHA,
            "hybrid_no_action_behavior": "lexical_fallback",
            "structured_weights": {
                "action": weights.action,
                "object": weights.object,
                "context": weights.context,
            },
            "structured_scorer": STRUCTURED_SCORER_VERSION,
            "taxonomy_used_for_ranking": False,
            "spacy_model": DEFAULT_SPACY_MODEL,
            "index_build_random_seed": DEFAULT_RANDOM_SEED,
            "index_min_taxonomy_support": FROZEN_MIN_TAXONOMY_SUPPORT,
            "timestamp_tolerance_seconds": DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": seed,
            "bootstrap_confidence": 0.95,
            "bootstrap_cluster_unit": "step_id",
            "tie_policy": "exact floating-point equality receives 0.5 credit",
        },
        "index": {
            "source_kind": index_artifacts.source_kind,
            "built_for_evaluation": index_artifacts.built_for_evaluation,
            "rebuilt_this_run": index_artifacts.rebuilt_this_run,
            "root": str(index_artifacts.root),
            "canonical_clip_count": len(clips),
            "index_manifest": (
                {
                    "path": str(index_artifacts.root / "index_manifest.json"),
                    "sha256": sha256_file(index_artifacts.root / "index_manifest.json"),
                }
                if (index_artifacts.root / "index_manifest.json").is_file()
                else None
            ),
        },
        "provenance": {
            "retrieval": build_retrieval_provenance(
                clips_jsonl=index_artifacts.clips_jsonl,
                month1_dir=index_artifacts.month1_dir,
                month2_dir=index_artifacts.month2_dir,
                spacy_model=DEFAULT_SPACY_MODEL,
            ),
            "steps": {"path": str(steps_jsonl), "sha256": sha256_file(steps_jsonl)},
            "pairs": {"path": str(pairs_jsonl), "sha256": sha256_file(pairs_jsonl)},
        },
        "interpretation": (
            "Direct agreement with observed A/B preferences; not top-k corpus retrieval "
            "accuracy. Provenance slices are descriptive and are not tuning subsets."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_markdown_path = output_dir / "summary.md"
    _write_json(summary_path, summary)
    summary_markdown_path.write_text(_markdown_summary(summary), encoding="utf-8")
    validation_report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "passed" if not validation_issues else "passed_with_resolution_issues",
        "fatal_error_count": 0,
        "fatal_errors": [],
        "resolution_issue_count": len(validation_issues),
        "resolution_issues": validation_issues,
        "input_counts": {
            "steps_loaded": len(steps),
            "comparisons_loaded": len(pairs),
        },
        "output_counts": {
            "pair_score_rows": len(detail_rows),
            "resolved_comparisons": resolved_count,
            "unresolved_comparisons": unresolved_count,
        },
        "comparisons_silently_dropped": 0,
    }
    _write_json(validation_path, validation_report)
    manifest_path = output_dir / "manifest.json"
    manifest_inputs = [
        index_artifacts.clips_jsonl,
        index_artifacts.month1_dir / "action_object_tool_triples.jsonl",
        index_artifacts.month1_dir / "verbnet_mappings.jsonl",
        index_artifacts.month2_dir / "framenet_mappings.jsonl",
        index_artifacts.month2_dir / "diy_actionnet_v1.jsonl",
        steps_jsonl,
        pairs_jsonl,
    ]
    index_manifest_path = index_artifacts.root / "index_manifest.json"
    if index_manifest_path.is_file():
        manifest_inputs.append(index_manifest_path)
    write_manifest(
        manifest_path,
        build_manifest(
            command="evaluate-pairs",
            input_files=manifest_inputs,
            output_files=[
                pair_scores_path,
                summary_path,
                summary_markdown_path,
                validation_path,
            ],
            parameters=summary["configuration"],
        ),
    )
    return {
        "pair_scores": pair_scores_path,
        "summary": summary_path,
        "summary_markdown": summary_markdown_path,
        "validation_report": validation_path,
        "manifest": manifest_path,
        "index_root": index_artifacts.root,
    }


def run_preference_evaluation(
    *,
    index: Path,
    steps_jsonl: Path,
    pairs_jsonl: Path,
    output_dir: Path,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, Path]:
    """Run the one-command Step-1 path from a raw or already-built index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Validate the two lightweight experiment files before an expensive index
    # build. The evaluator repeats this check so its direct API is equally safe.
    steps, pairs, errors = _load_and_validate_inputs(steps_jsonl, pairs_jsonl)
    if errors:
        validation_path = output_dir / "validation_report.json"
        _write_json(
            validation_path,
            {
                "schema_version": VALIDATION_SCHEMA_VERSION,
                "status": "failed",
                "fatal_error_count": len(errors),
                "fatal_errors": errors,
                "input_counts": {
                    "steps_loaded": len(steps),
                    "comparisons_loaded": len(pairs),
                },
                "comparisons_silently_dropped": 0,
            },
        )
        raise ValueError(f"Preference input validation failed; see {validation_path}")
    try:
        artifacts = prepare_preference_index(index, output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        validation_path = output_dir / "validation_report.json"
        _write_json(
            validation_path,
            {
                "schema_version": VALIDATION_SCHEMA_VERSION,
                "status": "failed",
                "fatal_error_count": 1,
                "fatal_errors": [
                    {"code": "index_build_error", "input": "index", "message": str(exc)}
                ],
                "input_counts": {
                    "steps_loaded": len(steps),
                    "comparisons_loaded": len(pairs),
                },
                "comparisons_silently_dropped": 0,
            },
        )
        raise ValueError(f"Preference index preparation failed; see {validation_path}") from exc
    return evaluate_preference_pairs(
        index_artifacts=artifacts,
        steps_jsonl=steps_jsonl,
        pairs_jsonl=pairs_jsonl,
        output_dir=output_dir,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
    )
