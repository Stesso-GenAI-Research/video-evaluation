"""Descriptive diagnostics for completed pairwise-preference evaluations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final

from action_semantics.io_utils import iter_jsonl, write_csv
from action_semantics.provenance import build_manifest


DIAGNOSTIC_SCHEMA_VERSION: Final[str] = "pair-diagnostics.v1"
METHODS: Final[tuple[str, ...]] = ("lexical", "structured", "hybrid")
FLAG_ORDER: Final[tuple[str, ...]] = (
    "query_parse_failure",
    "structured_only_correct",
    "lexical_only_correct",
    "hybrid_regression_vs_lexical",
    "hybrid_recovery_vs_lexical",
    "all_methods_tie",
    "structured_zero_evidence",
    "structured_tie",
    "lexical_structured_disagreement",
)


def _required_fields() -> set[str]:
    fields = {
        "comparison_id",
        "step_id",
        "pair_resolution_status",
        "query_text",
        "query_diagnostics",
        "judge_provenance",
        "winner_position",
    }
    for method in METHODS:
        fields.update(
            {
                f"{method}_a_score",
                f"{method}_b_score",
                f"{method}_predicted_winner",
                f"{method}_agreement_credit",
            }
        )
    return fields


def _validate_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Pair score input is empty: {path}")
    required = _required_fields()
    errors: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"row {row_number} is missing fields: {', '.join(missing)}")
        comparison_id = row.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id.strip():
            errors.append(f"row {row_number} has an invalid comparison_id")
        elif comparison_id in seen:
            errors.append(f"row {row_number} duplicates comparison_id {comparison_id!r}")
        else:
            seen.add(comparison_id)
        if len(errors) == 20:
            break
    if errors:
        raise ValueError(f"Invalid pair score input {path}:\n" + "\n".join(errors))


def _diagnostic_flags(row: dict[str, Any]) -> list[str]:
    if row["pair_resolution_status"] != "resolved":
        return []
    query = row.get("query_diagnostics") or {}
    lexical_credit = row["lexical_agreement_credit"]
    structured_credit = row["structured_agreement_credit"]
    hybrid_credit = row["hybrid_agreement_credit"]
    lexical_prediction = row["lexical_predicted_winner"]
    structured_prediction = row["structured_predicted_winner"]
    hybrid_prediction = row["hybrid_predicted_winner"]
    flags: set[str] = set()
    if query.get("missing_action_parse") is True:
        flags.add("query_parse_failure")
    if structured_credit == 1.0 and lexical_credit == 0.0:
        flags.add("structured_only_correct")
    if lexical_credit == 1.0 and structured_credit == 0.0:
        flags.add("lexical_only_correct")
    if lexical_credit == 1.0 and hybrid_credit == 0.0:
        flags.add("hybrid_regression_vs_lexical")
    if lexical_credit == 0.0 and hybrid_credit == 1.0:
        flags.add("hybrid_recovery_vs_lexical")
    if all(row[f"{method}_predicted_winner"] == "tie" for method in METHODS):
        flags.add("all_methods_tie")
    if row["structured_a_score"] == 0.0 and row["structured_b_score"] == 0.0:
        flags.add("structured_zero_evidence")
    if structured_prediction == "tie":
        flags.add("structured_tie")
    if lexical_prediction != structured_prediction:
        flags.add("lexical_structured_disagreement")
    if hybrid_prediction not in {lexical_prediction, structured_prediction}:
        flags.add("hybrid_distinct_prediction")
    return [flag for flag in FLAG_ORDER if flag in flags] + sorted(flags - set(FLAG_ORDER))


def _priority(flags: list[str], resolved: bool) -> tuple[int, str]:
    if not resolved:
        return -1, "unresolved"
    if not flags:
        return len(FLAG_ORDER), "other"
    first = flags[0]
    return FLAG_ORDER.index(first) if first in FLAG_ORDER else len(FLAG_ORDER), first


def _pair_type(row: dict[str, Any]) -> str:
    metadata = row.get("input_metadata")
    if not isinstance(metadata, dict):
        return "unspecified"
    value = metadata.get("synthetic_pair_type") or metadata.get("pair_type")
    return str(value).strip() if value is not None and str(value).strip() else "unspecified"


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    credits = [
        float(row[f"{method}_agreement_credit"])
        for row in rows
        if row["pair_resolution_status"] == "resolved"
        and row[f"{method}_agreement_credit"] is not None
    ]
    predictions = Counter(
        str(row[f"{method}_predicted_winner"])
        for row in rows
        if row["pair_resolution_status"] == "resolved"
    )
    return {
        "resolved_comparisons": len(credits),
        "mean_agreement_credit": sum(credits) / len(credits) if credits else None,
        "wins": sum(value == 1.0 for value in credits),
        "ties": sum(value == 0.5 for value in credits),
        "losses": sum(value == 0.0 for value in credits),
        "prediction_counts": dict(sorted(predictions.items())),
    }


def _case_row(row: dict[str, Any], flags: list[str]) -> dict[str, Any]:
    resolved = row["pair_resolution_status"] == "resolved"
    priority, primary = _priority(flags, resolved)
    a_diagnostics = row.get("structured_a_diagnostics") or {}
    b_diagnostics = row.get("structured_b_diagnostics") or {}
    return {
        "priority": priority,
        "primary_diagnostic": primary,
        "diagnostic_flags": ";".join(flags),
        "comparison_id": row["comparison_id"],
        "step_id": row["step_id"],
        "pair_type": _pair_type(row),
        "query_text": row["query_text"],
        "pair_resolution_status": row["pair_resolution_status"],
        "clip_a_id": row.get("resolved_canonical_clip_a_id"),
        "clip_b_id": row.get("resolved_canonical_clip_b_id"),
        "winner_position": row["winner_position"],
        "judge_provenance": row["judge_provenance"],
        "judge_confidence": row.get("judge_confidence"),
        "lexical_a_score": row["lexical_a_score"],
        "lexical_b_score": row["lexical_b_score"],
        "lexical_prediction": row["lexical_predicted_winner"],
        "lexical_credit": row["lexical_agreement_credit"],
        "structured_a_score": row["structured_a_score"],
        "structured_b_score": row["structured_b_score"],
        "structured_prediction": row["structured_predicted_winner"],
        "structured_credit": row["structured_agreement_credit"],
        "hybrid_a_score": row["hybrid_a_score"],
        "hybrid_b_score": row["hybrid_b_score"],
        "hybrid_prediction": row["hybrid_predicted_winner"],
        "hybrid_credit": row["hybrid_agreement_credit"],
        "structured_a_action": a_diagnostics.get("action_match"),
        "structured_a_object": a_diagnostics.get("object_match"),
        "structured_a_context": a_diagnostics.get("context_match"),
        "structured_a_missing_evidence": a_diagnostics.get(
            "missing_structured_evidence"
        ),
        "structured_b_action": b_diagnostics.get("action_match"),
        "structured_b_object": b_diagnostics.get("object_match"),
        "structured_b_context": b_diagnostics.get("context_match"),
        "structured_b_missing_evidence": b_diagnostics.get(
            "missing_structured_evidence"
        ),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    lines = [
        "# Pairwise diagnostic summary",
        "",
        "This report is descriptive. It does not change scorer settings or establish a",
        "causal error category. Review the case table before drawing conclusions.",
        "",
        "## Coverage",
        "",
        f"- Input comparisons: {counts['total_comparisons']}",
        f"- Resolved comparisons: {counts['resolved_comparisons']}",
        f"- Unresolved comparisons: {counts['unresolved_comparisons']}",
        "",
        "## Method outcomes",
        "",
        "| method | resolved | agreement | wins | ties | losses |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = summary["methods"][method]
        agreement = values["mean_agreement_credit"]
        agreement_text = "n/a" if agreement is None else f"{agreement:.3f}"
        lines.append(
            f"| {method} | {values['resolved_comparisons']} | {agreement_text} | "
            f"{values['wins']} | {values['ties']} | {values['losses']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic flags",
            "",
            "| flag | comparisons |",
            "| --- | ---: |",
        ]
    )
    for flag, count in summary["diagnostic_flag_counts"].items():
        lines.append(f"| {flag} | {count} |")
    lines.extend(
        [
            "",
            "Flags can overlap. Agreement values repeat the evaluator's descriptive",
            "credits and do not replace its clustered confidence intervals.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_pair_scores(
    *,
    pair_scores_jsonl: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Write a prioritized, non-tuning diagnostic view of pair score rows."""
    rows = list(iter_jsonl(pair_scores_jsonl))
    _validate_rows(rows, pair_scores_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)

    flags_by_id = {
        row["comparison_id"]: _diagnostic_flags(row)
        for row in rows
    }
    case_rows = [
        _case_row(row, flags_by_id[row["comparison_id"]])
        for row in rows
    ]
    case_rows.sort(
        key=lambda row: (row["priority"], row["step_id"], row["comparison_id"])
    )
    flag_counts = Counter(flag for flags in flags_by_id.values() for flag in flags)
    by_pair_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair_type[_pair_type(row)].append(row)

    summary: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "analysis_type": "descriptive_pairwise_error_inventory",
        "changes_scoring_configuration": False,
        "counts": {
            "total_comparisons": len(rows),
            "resolved_comparisons": sum(
                row["pair_resolution_status"] == "resolved" for row in rows
            ),
            "unresolved_comparisons": sum(
                row["pair_resolution_status"] != "resolved" for row in rows
            ),
        },
        "methods": {method: _method_summary(rows, method) for method in METHODS},
        "diagnostic_flag_counts": {
            flag: flag_counts.get(flag, 0)
            for flag in (*FLAG_ORDER, "hybrid_distinct_prediction")
        },
        "pair_type_slices": {
            pair_type: {
                "comparison_count": len(sliced),
                "methods": {
                    method: _method_summary(sliced, method) for method in METHODS
                },
            }
            for pair_type, sliced in sorted(by_pair_type.items())
        },
        "limitations": [
            "Flags are deterministic descriptions, not adjudicated root causes.",
            "Pair-type slices are descriptive and must not be used for post-hoc tuning on a frozen evaluation set.",
            "Synthetic or target-derived inputs are not human preference evidence.",
        ],
    }

    summary_path = output_dir / "diagnostic_summary.json"
    cases_path = output_dir / "diagnostic_cases.csv"
    markdown_path = output_dir / "diagnostic_summary.md"
    manifest_path = output_dir / "diagnostic_manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(cases_path, case_rows)
    _write_markdown(markdown_path, summary)
    manifest = build_manifest(
        command="analyze-pairs",
        input_files=[pair_scores_jsonl],
        output_files=[summary_path, cases_path, markdown_path],
        parameters={
            "analysis_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "descriptive_only": True,
            "changes_scoring_configuration": False,
        },
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "cases": cases_path,
        "markdown": markdown_path,
        "manifest": manifest_path,
    }
