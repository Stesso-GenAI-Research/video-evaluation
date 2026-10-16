"""Analyze observed diagnostic reviews against scores recorded before review."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from .io_utils import sha256_file, write_csv, write_jsonl
from .retrieval.evaluation import clustered_bootstrap_mean_ci
from .study_preparation import export_reviewed_pairs

METHODS = ("lexical", "structured", "hybrid")
DIMENSIONS = ("action_correct", "object_correct", "tool_correct")
PREFERENCE_FIELDS = {"preference", "confidence", "notes"}
EXTRACTION_FIELDS = {*DIMENSIONS, "notes"}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_key(kit):
    manifest = _read(kit / "manifest.json")
    if manifest.get("schema_version") != 2:
        raise ValueError("Use a version 2 packet from prepare-review; preserve earlier labeled sheets")
    path = kit / "private/review_key.json"
    if sha256_file(path) != manifest["review_key_sha256"]:
        raise ValueError("Review key hash changed")
    return _read(path)


def _validated_rows(path, expected, answer_fields):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not answer_fields <= set(fields):
            raise ValueError("Missing or duplicated response columns")
        rows = list(reader)
    ids = [row.get("review_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Review IDs are missing, duplicated, or unknown")
    for row in rows:
        if any(value is None for value in row.values()):
            raise ValueError("Incomplete CSV row; keep empty cells and delimiters")
        if {k: v for k, v in row.items() if k not in answer_fields} != expected[row["review_id"]]:
            raise ValueError(f"Protected worksheet fields changed: {row['review_id']}")
    return sorted(rows, key=lambda row: row["review_id"])


def _preferences(path, key):
    expected = {k: v["immutable"] for k, v in key["rows"].items()}
    rows = _validated_rows(path, expected, PREFERENCE_FIELDS)
    for row in rows:
        row["preference"] = row["preference"].strip().lower()
        if row["preference"] not in {"a", "b", "tie", "unjudgeable", ""}:
            raise ValueError(f"Invalid preference in {row['review_id']}")
        confidence = row["confidence"].strip()
        if confidence and (not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1):
            raise ValueError("Confidence must be finite and between 0 and 1")
    return rows


def _extraction(path, key):
    rows = _validated_rows(path, key["extraction_rows"], EXTRACTION_FIELDS)
    for row in rows:
        for dimension in DIMENSIONS:
            value = row[dimension].strip().lower()
            if value not in {"yes", "no", "unjudgeable", ""}:
                raise ValueError(f"Invalid {dimension} in {row['review_id']}; use yes/no/unjudgeable")
            row[dimension] = value
    return rows


def _extraction_summary(rows):
    result = {}
    for dimension in DIMENSIONS:
        counts = Counter(row[dimension] or "pending" for row in rows)
        judged = counts["yes"] + counts["no"]
        result[dimension] = {
            "rows": len(rows), "correct": counts["yes"], "incorrect": counts["no"],
            "judged": judged, "pending": counts["pending"], "unjudgeable": counts["unjudgeable"],
            "precision": counts["yes"] / judged if judged else None,
            "coverage": judged / len(rows) if rows else None,
        }
    return result


def _interval(values, clusters, seed, draws):
    return clustered_bootstrap_mean_ci(values, clusters, seed=seed, draws=draws, cluster_unit="source_step")


def _percentage(value):
    return "--" if value is None else f"{100 * value:.1f}"


def render_review_tex(summary):
    """Render numeric results only; reviewer text is never evaluated as LaTeX."""
    if summary["status"] != "complete":
        return (r"\newcommand{\HumanReviewResults}{\pending{Diagnostic review is still incomplete. "
                r"Complete or explicitly mark each response unjudgeable before reporting results.}}" + "\n")
    pref = summary["preferences"]
    lines = [r"\newcommand{\HumanReviewResults}{",
             f"The diagnostic review contains {pref['rows']} preference cases. "
             f"Of these, {pref['binary_judgments']} received an A/B choice, "
             f"{pref['counts']['tie']} were tied, and {pref['counts']['unjudgeable']} "
             "could not be judged. Agreement uses only A/B choices; an exact model-score "
             "tie receives one-half credit. These selected development cases do not "
             "estimate accuracy on new videos.",
             r"\begin{table}[htbp]\centering\small",
             r"\caption{Diagnostic preference agreement. Percentages and 95\% intervals use source-step clusters.}",
             r"\begin{tabular}{lrr}\toprule Method & Agreement & 95\% interval\\\midrule"]
    for method in METHODS:
        ci = pref["methods"][method]
        lines.append(f"{method.title()} & {_percentage(ci['mean'])} & "
                     f"[{_percentage(ci['lower'])}, {_percentage(ci['upper'])}]" + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}",
              "Extraction correctness is reported by sampling group. Groups can contain "
              "the same extracted record, so their totals must not be interpreted as "
              "distinct random observations from the corpus.",
              r"\begin{table}[htbp]\centering\small",
              r"\caption{Correct extraction judgments divided by judged entries; unjudgeable entries are excluded.}",
              r"\begin{tabular}{lrrr}\toprule Group & Action & Object & Tool\\\midrule"]
    labels = {"complete_direct": "Complete direct", "metadata_tool_fallback": "Inventory fallback",
              "missing_object": "Missing object", "low_confidence": "Low confidence"}
    for number, (group, dimensions) in enumerate(summary["extraction"]["by_group"].items(), 1):
        cells = [f"{dimensions[d]['correct']}/{dimensions[d]['judged']}" for d in DIMENSIONS]
        lines.append(labels.get(group, f"Group {number}") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}", "}"]
    return "\n".join(lines) + "\n"


def analyze_review(*, kit: Path, extraction: Path, preferences: Path, output_dir: Path,
                   judge: str, allow_partial: bool = False, seed: int = 42, draws: int = 5000):
    if not judge.strip():
        raise ValueError("A nonblank judge identifier is required")
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    key = _load_key(kit)
    ext = _extraction(extraction, key)
    pref = _preferences(preferences, key)
    counts = {label: sum((r["preference"] or "pending") == label for r in pref)
              for label in ("a", "b", "tie", "unjudgeable", "pending")}
    pending = counts["pending"] + sum(not r[d] for r in ext for d in DIMENSIONS)
    if pending and not allow_partial:
        raise ValueError(f"{pending} responses are pending; use --allow-partial for a progress report")
    details = []
    for row in pref:
        case = key["rows"][row["review_id"]]
        detail = {"review_id": row["review_id"], "step_id": row["step_id"],
                  "stratum": case["stratum"], "preference": row["preference"] or "pending"}
        for method in METHODS:
            score = case["frozen_scores"][method]
            if any(not math.isfinite(v) for v in score.values()):
                raise ValueError("Non-finite recorded model score")
            answer = row["preference"]
            credit = None
            if answer in {"a", "b"}:
                credit = 0.5 if score["a"] == score["b"] else float(
                    (score["a"] > score["b"]) == (answer == "a"))
            detail[f"{method}_credit"] = credit
        details.append(detail)
    scored = [r for r in details if r["lexical_credit"] is not None]
    clusters = [r["step_id"] for r in scored]
    methods = {m: _interval([r[f"{m}_credit"] for r in scored], clusters, seed, draws) for m in METHODS}
    deltas = {m: _interval([r[f"{m}_credit"] - r["lexical_credit"] for r in scored],
                           clusters, seed, draws) for m in ("structured", "hybrid")}
    strata = {}
    for stratum in sorted({r["stratum"] for r in details}):
        subset = [r for r in scored if r["stratum"] == stratum]
        strata[stratum] = {"binary_judgments": len(subset), "mean_credit": {
            m: sum(r[f"{m}_credit"] for r in subset) / len(subset) if subset else None for m in METHODS}}
    summary = {"schema_version": 1, "status": "partial" if pending else "complete",
               "purpose": "diagnostic_human_review_pilot", "held_out": False,
               "source_suite_summary_sha256": _read(kit / "manifest.json").get("source_suite_summary_sha256"),
               "judge": judge.strip(), "pending_responses": pending,
               "extraction": {"rows": len(ext), "overall_descriptive": _extraction_summary(ext),
                              "by_group": {g: _extraction_summary([r for r in ext if r['review_group'] == g])
                                           for g in sorted({r['review_group'] for r in ext})}},
               "preferences": {"rows": len(pref), "counts": counts, "binary_judgments": len(scored),
                               "methods": methods, "paired_deltas_vs_lexical": deltas, "by_stratum": strata},
               "interpretation": "Selected development cases; neither corpus-wide precision nor independent held-out accuracy. "
                                 "Bootstrap intervals describe this selected sample and do not remove selection bias."}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "analysis.json", summary)
    write_jsonl(output_dir / "case_results.jsonl", details)
    if scored:
        export_reviewed_pairs(kit=kit, worksheet=preferences, output_dir=output_dir / "observed-pairs", judge=judge)
    else:
        write_jsonl(output_dir / "excluded_judgments.jsonl", [
            {"review_id": r["review_id"], "reason": r["preference"] or "pending",
             "confidence": r["confidence"], "notes": r["notes"]} for r in pref])
    lines = ["# Diagnostic human review", "", f"Status: {summary['status']}. Pending responses: {pending}.",
             "", summary["interpretation"], "",
             f"Preference rows: {len(pref)}; A/B judgments: {len(scored)}; ties: {counts['tie']}; "
             f"unjudgeable: {counts['unjudgeable']}; pending: {counts['pending']}.", "",
             "| Method | Agreement (%) | 95% interval (%) |", "|---|---:|---:|"]
    for method, ci in methods.items():
        lines.append(f"| {method} | {_percentage(ci['mean'])} | [{_percentage(ci['lower'])}, {_percentage(ci['upper'])}] |")
    lines += ["", "| Difference from lexical | Estimate (points) | 95% interval (points) |",
              "|---|---:|---:|"]
    for method, ci in deltas.items():
        lines.append(f"| {method} | {_percentage(ci['mean'])} | [{_percentage(ci['lower'])}, {_percentage(ci['upper'])}] |")
    lines += ["", "| Extraction group | Dimension | Correct / judged | Unjudgeable | Pending |",
              "|---|---|---:|---:|---:|"]
    for group, values in summary["extraction"]["by_group"].items():
        for dimension, v in values.items():
            lines.append(f"| {group} | {dimension} | {v['correct']}/{v['judged']} | {v['unjudgeable']} | {v['pending']} |")
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "review-results.tex").write_text(render_review_tex(summary), encoding="utf-8")
    _write(output_dir / "manifest.json", {
        "inputs": {str(p): sha256_file(p) for p in (extraction, preferences, kit / 'private/review_key.json', kit / 'manifest.json')},
        "code_sha256": {name: sha256_file(Path(__file__).parent / name)
                        for name in ('review_analysis.py', 'study_preparation.py', 'retrieval/evaluation.py')},
        "outputs": {str(p.relative_to(output_dir)): sha256_file(p) for p in sorted(output_dir.rglob('*')) if p.is_file()},
    })
    return summary


def compare_reviewers(*, kit: Path, first: Path, second: Path, output_dir: Path):
    key = _load_key(kit)
    a, b = _preferences(first, key), _preferences(second, key)
    eligible = [(x, y) for x, y in zip(a, b, strict=True)
                if x['preference'] in {'a', 'b', 'tie'} and y['preference'] in {'a', 'b', 'tie'}]
    n = len(eligible)
    observed = sum(x['preference'] == y['preference'] for x, y in eligible) / n if n else None
    expected = sum(sum(x['preference'] == v for x, _ in eligible) *
                   sum(y['preference'] == v for _, y in eligible) for v in ('a', 'b', 'tie')) / n**2 if n else None
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    differences = [{**key['rows'][x['review_id']]['immutable'], 'first_preference': x['preference'],
                    'second_preference': y['preference'], 'adjudicated_preference': '', 'adjudication_notes': ''}
                   for x, y in zip(a, b, strict=True) if x['preference'] != y['preference']]
    result = {'rows': len(a), 'jointly_judged': n, 'excluded_from_agreement': len(a) - n,
              'raw_agreement': observed, 'cohen_kappa': kappa, 'different_responses': len(differences),
              'held_out': False, 'interpretation': 'A/B/tie are nominal categories; pending and unjudgeable are excluded. '
              'Kappa is undefined if no joint judgments or expected agreement is one. No automatic adjudication.',
              'first_sha256': sha256_file(first), 'second_sha256': sha256_file(second),
              'review_key_sha256': sha256_file(kit / 'private/review_key.json')}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / 'agreement.json', result)
    fields = [*next(iter(key['rows'].values()))['immutable'], 'first_preference', 'second_preference',
              'adjudicated_preference', 'adjudication_notes']
    write_csv(output_dir / 'disagreements.csv', differences, fieldnames=fields)
    return result
