"""Human-review preparation and prospective source-overlap checks.

These utilities never fill human judgment columns or promote development data
to a held-out evaluation. Diagnostic sampling is explicitly selection-biased.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .experiment_suite import CORPORA
from .io_utils import sha256_file, write_csv, write_jsonl
from .pair_diagnostics import _diagnostic_flags

ANSWER_FIELDS = {"preference", "confidence", "notes"}
STRATA = ("query_parse_failure", "structured_zero_evidence",
          "structured_only_correct", "hybrid_regression_vs_lexical")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _immutable(row):
    return {key: value for key, value in row.items() if key not in ANSWER_FIELDS}


def _evidence_group(row):
    if row["structured_a_score"] != 0 or row["structured_b_score"] != 0:
        return "at_least_one_positive_structured_score"
    if row["query_diagnostics"]["missing_action_parse"]:
        return "query_has_no_parsed_action"
    a, b = row["structured_a_diagnostics"], row["structured_b_diagnostics"]
    if not a["candidate_action_count"] and not b["candidate_action_count"]:
        return "neither_candidate_has_parsed_actions"
    if not a["candidate_action_count"] or not b["candidate_action_count"]:
        return "one_candidate_has_no_parsed_actions"
    if a["action_match"] == 0 and b["action_match"] == 0:
        return "both_candidates_parsed_but_zero_action_match"
    return "other_zero_score"


def prepare_review_kit(*, suite: Path, output_dir: Path, per_stratum: int = 12,
                       seed: int = 42) -> dict:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    if _json(suite / "status.json")["status"] != "complete":
        raise ValueError("A completed experiment suite is required")
    clips_path = suite / "index/input/indexed_video_clips.jsonl"
    clips = {c["clip_id"]: c for c in _rows(clips_path)}
    extraction_path = suite / "index/quality/manual_review_sample.csv"
    extraction = _csv(extraction_path)
    inputs = [clips_path, extraction_path]
    candidates, evidence, steps = [], {}, {}
    for corpus in CORPORA:
        root = suite / "pairwise" / corpus
        scores_path, steps_path = root / "step1/pair_scores.jsonl", root / "steps.jsonl"
        inputs.extend([scores_path, steps_path])
        for step in _rows(steps_path):
            sid = step.get("id", step.get("step_id"))
            steps[(corpus, sid)] = {
                "id": sid, **{f: step.get(f, [] if f in ("tools", "materials") else "")
                              for f in ("title", "description", "tools", "materials")}}
        rows = _rows(scores_path)
        resolved = [r for r in rows if r["pair_resolution_status"] == "resolved"]
        evidence[corpus] = {
            "comparisons": len(rows), "resolved": len(resolved),
            "mutually_exclusive_evidence_groups": dict(Counter(_evidence_group(r) for r in resolved)),
            "interpretation": "Score diagnostics, not human-validated causes; zero action match can include negation mismatch.",
        }
        for row in resolved:
            candidates.append((corpus, row, set(_diagnostic_flags(row))))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen, selected_steps, stratum_counts = [], set(), Counter()
    for stratum in STRATA:
        for corpus, row, flags in candidates:
            key = (corpus, row["step_id"])
            if stratum not in flags or key in selected_steps:
                continue
            chosen.append((corpus, row, stratum))
            selected_steps.add(key)
            stratum_counts[stratum] += 1
            if stratum_counts[stratum] == per_stratum:
                break
    rng.shuffle(chosen)
    if not chosen:
        raise ValueError("No eligible diagnostic cases")
    review, key_rows = [], {}
    selected_step_rows = {}

    def details(clip_id):
        clip = clips[clip_id]
        metadata = clip["gemini_metadata"]["clip"]
        return {"clip_id": clip_id, "title": clip.get("title") or "",
                "description": clip.get("description") or "", "goal": clip.get("summary") or "",
                "video_url": clip.get("url") or "",
                "start_seconds": str(metadata["start_seconds"]),
                "end_seconds": str(metadata["end_seconds"])}

    for number, (corpus, pair, stratum) in enumerate(chosen, start=1):
        ids = [pair["resolved_canonical_clip_a_id"], pair["resolved_canonical_clip_b_id"]]
        if rng.getrandbits(1):
            ids.reverse()
        review_id = f"pilot-{number:03d}"
        # Prefix step IDs to prevent collisions across separately generated corpora.
        step_id = f"{corpus}:{pair['step_id']}"
        row = {"review_id": review_id, "step_id": step_id, "query": pair["query_text"]}
        for side, clip_id in zip(("a", "b"), ids, strict=True):
            row.update({f"{side}_{field}": value for field, value in details(clip_id).items()})
        row.update(preference="", confidence="", notes="")
        review.append(row)
        key_rows[review_id] = {"immutable": _immutable(row), "stratum": stratum,
                               "original_comparison_id": pair["comparison_id"], "corpus": corpus}
        selected_step_rows[step_id] = {**steps[(corpus, pair["step_id"])], "id": step_id}
    for row in extraction:
        clip = clips[row["clip_id"]]
        metadata = clip["gemini_metadata"]["clip"]
        row.update(video_url=clip.get("url") or "", start_seconds=metadata["start_seconds"],
                   end_seconds=metadata["end_seconds"])
    output_dir.mkdir(parents=True, exist_ok=False)
    private = output_dir / "private"
    private.mkdir()
    write_csv(output_dir / "extraction_review.csv", extraction)
    write_csv(output_dir / "preference_review.csv", review)
    _write(private / "review_key.json", {"rows": key_rows, "steps": selected_step_rows})
    _write(output_dir / "diagnostic_evidence.json", evidence)
    manifest = {"purpose": "diagnostic_human_review_pilot", "held_out": False,
                "representative_accuracy_sample": False, "seed": seed,
                "requested_per_stratum": per_stratum, "selected_by_stratum": dict(stratum_counts),
                "preference_rows": len(review), "extraction_rows": len(extraction),
                "one_comparison_per_source_step": True,
                "input_sha256": {str(p): sha256_file(p) for p in inputs}}
    _write(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "# Diagnostic review packet\n\n"
        "Complete the two CSV worksheets. Keep `private/` and diagnostic reports away from "
        "preference reviewers until their judgments are complete. The preference worksheet "
        "contains no scorer outputs or constructed winners; A/B order was randomized.\n\n"
        "## Extraction review\n\n"
        "Read the source sentence and proposed action, object and direct-tool terms. "
        "Enter yes/no in `action_correct`, `object_correct`, and `tool_correct`. "
        "Leave a dimension blank if you cannot judge it; explain in notes. A missing direct tool "
        "is correct when the sentence states no direct tool. Record-level tool context is "
        "not evidence of a grammatical tool attachment. Do not treat confidence as correctness. "
        "These 60 rows were sampled by diagnostic group; group results do not estimate "
        "unbiased whole-corpus precision.\n\n"
        "## Preference review\n\n"
        "For each query, inspect both clips at their listed time intervals. Choose the clip "
        "that better demonstrates the requested action on the requested object. Consider "
        "tools/materials when the query specifies them. Shared words alone are insufficient. "
        "Enter `A`, `B`, `tie`, or `unjudgeable` in `preference`; blank means pending. "
        "Optional `confidence` is a number from 0 to 1. Explain uncertainty in `notes`. "
        "Judge video content when accessible; otherwise use unjudgeable rather than guessing "
        "from the descriptions. Do not edit identifiers, query text, or clip columns.\n\n"
        "This packet deliberately samples diagnostic cases from existing development data. "
        "Human review makes the decisions human-observed, but does not make the sample "
        "independent, representative, or held out. Ties, unjudgeable cases and pending rows "
        "are retained in export accounting and excluded explicitly from binary scoring. "
        "Each reviewer should complete a separate copy; do not combine different reviewers "
        "into one judgment without a stated adjudication procedure.\n", encoding="utf-8")
    return manifest


def export_reviewed_pairs(*, kit: Path, worksheet: Path, output_dir: Path, judge: str) -> dict:
    if not judge.strip():
        raise ValueError("A nonblank judge identifier is required")
    key = _json(kit / "private/review_key.json")
    rows = _csv(worksheet)
    ids = [r.get("review_id") for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(key["rows"]):
        raise ValueError("Review IDs are missing, duplicated, or unknown; keep pending rows blank")
    pairs, omitted, counts = [], [], Counter()
    for row in rows:
        if _immutable(row) != key["rows"][row["review_id"]]["immutable"]:
            raise ValueError(f"Query, clip ordering, or another protected field changed: {row['review_id']}")
        answer = row["preference"].strip().lower()
        if answer not in {"a", "b", "tie", "unjudgeable", ""}:
            raise ValueError(f"Invalid preference: {answer!r}")
        confidence = None if not row["confidence"].strip() else float(row["confidence"])
        if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Confidence must be finite and between 0 and 1")
        counts[answer or "pending"] += 1
        if answer not in {"a", "b"}:
            omitted.append({"review_id": row["review_id"], "reason": answer or "pending",
                            "confidence": confidence, "notes": row["notes"]})
            continue
        pairs.append({"comparison_id": row["review_id"], "step_id": row["step_id"],
                      "clip_a": {"clip_id": row["a_clip_id"]},
                      "clip_b": {"clip_id": row["b_clip_id"]}, "winner_position": answer.upper(),
                      "judge_id": judge.strip(), "judge_provenance": "human_diagnostic_pilot",
                      "judge_confidence": confidence, "notes": row["notes"],
                      "selection": "development_diagnostic_cases_not_held_out"})
    if not pairs:
        raise ValueError("No A/B judgments supplied; no inferred winners or evaluator inputs were written")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "pairwise.jsonl", pairs)
    selected = {r["step_id"] for r in pairs}
    write_jsonl(output_dir / "steps.jsonl", [key["steps"][sid] for sid in sorted(selected)])
    write_jsonl(output_dir / "excluded_judgments.jsonl", omitted)
    summary = {"total_rows": len(rows), "exported_binary_judgments": len(pairs),
               "counts": dict(counts), "excluded_rows": len(omitted),
               "held_out": False, "judge": judge.strip(),
               "worksheet_sha256": sha256_file(worksheet),
               "review_key_sha256": sha256_file(kit / "private/review_key.json")}
    _write(output_dir / "export_summary.json", summary)
    return summary


def _video_keys(video):
    video_id = video.get("video_id")
    if isinstance(video_id, bool) or video_id is None or not str(video_id).strip():
        raise ValueError("Each source video needs a nonblank video_id")
    keys = {f"video_id:{video_id}"}
    youtube = str(video.get("youtube_id") or "").strip()
    url = str(video.get("url") or "").strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host == "youtu.be":
        youtube_url = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com"}:
        youtube_url = parse_qs(parsed.query).get("v", [""])[0]
        if not youtube_url and parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            youtube_url = parsed.path.split("/")[2]
    else:
        youtube_url = ""
    for identity in (youtube, youtube_url):
        if identity:
            keys.add(f"youtube:{identity}")
    if host and not youtube_url:
        keys.add("url:" + parsed._replace(fragment="", netloc=host).geturl().rstrip("/"))
    return keys


def check_prospective_holdout(*, development_source: Path, candidate_source: Path,
                             output_dir: Path) -> dict:
    """Fail on source overlap, even if a source video has been assigned a new ID."""
    development, candidates = _rows(development_source), _rows(candidate_source)
    if not development or not candidates:
        raise ValueError("Both source exports must contain video records")
    exposed = set().union(*(_video_keys(v) for v in development))
    seen, overlap, duplicates = set(), [], []
    for row_number, video in enumerate(candidates, start=1):
        keys = _video_keys(video)
        if keys & exposed:
            overlap.append({"row": row_number, "video_id": video["video_id"],
                            "matching_keys": sorted(keys & exposed)})
        if keys & seen:
            duplicates.append({"row": row_number, "video_id": video["video_id"],
                               "matching_keys": sorted(keys & seen)})
        seen.update(keys)
    result = {"status": "fail" if overlap or duplicates else "pass_identity_check_only",
              "development_videos": len(development), "candidate_videos": len(candidates),
              "overlapping_candidate_rows": len(overlap), "duplicate_candidate_rows": len(duplicates),
              "overlaps": overlap, "duplicates": duplicates,
              "development_sha256": sha256_file(development_source),
              "candidate_sha256": sha256_file(candidate_source),
              "limitations": ["Does not detect reuploads with different identifiers or content duplicates.",
                              "Does not prove unseen queries, independent judgments, or absence of prior tuning.",
                              "Passing is an identity check, not a held-out accuracy result."]}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "holdout_check.json", result)
    return result
