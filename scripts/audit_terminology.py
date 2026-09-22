"""Read-only terminology audit; writes review artifacts, never rewrites annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("indexed-videos-250.jsonl"))
    parser.add_argument("--clips", type=Path, default=Path(
        "project1_outputs/indexed-video-sample/input/indexed_video_clips.jsonl"))
    parser.add_argument("--lexicon", type=Path, default=Path(
        "data_contracts/terminology-seed.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "project1_outputs/terminology-audit"))
    args = parser.parse_args()
    videos, clips = read_rows(args.source), read_rows(args.clips)
    concepts = json.loads(args.lexicon.read_text())["concepts"]
    args.output.mkdir(parents=True, exist_ok=True)
    raw_clips = [clip for video in videos for clip in video["clips"]]
    summary = {
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.source, args.clips, args.lexicon)},
        "source_videos": len(videos), "raw_annotations": len(raw_clips),
        "canonical_clips": len(clips),
        "raw_empty_fields": {f: sum(not c.get(f) for c in raw_clips)
                             for f in ("name", "description", "goal", "tools", "supplies")},
        "canonical_empty_fields": {
            "description": sum(not c.get("description") for c in clips),
            "goal": sum(not c.get("summary") for c in clips)},
        "inventory": {},
        "limitations": [
            "Lexicon is a proposal; phrase occurrences do not prove synonym equivalence.",
            "Counts include alternative and purpose mentions in clip-local fields.",
            "No parent video narrative is counted; alias clip counts may overlap.",
            "Inventory flags are review candidates, not adjudicated errors.",
            "Raw and canonical inputs are hashed separately; use the matching built index.",
        ],
    }
    inventory, cases, alias_counts, alias_examples = [], [], [], []
    for field in ("tool_items", "supply_items"):
        names = Counter()
        refs = defaultdict(set)
        totals = Counter()
        for clip in clips:
            for item in clip["gemini_metadata"]["clip"].get(field, []):
                name = item["name"]
                names[name] += 1
                refs[name].add(clip["clip_id"])
                flags = []
                if name.count("(") != name.count(")"):
                    flags.append("unbalanced_parentheses")
                if re.search(r"\b(?:unknown|unspecified|n/a)\b", name, re.I):
                    flags.append("placeholder_in_name")
                if re.search(r"\balternatives:|\bused for\b", name, re.I):
                    flags.append("annotation_marker_in_name")
                totals.update(flags)
                totals["with_alternatives"] += bool(item.get("alternatives"))
                if flags:
                    cases.append({"clip_id": clip["clip_id"], "field": field,
                                  "name": name, "raw": item.get("raw", ""),
                                  "flags": ";".join(flags)})
        summary["inventory"][field] = {
            "item_occurrences": sum(names.values()), "distinct_exact_names": len(names),
            "names_seen_once": sum(n == 1 for n in names.values()), **dict(totals)}
        for name, count in names.most_common():
            inventory.append({"field": field, "name": name, "occurrences": count,
                              "clip_count": len(refs[name]),
                              "example_clip_id": sorted(refs[name])[0]})
    for concept in concepts:
        for alias in concept["aliases"]:
            pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.I)
            matched_clips = set()
            for clip in clips:
                meta = clip["gemini_metadata"]["clip"]
                fields = {"name": clip.get("title"), "description": clip.get("description"),
                          "goal": clip.get("summary"),
                          "tools": meta.get("source_tools_text"),
                          "supplies": meta.get("source_supplies_text")}
                for field, value in fields.items():
                    if value and pattern.search(value):
                        matched_clips.add(clip["clip_id"])
                        alias_examples.append({"concept_id": concept["id"], "alias": alias,
                                               "clip_id": clip["clip_id"], "field": field,
                                               "text": value, "review_decision": ""})
            alias_counts.append({"concept_id": concept["id"], "alias": alias,
                                 "clip_count": len(matched_clips)})
    write_csv(args.output / "inventory.csv", inventory,
              ["field", "name", "occurrences", "clip_count", "example_clip_id"])
    write_csv(args.output / "inventory_review.csv", cases,
              ["clip_id", "field", "name", "raw", "flags"])
    write_csv(args.output / "alias_counts.csv", alias_counts,
              ["concept_id", "alias", "clip_count"])
    write_csv(args.output / "alias_review.csv", alias_examples,
              ["concept_id", "alias", "clip_id", "field", "text", "review_decision"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
