"""Run the parts of the pipeline supported by the IndexedVideo sample."""

from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig
from .indexed_videos import prepare_indexed_videos
from .month1 import run_month1
from .month2 import run_month2
from .provenance import build_manifest, write_manifest
from .quality_review import build_quality_review
from .retrieval.lexical import PRODUCTION_CANDIDATE_FIELDS
from .retrieval.scorers import STRUCTURED_SCORER_VERSION
from .verification import verify_structured_analysis


def run_indexed_video_analysis(
    *,
    indexed_videos_jsonl: Path,
    output_dir: Path,
    spacy_model: str,
    random_seed: int,
    min_taxonomy_support: int,
) -> dict[str, Path]:
    """Flatten the real sample and run its valid Month 1/2 analysis path."""
    input_paths = prepare_indexed_videos(indexed_videos_jsonl, output_dir / "input")
    config = PipelineConfig(
        output_dir=output_dir,
        spacy_model=spacy_model,
        random_seed=random_seed,
        clip_limit=None,
    )
    month1 = run_month1(clips_jsonl=input_paths["clips"], steps_jsonl=None, config=config)
    month2 = run_month2(
        month1_dir=month1["month_dir"],
        config=config,
        min_taxonomy_support=min_taxonomy_support,
    )
    quality = build_quality_review(
        clips_jsonl=input_paths["clips"],
        month1_dir=month1["month_dir"],
        month2_dir=month2["month_dir"],
        output_dir=output_dir / "quality",
        random_seed=random_seed,
    )
    verification = verify_structured_analysis(output_dir)
    profile = json.loads(input_paths["profile"].read_text(encoding="utf-8"))
    summary = json.loads(month1["summary"].read_text(encoding="utf-8"))
    diagnostics = json.loads(month2["diagnostics"].read_text(encoding="utf-8"))
    quality_summary = json.loads(quality["quality_summary"].read_text(encoding="utf-8"))
    index_manifest_path = output_dir / "index_manifest.json"
    manifest = build_manifest(
        command="build-index",
        input_files=[indexed_videos_jsonl],
        output_files=[
            input_paths["clips"],
            input_paths["rejected"],
            month1["triples"],
            month1["verbnet"],
            month2["framenet"],
        ],
        parameters={
            "index_schema_version": "indexed-video-segments-v2",
            "scorer_version": STRUCTURED_SCORER_VERSION,
            "spacy_model": spacy_model,
            "random_seed": random_seed,
            "canonical_clip_count": profile["canonical_clip_count"],
            "lexical_candidate_fields": PRODUCTION_CANDIDATE_FIELDS,
            "structured_candidate_fields": [
                "title",
                "description",
                "goal",
                "associated tool inventory",
                "associated supply inventory",
            ],
            "taxonomy_used_for_ranking": False,
        },
    )
    manifest["schema_version"] = "index-manifest.v2"
    write_manifest(index_manifest_path, manifest)
    report_path = output_dir / "sample_analysis_report.json"
    report_path.write_text(
        json.dumps(
            {
                "input_profile": profile,
                "month1": summary,
                "month2_taxonomy": diagnostics,
                "extraction_quality": quality_summary,
                "verification_artifacts": sorted(verification),
                "index_manifest": str(index_manifest_path),
                "search_status": (
                    "Ready: the structured index can return top-k clip matches "
                    "for an action query."
                ),
                "evaluation_status": (
                    "Top-k search is runnable now. Human-labeled comparative accuracy "
                    "is a separate optional evaluation stage."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "report": report_path,
        "profile": input_paths["profile"],
        "index_manifest": index_manifest_path,
        **month1,
        **month2,
        **quality,
    }
