from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from .config import DEFAULT_RANDOM_SEED, DEFAULT_SPACY_MODEL, PipelineConfig
from .indexed_videos import prepare_indexed_videos as prepare_indexed_videos_impl
from .index_freshness import index_staleness_reasons
from .logging_utils import info
from .month1 import run_month1 as run_month1_impl
from .month2 import run_month2 as run_month2_impl
from .quality import require_nonempty_report, validate_jsonl_basic
from .quality_review import summarize_manual_review
from .retrieval.batch_comparison import run_batch_comparison, score_batch_review
from .retrieval.experiments import run_month3 as run_month3_impl
from .retrieval.comparison import compare_result_sets, write_comparison_results
from .retrieval.benchmark import run_field_heldout_benchmark
from .retrieval.search import rank_indexed_clips, write_search_results
from .sample_analysis import run_indexed_video_analysis as run_indexed_video_analysis_impl
from .verification import verify_output_repository, verify_structured_analysis

app = typer.Typer(no_args_is_help=True, add_completion=False)


class SearchMethodChoice(str, Enum):
    lexical = "lexical"
    structured = "structured"
    hybrid = "hybrid"


class ChallengerMethodChoice(str, Enum):
    structured = "structured"
    hybrid = "hybrid"


def _config(output_dir: Path, spacy_model: str, random_seed: int, clip_limit: int | None) -> PipelineConfig:
    if clip_limit is not None and clip_limit <= 0:
        clip_limit = None
    return PipelineConfig(
        output_dir=output_dir,
        spacy_model=spacy_model,
        random_seed=random_seed,
        clip_limit=clip_limit,
    )


@app.command(hidden=True)
def validate_inputs(
    clips_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    steps_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    pairwise_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
) -> None:
    """Validate real input JSONL files before any extraction or scoring."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "clips": validate_jsonl_basic(clips_jsonl, "clip_id", ["clip_id"]),
        "steps": validate_jsonl_basic(steps_jsonl, "step_id", ["step_id"]),
        "pairwise": validate_jsonl_basic(
            pairwise_jsonl,
            "comparison_id",
            ["comparison_id", "step_id", "clip_a_id", "clip_b_id", "winner_clip_id"],
        ),
    }
    for name, report in reports.items():
        require_nonempty_report(report, name)
    report_path = output_dir / "input_validation_report.json"
    report_path.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    info(f"Input validation passed. Report written to {report_path}")


@app.command(hidden=True)
def prepare_indexed_videos(
    indexed_videos_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
) -> None:
    """Flatten the nested IndexedVideo sample into real clip records for Month 1/2."""
    paths = prepare_indexed_videos_impl(indexed_videos_jsonl, output_dir)
    info(f"Prepared {paths['clips']}. Data profile written to {paths['profile']}")


@app.command("build-index")
def build_index(
    indexed_videos_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
    min_taxonomy_support: Annotated[int, typer.Option(min=1)] = 2,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
    random_seed: Annotated[int, typer.Option()] = DEFAULT_RANDOM_SEED,
) -> None:
    """Build and verify the structured search index from an IndexedVideo export."""
    paths = run_indexed_video_analysis_impl(
        indexed_videos_jsonl=indexed_videos_jsonl,
        output_dir=output_dir,
        min_taxonomy_support=min_taxonomy_support,
        spacy_model=spacy_model,
        random_seed=random_seed,
    )
    info(f"IndexedVideo analysis complete. Report written to {paths['report']}")


@app.command(hidden=True)
def index_current(
    indexed_videos_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
) -> None:
    """Exit successfully only when the generated index is safe to reuse."""
    reasons = index_staleness_reasons(
        source_jsonl=indexed_videos_jsonl,
        output_dir=output_dir,
        spacy_model=spacy_model,
    )
    if reasons:
        typer.echo("Index rebuild required:", err=True)
        for reason in reasons:
            typer.echo(f"- {reason}", err=True)
        raise typer.Exit(code=1)
    info("Index source, code, model versions, configuration, and artifact hashes match.")


@app.command(hidden=True)
def run_month1(
    clips_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    steps_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
    clip_limit: Annotated[int | None, typer.Option()] = 5000,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
    random_seed: Annotated[int, typer.Option()] = DEFAULT_RANDOM_SEED,
) -> None:
    """Month 1: extract action-object-tool-material records and VerbNet mappings."""
    paths = run_month1_impl(
        clips_jsonl=clips_jsonl,
        steps_jsonl=steps_jsonl,
        config=_config(output_dir, spacy_model, random_seed, clip_limit),
    )
    info(f"Month 1 complete. Output directory: {paths['month_dir']}")


@app.command(hidden=True)
def run_month2(
    month1_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
    min_taxonomy_support: Annotated[int, typer.Option(min=1)] = 2,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
    random_seed: Annotated[int, typer.Option()] = DEFAULT_RANDOM_SEED,
) -> None:
    """Month 2: FrameNet, SRL roles, and DIY-ActionNet v1."""
    paths = run_month2_impl(
        month1_dir=month1_dir,
        config=_config(output_dir, spacy_model, random_seed, None),
        min_taxonomy_support=min_taxonomy_support,
    )
    info(f"Month 2 complete. Output directory: {paths['month_dir']}")


@app.command(hidden=True)
def run_month3(
    clips_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    steps_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    pairwise_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    month1_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    month2_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
    dense_key: Annotated[list[str] | None, typer.Option()] = None,
    hybrid_alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
    random_seed: Annotated[int, typer.Option()] = DEFAULT_RANDOM_SEED,
) -> None:
    """Month 3: dense vs structured vs hybrid pairwise retrieval experiments."""
    paths = run_month3_impl(
        clips_jsonl=clips_jsonl,
        steps_jsonl=steps_jsonl,
        pairwise_jsonl=pairwise_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        config=_config(output_dir, spacy_model, random_seed, None),
        dense_keys=dense_key,
        hybrid_alpha=hybrid_alpha,
    )
    info(f"Month 3 complete. Output directory: {paths['month_dir']}")


@app.command(hidden=True)
def run_all(
    clips_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    steps_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    pairwise_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
    clip_limit: Annotated[int | None, typer.Option()] = 5000,
    min_taxonomy_support: Annotated[int, typer.Option(min=1)] = 2,
    dense_key: Annotated[list[str] | None, typer.Option()] = None,
    hybrid_alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
    random_seed: Annotated[int, typer.Option()] = DEFAULT_RANDOM_SEED,
) -> None:
    """Run the full month 1-3 pipeline on real exports."""
    config = _config(output_dir, spacy_model, random_seed, clip_limit)
    month1 = run_month1_impl(clips_jsonl=clips_jsonl, steps_jsonl=steps_jsonl, config=config)
    month2 = run_month2_impl(
        month1_dir=month1["month_dir"],
        config=_config(output_dir, spacy_model, random_seed, None),
        min_taxonomy_support=min_taxonomy_support,
    )
    run_month3_impl(
        clips_jsonl=clips_jsonl,
        steps_jsonl=steps_jsonl,
        pairwise_jsonl=pairwise_jsonl,
        month1_dir=month1["month_dir"],
        month2_dir=month2["month_dir"],
        config=_config(output_dir, spacy_model, random_seed, None),
        dense_keys=dense_key,
        hybrid_alpha=hybrid_alpha,
    )
    verify_output_repository(output_dir)
    info(f"Months 1-3 complete and verified. Output directory: {output_dir}")


@app.command(hidden=True)
def verify_repository(
    output_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
) -> None:
    """Verify that generated month 1-3 artifacts are present, nonempty, and internally valid."""
    verify_output_repository(output_dir)
    info(f"Repository verification passed. Report written to {output_dir / 'verification_report.json'}")


@app.command(hidden=True)
def verify_structured_outputs(
    output_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
) -> None:
    """Verify Month 1/2 output when pairwise retrieval data is not available."""
    verify_structured_analysis(output_dir)
    info(
        "Structured-analysis verification passed. Report written to "
        f"{output_dir / 'structured_analysis_verification_report.json'}"
    )


@app.command("review")
def review(
    review_csv: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_json: Annotated[Path, typer.Option()],
) -> None:
    """Calculate extraction precision after the manual review CSV is labeled."""
    summary = summarize_manual_review(review_csv, output_json)
    labeled = summary["overall"]["action_correct"]["labeled"]
    info(f"Summarized {labeled} labeled actions. Report written to {output_json}")


@app.command("search")
def search(
    query: Annotated[str, typer.Option()],
    clips_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    month1_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    month2_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output_json: Annotated[Path | None, typer.Option()] = None,
    top_k: Annotated[int, typer.Option(min=1)] = 3,
    method: Annotated[SearchMethodChoice, typer.Option()] = SearchMethodChoice.hybrid,
    hybrid_alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    max_per_video: Annotated[int | None, typer.Option(min=1)] = None,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
) -> None:
    """Search the canonical clip corpus with lexical, structured, or hybrid rank."""
    results = rank_indexed_clips(
        query_text=query,
        clips_jsonl=clips_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        spacy_model=spacy_model,
        top_k=top_k,
        method=method.value,
        hybrid_alpha=hybrid_alpha,
        max_per_video=max_per_video,
    )
    if output_json is not None:
        write_search_results(output_json, results)
        info(f"Search results written to {output_json}")
    typer.echo(json.dumps(results, indent=2, sort_keys=True))


@app.command("compare")
def compare(
    query: Annotated[str, typer.Option()],
    clips_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    month1_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    month2_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output_json: Annotated[Path, typer.Option()],
    original_clip_id: Annotated[list[str] | None, typer.Option()] = None,
    top_k: Annotated[int, typer.Option(min=1)] = 3,
    challenger_method: Annotated[
        ChallengerMethodChoice, typer.Option()
    ] = ChallengerMethodChoice.hybrid,
    hybrid_alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
) -> None:
    """Describe how an explicit old set or lexical baseline differs from a new set."""
    results = compare_result_sets(
        query_text=query,
        clips_jsonl=clips_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        spacy_model=spacy_model,
        top_k=top_k,
        original_clip_ids=original_clip_id,
        challenger_method=challenger_method.value,
        hybrid_alpha=hybrid_alpha,
    )
    write_comparison_results(output_json, results)
    info(f"Comparison written to {output_json}")
    typer.echo(json.dumps(results, indent=2, sort_keys=True))


@app.command("benchmark")
def benchmark(
    clips_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    month1_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    month2_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
    hybrid_alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
) -> None:
    """Run a title-to-description benchmark without candidate-title leakage."""
    paths = run_field_heldout_benchmark(
        clips_jsonl=clips_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        output_dir=output_dir,
        spacy_model=spacy_model,
        hybrid_alpha=hybrid_alpha,
    )
    info(f"Benchmark complete. Summary written to {paths['summary']}")


@app.command("compare-batch")
def compare_batch(
    comparisons_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    clips_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    month1_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    month2_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
    challenger_method: Annotated[
        ChallengerMethodChoice, typer.Option()
    ] = ChallengerMethodChoice.hybrid,
    top_k: Annotated[int, typer.Option(min=1)] = 3,
    hybrid_alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    timestamp_tolerance_seconds: Annotated[float, typer.Option(min=0.0)] = 0.05,
    spacy_model: Annotated[str, typer.Option()] = DEFAULT_SPACY_MODEL,
) -> None:
    """Compare many supplied original rankings and create a blinded review sheet."""
    paths = run_batch_comparison(
        comparisons_jsonl=comparisons_jsonl,
        clips_jsonl=clips_jsonl,
        month1_dir=month1_dir,
        month2_dir=month2_dir,
        output_dir=output_dir,
        spacy_model=spacy_model,
        challenger_method=challenger_method.value,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
    )
    info(f"Batch comparison complete. Summary written to {paths['summary']}")
    info(f"Blind review worksheet written to {paths['blind_review']}")


@app.command("score-review")
def score_review(
    rankings_jsonl: Annotated[Path, typer.Option(exists=True, readable=True)],
    review_csv: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_json: Annotated[Path, typer.Option()],
    random_seed: Annotated[int, typer.Option()] = DEFAULT_RANDOM_SEED,
) -> None:
    """Score a completed blinded old-vs-new relevance worksheet."""
    report = score_batch_review(
        rankings_jsonl=rankings_jsonl,
        review_csv=review_csv,
        output_json=output_json,
        seed=random_seed,
    )
    scored_steps = report["dimensions"]["overall_relevant"]["scored_steps"]
    info(f"Scored {scored_steps} fully labeled steps. Report written to {output_json}")


if __name__ == "__main__":
    app()
