#!/usr/bin/env bash
# Run the local checks and Project 1 pipelines without remembering each command.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
SAMPLE_JSONL="${SAMPLE_JSONL:-$ROOT_DIR/indexed-videos-250.jsonl}"
PROJECT_OUTPUT_DIR="${PROJECT_OUTPUT_DIR:-$ROOT_DIR/project1_outputs}"
SAMPLE_OUTPUT_DIR="${SAMPLE_OUTPUT_DIR:-$PROJECT_OUTPUT_DIR/indexed-video-sample}"
COMMAND="${1:-all}"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_local_pipeline.sh [setup|test|build|sample|search|compare|compare-batch|score-review|benchmark|review|all]

  setup   Create a Python 3.11-3.13 virtual environment and install dependencies.
  test    Compile the project, run pytest, run Ruff, and show CLI help.
  build   Build the action-semantic index from the IndexedVideo JSONL file.
  sample  Backward-compatible alias for build.
  search  Search clips; extra flags such as --method and --top-k are forwarded.
  compare Diff an explicit old/lexical set against structured or hybrid search.
  compare-batch  Compare a JSONL file of real original rankings and make a blind review CSV.
  score-review   Score the completed batch-comparison blind review worksheet.
  benchmark Run the field-held-out lexical/structured/hybrid experiment.
  review  Summarize the human labels added to manual_review_sample.csv.
  all     Test, build, and benchmark. Setup runs only if the environment is absent.

Optional environment variables:
  SAMPLE_JSONL        Path to indexed-videos-250.jsonl for the sample command.
  SAMPLE_OUTPUT_DIR   Destination for sample artifacts.
  PROJECT_OUTPUT_DIR  Parent output directory; defaults to project1_outputs/.
  VENV_DIR            Virtual environment location; defaults to .venv/.

EOF
}

find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  echo "Python 3.11, 3.12, or 3.13 is required. Install one and try again." >&2
  return 1
}

setup() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    local python_bin
    python_bin="$(find_python)"
    echo "Creating virtual environment with $python_bin"
    "$python_bin" -m venv "$VENV_DIR"
  fi

  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR[dev]"
  "$VENV_DIR/bin/python" -m spacy download en_core_web_sm
  "$VENV_DIR/bin/python" -m nltk.downloader verbnet wordnet omw-1.4 framenet_v17
}

ensure_environment() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    setup
  fi
}

run_tests() {
  ensure_environment
  "$VENV_DIR/bin/python" -m compileall -q "$ROOT_DIR/src" "$ROOT_DIR/tests"
  "$VENV_DIR/bin/python" -m pytest -q
  "$VENV_DIR/bin/ruff" check "$ROOT_DIR/src" "$ROOT_DIR/tests"
  bash -n "$ROOT_DIR/scripts/run_local_pipeline.sh"
  "$VENV_DIR/bin/action-semantics" --help
}

run_sample() {
  ensure_environment
  if [[ ! -f "$SAMPLE_JSONL" ]]; then
    echo "Sample file not found: $SAMPLE_JSONL" >&2
    echo "Set SAMPLE_JSONL to the private IndexedVideo JSONL path and try again." >&2
    return 1
  fi
  "$VENV_DIR/bin/action-semantics" build-index \
    --indexed-videos-jsonl "$SAMPLE_JSONL" \
    --output-dir "$SAMPLE_OUTPUT_DIR" \
    --min-taxonomy-support 2
  echo
  echo "Sample report: $SAMPLE_OUTPUT_DIR/sample_analysis_report.json"
  echo "Review worksheet: $SAMPLE_OUTPUT_DIR/quality/manual_review_sample.csv"
}

index_is_current() {
  [[ -f "$SAMPLE_JSONL" && -d "$SAMPLE_OUTPUT_DIR" ]] || return 1
  "$VENV_DIR/bin/action-semantics" index-current \
    --indexed-videos-jsonl "$SAMPLE_JSONL" \
    --output-dir "$SAMPLE_OUTPUT_DIR" \
    >/dev/null 2>&1
}

ensure_sample_index() {
  if ! index_is_current; then
    echo "The index is missing or belongs to a different SAMPLE_JSONL; rebuilding it."
    run_sample
  fi
}

summarize_review() {
  ensure_environment
  local review_csv="$SAMPLE_OUTPUT_DIR/quality/manual_review_sample.csv"
  if [[ ! -f "$review_csv" ]]; then
    echo "Review worksheet not found: $review_csv" >&2
    echo "Run the sample command first." >&2
    return 1
  fi
  "$VENV_DIR/bin/action-semantics" review \
    --review-csv "$review_csv" \
    --output-json "$SAMPLE_OUTPUT_DIR/quality/manual_review_results.json"
}

search_sample() {
  ensure_environment
  local query="${2:-${SEARCH_QUERY:-}}"
  if [[ -z "$query" ]]; then
    echo 'Provide a query, for example: ./scripts/run_local_pipeline.sh search "remove old faucet"' >&2
    return 1
  fi
  ensure_sample_index
  local slug
  slug="$(printf '%s' "$query" | tr '[:upper:] ' '[:lower:]_' | tr -cd 'a-z0-9_-')"
  local query_hash
  query_hash="$(printf '%s|' "$query" "${@:3}" | shasum -a 256 | cut -c 1-8)"
  slug="${slug:0:48}_${query_hash}"
  "$VENV_DIR/bin/action-semantics" search \
    --query "$query" \
    --clips-jsonl "$SAMPLE_OUTPUT_DIR/input/indexed_video_clips.jsonl" \
    --month1-dir "$SAMPLE_OUTPUT_DIR/month1" \
    --month2-dir "$SAMPLE_OUTPUT_DIR/month2" \
    --output-json "$SAMPLE_OUTPUT_DIR/search_${slug}.json" \
    "${@:3}"
}

compare_sample() {
  ensure_environment
  local query="${2:-${SEARCH_QUERY:-}}"
  if [[ -z "$query" ]]; then
    echo 'Provide a query, for example: ./scripts/run_local_pipeline.sh compare "remove old faucet"' >&2
    return 1
  fi
  ensure_sample_index
  local slug
  slug="$(printf '%s' "$query" | tr '[:upper:] ' '[:lower:]_' | tr -cd 'a-z0-9_-')"
  local query_hash
  query_hash="$(printf '%s|' "$query" "${ORIGINAL_CLIP_IDS:-}" "${@:3}" | shasum -a 256 | cut -c 1-8)"
  slug="${slug:0:48}_${query_hash}"
  local compare_command=(
    "$VENV_DIR/bin/action-semantics" compare
    --query "$query"
    --clips-jsonl "$SAMPLE_OUTPUT_DIR/input/indexed_video_clips.jsonl"
    --month1-dir "$SAMPLE_OUTPUT_DIR/month1"
    --month2-dir "$SAMPLE_OUTPUT_DIR/month2"
    --output-json "$SAMPLE_OUTPUT_DIR/comparison_${slug}.json"
  )
  if [[ -n "${ORIGINAL_CLIP_IDS:-}" ]]; then
    local clip_id
    IFS=',' read -r -a original_ids <<< "$ORIGINAL_CLIP_IDS"
    for clip_id in "${original_ids[@]}"; do
      compare_command+=(--original-clip-id "$clip_id")
    done
  fi
  compare_command+=("${@:3}")
  "${compare_command[@]}"
}

compare_batch_sample() {
  ensure_environment
  local comparisons_jsonl="${2:-}"
  if [[ -z "$comparisons_jsonl" || ! -f "$comparisons_jsonl" ]]; then
    echo 'Provide an existing JSONL file: ./scripts/run_local_pipeline.sh compare-batch path/to/comparisons.jsonl' >&2
    return 1
  fi
  ensure_sample_index
  "$VENV_DIR/bin/action-semantics" compare-batch \
    --comparisons-jsonl "$comparisons_jsonl" \
    --clips-jsonl "$SAMPLE_OUTPUT_DIR/input/indexed_video_clips.jsonl" \
    --month1-dir "$SAMPLE_OUTPUT_DIR/month1" \
    --month2-dir "$SAMPLE_OUTPUT_DIR/month2" \
    --output-dir "$SAMPLE_OUTPUT_DIR/batch-comparison" \
    "${@:3}"
}

score_batch_review_sample() {
  ensure_environment
  local comparison_dir="$SAMPLE_OUTPUT_DIR/batch-comparison"
  if [[ ! -f "$comparison_dir/rankings.jsonl" || ! -f "$comparison_dir/blind_review.csv" ]]; then
    echo "Batch comparison artifacts were not found under: $comparison_dir" >&2
    echo "Run compare-batch first." >&2
    return 1
  fi
  "$VENV_DIR/bin/action-semantics" score-review \
    --rankings-jsonl "$comparison_dir/rankings.jsonl" \
    --review-csv "$comparison_dir/blind_review.csv" \
    --output-json "$comparison_dir/review_scores.json" \
    "${@:2}"
}

benchmark_sample() {
  ensure_environment
  ensure_sample_index
  "$VENV_DIR/bin/action-semantics" benchmark \
    --clips-jsonl "$SAMPLE_OUTPUT_DIR/input/indexed_video_clips.jsonl" \
    --month1-dir "$SAMPLE_OUTPUT_DIR/month1" \
    --month2-dir "$SAMPLE_OUTPUT_DIR/month2" \
    --output-dir "$SAMPLE_OUTPUT_DIR/benchmark" \
    "${@:2}"
  echo "Benchmark summary: $SAMPLE_OUTPUT_DIR/benchmark/benchmark_summary.json"
}

case "$COMMAND" in
  setup) setup ;;
  test) run_tests ;;
  build|sample) run_sample ;;
  search) search_sample "$@" ;;
  compare) compare_sample "$@" ;;
  compare-batch) compare_batch_sample "$@" ;;
  score-review) score_batch_review_sample "$@" ;;
  review) summarize_review ;;
  all)
    run_tests
    run_sample
    benchmark_sample
    ;;
  benchmark) benchmark_sample "$@" ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
