"""Reproducibility metadata shared by retrieval experiment artifacts."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from action_semantics.io_utils import sha256_file
from action_semantics.retrieval.lexical import PRODUCTION_CANDIDATE_FIELDS
from action_semantics.retrieval.scorers import STRUCTURED_SCORER_VERSION


def build_retrieval_provenance(
    *,
    clips_jsonl: Path,
    month1_dir: Path,
    month2_dir: Path,
    spacy_model: str,
) -> dict[str, Any]:
    """Hash every artifact that can affect a structured retrieval report."""
    artifact_paths = {
        "clips": clips_jsonl,
        "action_object_tool_triples": month1_dir / "action_object_tool_triples.jsonl",
        "verbnet_mappings": month1_dir / "verbnet_mappings.jsonl",
        "framenet_mappings": month2_dir / "framenet_mappings.jsonl",
        "diy_actionnet": month2_dir / "diy_actionnet_v1.jsonl",
    }
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot record retrieval provenance because artifacts are missing: "
            + ", ".join(missing)
        )
    def installed_version(distribution: str) -> str | None:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return None

    return {
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
        "spacy_model": spacy_model,
        "spacy_version": installed_version("spacy"),
        "spacy_model_version": installed_version(spacy_model.replace("_", "-")),
        "structured_scorer": STRUCTURED_SCORER_VERSION,
        "production_lexical_fields": PRODUCTION_CANDIDATE_FIELDS,
        "taxonomy_used_for_ranking": False,
        "taxonomy_used_for_diagnostics": True,
    }
