"""Reproducibility metadata shared by retrieval experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from action_semantics.io_utils import sha256_file
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
    return {
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
        "spacy_model": spacy_model,
        "structured_scorer": STRUCTURED_SCORER_VERSION,
        "taxonomy_used_for_ranking": False,
        "taxonomy_used_for_diagnostics": True,
    }
