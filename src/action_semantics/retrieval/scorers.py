from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from action_semantics.extraction.triples import triples_to_lookup
from action_semantics.models import (
    ActionTriple,
    ClipRecord,
    FrameNetMapping,
    ScoreRow,
    StepRecord,
    TaxonomyAssignment,
    VerbNetMapping,
)
from action_semantics.retrieval.embeddings import mean_dense_score


STRUCTURED_SCORER_VERSION = "aligned-action-object-context-v4"


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {value for value in left if value}
    right_set = {value for value in right if value}
    if not left_set and not right_set:
        return 0.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


@dataclass(frozen=True)
class StructuredResources:
    triples: list[ActionTriple]
    verbnet: list[VerbNetMapping]
    framenet: list[FrameNetMapping]
    taxonomy: list[TaxonomyAssignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "triple_lookup", triples_to_lookup(self.triples))
        object.__setattr__(
            self,
            "verbnet_lookup",
            {row.action_lemma: set(row.verbnet_classes) for row in self.verbnet},
        )
        object.__setattr__(
            self,
            "framenet_lookup",
            {row.action_lemma: set(row.frames) for row in self.framenet},
        )
        object.__setattr__(
            self,
            "taxonomy_lookup",
            {row.action_lemma: int(row.cluster_id) for row in self.taxonomy},
        )

    def triples_for(self, record_type: str, record_id: str) -> list[ActionTriple]:
        return self.triple_lookup.get((record_type, record_id), [])


@dataclass(frozen=True)
class StructuredWeights:
    """Weights for one aligned action-object-context comparison.

    VerbNet and FrameNet are fallbacks inside the action component.  They are
    not added again as independent evidence.  The exploratory taxonomy is
    deliberately excluded from the production score until it has been manually
    evaluated.  Spatial scope is diagnostic-only because the current parser
    does not measure temporal step boundaries.
    """

    action: float = 0.55
    object: float = 0.35
    context: float = 0.10

    def normalized(
        self,
        *,
        has_object: bool = True,
        has_context: bool = True,
    ) -> "StructuredWeights":
        object_weight = self.object if has_object else 0.0
        context_weight = self.context if has_context else 0.0
        total = self.action + object_weight + context_weight
        if total <= 0:
            raise ValueError("Structured weights must sum to a positive value.")
        return StructuredWeights(
            action=self.action / total,
            object=object_weight / total,
            context=context_weight / total,
        )


def _query_term_coverage(query_terms: Iterable[str], candidate_terms: Iterable[str]) -> float:
    """Measure how much of the short query is present in the candidate.

    Candidate descriptions are usually much longer than queries.  A symmetric
    Jaccard score incorrectly punishes a correct candidate for containing extra
    detail, so the denominator is the query vocabulary only.
    """
    query = {value for value in query_terms if value}
    candidate = {value for value in candidate_terms if value}
    if not query or not candidate:
        return 0.0
    return len(query & candidate) / len(query)


def _action_similarity(
    query: ActionTriple,
    candidate: ActionTriple,
    resources: StructuredResources,
) -> tuple[float, float, float, float]:
    exact = float(query.action_lemma == candidate.action_lemma)
    verbnet = jaccard(
        resources.verbnet_lookup.get(query.action_lemma, set()),
        resources.verbnet_lookup.get(candidate.action_lemma, set()),
    )
    framenet = jaccard(
        resources.framenet_lookup.get(query.action_lemma, set()),
        resources.framenet_lookup.get(candidate.action_lemma, set()),
    )
    if exact:
        return 1.0, exact, verbnet, framenet
    # These mappings are useful backoffs, but weaker than the same verb.
    return max(0.80 * verbnet, 0.70 * framenet), exact, verbnet, framenet


def _taxonomy_pair_match(
    query: ActionTriple,
    candidate: ActionTriple,
    lookup: dict[str, int],
) -> float:
    left = lookup.get(query.action_lemma)
    right = lookup.get(candidate.action_lemma)
    return float(left is not None and left == right)


def _triple_pair_score(
    query: ActionTriple,
    candidate: ActionTriple,
    resources: StructuredResources,
    base_weights: StructuredWeights,
) -> dict[str, float]:
    action, exact, verbnet, framenet = _action_similarity(query, candidate, resources)
    object_score = _query_term_coverage(query.object_lemmas, candidate.object_lemmas)
    # A phrase introduced by "with" or "using" can name either a tool
    # (screwdriver) or a consumable supply (primer).  Compare it with both
    # inventories instead of forcing that ambiguous phrase into one class.
    query_tool_context = set(query.tool_lemmas) | set(query.context_tool_lemmas)
    query_supply_context = set(query.context_material_lemmas)
    candidate_tools = set(candidate.tool_lemmas) | set(candidate.context_tool_lemmas)
    candidate_supplies = set(candidate.context_material_lemmas)
    tool_score = _query_term_coverage(query_tool_context, candidate_tools)
    supply_score = max(
        _query_term_coverage(query_tool_context, candidate_supplies),
        _query_term_coverage(query_supply_context, candidate_supplies),
    )
    context_score = max(tool_score, supply_score)

    # ``material_lemmas`` is a legacy field name for objects of prepositions
    # such as "on", "in", and "into".  It is spatial location/scope, not a
    # supply inventory, and is intentionally diagnostic-only.
    scope_score = _query_term_coverage(
        query.material_lemmas, candidate.material_lemmas
    )
    taxonomy = _taxonomy_pair_match(query, candidate, resources.taxonomy_lookup)

    # A positive/negative mismatch changes the meaning of an instruction.  It
    # must not be rescued by a shared object or tool.
    if query.negated != candidate.negated:
        action = 0.0
        total = 0.0
    else:
        weights = base_weights.normalized(
            has_object=bool(query.object_lemmas),
            has_context=bool(query_tool_context or query_supply_context),
        )
        component_score = (
            weights.action * action
            + weights.object * object_score
            + weights.context * context_score
        )
        # Object/tool evidence is meaningful only when the actions are at
        # least compatible.  This prevents an unrelated action on the same
        # object from being ranked as a good semantic match.
        confidence = 0.90 + 0.10 * min(query.confidence, candidate.confidence)
        total = action * component_score * confidence

    return {
        "structured_score": float(total),
        "action_match": float(action),
        "exact_action_match": float(exact),
        "verbnet_match": float(verbnet),
        "framenet_match": float(framenet),
        "taxonomy_match": float(taxonomy),
        "object_match": float(object_score),
        "context_match": float(context_score),
        "tool_match": float(tool_score),
        "supply_match": float(supply_score),
        "scope_match": float(scope_score),
    }


def structured_score(
    step_id: str,
    clip_id: str,
    resources: StructuredResources,
    weights: StructuredWeights | None = None,
) -> dict[str, float]:
    return structured_score_for_triples(
        resources.triples_for("step", step_id),
        clip_id,
        resources,
        weights,
    )


def structured_score_for_triples(
    query_triples: list[ActionTriple],
    clip_id: str,
    resources: StructuredResources,
    weights: StructuredWeights | None = None,
) -> dict[str, float]:
    """Score already-parsed query triples without rebuilding corpus lookups."""
    clip_triples = resources.triples_for("clip", clip_id)
    metric_names = (
        "structured_score",
        "action_match",
        "exact_action_match",
        "verbnet_match",
        "framenet_match",
        "taxonomy_match",
        "object_match",
        "context_match",
        "tool_match",
        "supply_match",
        "scope_match",
    )
    if not query_triples or not clip_triples:
        return {name: 0.0 for name in metric_names}

    base_weights = weights or StructuredWeights()
    aligned: list[dict[str, float]] = []
    for query in query_triples:
        candidates = [
            _triple_pair_score(query, candidate, resources, base_weights)
            for candidate in clip_triples
        ]
        aligned.append(
            max(
                candidates,
                key=lambda row: (
                    row["structured_score"],
                    row["action_match"],
                    row["object_match"],
                    row["context_match"],
                ),
            )
        )
    return {
        name: float(sum(row[name] for row in aligned) / len(aligned))
        for name in metric_names
    }


def score_step_clip(
    step: StepRecord,
    clip: ClipRecord,
    resources: StructuredResources,
    dense_keys: list[str] | None = None,
    hybrid_alpha: float = 0.5,
) -> ScoreRow:
    dense = mean_dense_score(step.dense_embeddings, clip.dense_embeddings, dense_keys)
    structured_parts = structured_score(step.step_id, clip.clip_id, resources)
    structured = structured_parts["structured_score"]
    hybrid = None
    if dense is not None:
        dense_01 = (dense + 1.0) / 2.0
        hybrid = hybrid_alpha * dense_01 + (1.0 - hybrid_alpha) * structured
    return ScoreRow(
        step_id=step.step_id,
        clip_id=clip.clip_id,
        dense_score=dense,
        structured_score=structured,
        hybrid_score=hybrid,
        action_match=structured_parts["action_match"],
        object_match=structured_parts["object_match"],
        context_match=structured_parts["context_match"],
        tool_match=structured_parts["tool_match"],
        supply_match=structured_parts["supply_match"],
        scope_match=structured_parts["scope_match"],
        taxonomy_match=structured_parts["taxonomy_match"],
        framenet_match=structured_parts["framenet_match"],
        verbnet_match=structured_parts["verbnet_match"],
    )


def resources_from_files(month1_dir: Any, month2_dir: Any) -> StructuredResources:
    from pathlib import Path

    from action_semantics.io_utils import read_jsonl_model

    m1 = Path(month1_dir)
    m2 = Path(month2_dir)
    return StructuredResources(
        triples=read_jsonl_model(m1 / "action_object_tool_triples.jsonl", ActionTriple),
        verbnet=read_jsonl_model(m1 / "verbnet_mappings.jsonl", VerbNetMapping),
        framenet=read_jsonl_model(m2 / "framenet_mappings.jsonl", FrameNetMapping),
        taxonomy=read_jsonl_model(m2 / "diy_actionnet_v1.jsonl", TaxonomyAssignment),
    )
