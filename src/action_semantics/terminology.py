"""Conservative, versioned terminology view; original annotations remain intact.

Only unambiguous noun phrases are activated. Actions, states, generic 'tap',
'mud', 'Sheetrock' and substitutable tools intentionally have no rewrite rule.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import ActionTriple
from .text import normalize_term

TERMINOLOGY_VERSION = "conservative-noun-aliases-v1"
ALIASES = {
    "gypsum board": "drywall",
    "plasterboard": "drywall",
    "measuring tape": "tape measure",
    "crescent wrench": "adjustable wrench",
    "adjustable spanner": "adjustable wrench",
    "shop vac": "shop vacuum",
    "mitre saw": "miter saw",
    "allen key": "hex key",
    "allen wrench": "hex key",
}
_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(
        re.escape(term).replace(r"\ ", r"\s+")
        for term in sorted(ALIASES, key=len, reverse=True)
    ) + r")(?!\w)", re.I,
)


def normalize_terminology(text: str) -> str:
    """One boundary-aware pass; retain attributes, negation and action direction."""
    return _PATTERN.sub(lambda m: ALIASES[" ".join(m.group().lower().split())], text)


def terminology_provenance(enabled: bool, primary_inventory_only: bool = False) -> dict:
    return {
        "enabled": enabled,
        "version": TERMINOLOGY_VERSION if enabled else None,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inventory_alternatives_used": not primary_inventory_only,
    }


def normalize_terms(terms: list[str], phrase: str | None = None) -> list[str]:
    """Use phrase spans to reconcile multiword aliases before comparing lemmas."""
    result = {normalize_terminology(term) for term in terms}
    # Inventories contain complete names plus their component tokens. Direct
    # object/tool phrases supply ordering which a sorted lemma list loses.
    for text in [*terms, phrase or ""]:
        for match in _PATTERN.finditer(text):
            original = normalize_term(match.group())
            canonical = normalize_terminology(original)
            result.difference_update(original.split())
            result.update(canonical.split())
    return sorted(result)


def normalize_triple(triple: ActionTriple) -> ActionTriple:
    updates = {}
    for field, phrase in (
        ("object_lemmas", triple.object_text), ("tool_lemmas", triple.tool_text),
        ("material_lemmas", triple.material_text),
        ("context_tool_lemmas", None), ("context_material_lemmas", None),
    ):
        updates[field] = normalize_terms(getattr(triple, field), phrase)
    return triple.model_copy(update=updates)
