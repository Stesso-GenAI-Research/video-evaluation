from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from action_semantics.models import ActionTriple, TextSegment
from action_semantics.text import normalize_term

_OBJECT_DEPS = {"dobj", "obj", "pobj", "attr", "dative", "oprd"}
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent"}
_MATERIAL_PREPS = {"into", "onto", "on", "in", "over", "under", "through", "around"}
_TOOL_PREPS = {"with", "using", "via", "by"}
_NEG_DEPS = {"neg"}


@lru_cache(maxsize=4)
def load_spacy_model(model_name: str) -> Language:
    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is not installed. Install it before running extraction."
        ) from exc


def _clean_span_text(span: Span | list[Token] | tuple[Token, ...] | None) -> str | None:
    if not span:
        return None
    if isinstance(span, Span):
        text = span.text
    else:
        ordered = sorted(span, key=lambda t: t.i)
        text = " ".join(token.text for token in ordered)
    value = normalize_term(text)
    return value or None


def _content_lemmas(tokens: Iterable[Token]) -> list[str]:
    lemmas: list[str] = []
    for token in tokens:
        lemma = normalize_term(token.lemma_ or token.text)
        # Object identity should come from nouns. Adjectives such as "old"
        # otherwise make unrelated results (old faucet / old control) look
        # like object matches.
        if lemma and not token.is_stop and token.pos_ in {"NOUN", "PROPN", "VERB", "NUM"}:
            lemmas.append(lemma)
    return sorted(set(lemmas))


def _subtree_without_nested_verbs(token: Token) -> list[Token]:
    return [
        child
        for child in token.subtree
        if child == token or child.head == token or child.pos_ not in {"VERB", "AUX"}
    ]


def _first_object_span(verb: Token) -> list[Token]:
    direct = [child for child in verb.children if child.dep_ in _OBJECT_DEPS]
    if direct:
        return _subtree_without_nested_verbs(direct[0])
    for child in verb.children:
        if child.dep_ == "prep" and child.lemma_.lower() not in _TOOL_PREPS:
            pobj = [grand for grand in child.children if grand.dep_ in _OBJECT_DEPS]
            if pobj:
                return _subtree_without_nested_verbs(pobj[0])
    return []


def _prep_object_span(verb: Token, prep_lemmas: set[str]) -> list[Token]:
    # spaCy sometimes attaches "with a wrench" to the direct object rather
    # than the verb. Search the full verb phrase so common instructions such
    # as "tighten the connection with a wrench" still preserve the tool.
    for token in verb.subtree:
        if token.dep_ == "prep" and token.lemma_.lower() in prep_lemmas:
            objects = [grand for grand in token.children if grand.dep_ in _OBJECT_DEPS]
            if objects:
                return _subtree_without_nested_verbs(objects[0])
    return []


def _has_subject(verb: Token) -> bool:
    return any(child.dep_ in _SUBJECT_DEPS for child in verb.children)


def _is_negated(verb: Token) -> bool:
    return any(child.dep_ in _NEG_DEPS or child.lower_ in {"not", "n't", "never"} for child in verb.children)


def _verb_confidence(verb: Token, object_tokens: list[Token], tool_tokens: list[Token]) -> float:
    score = 0.45
    if object_tokens:
        score += 0.25
    if tool_tokens:
        score += 0.15
    if _has_subject(verb):
        score += 0.05
    if verb.pos_ == "VERB":
        score += 0.10
    return min(score, 1.0)


def triples_from_doc(segment: TextSegment, doc: Doc) -> list[ActionTriple]:
    triples: list[ActionTriple] = []
    for sentence in doc.sents:
        for token in sentence:
            # Auxiliary tokens (for example "will" in "will remove") are
            # grammatical support, not the instructional action.  The main
            # verb still carries the object and tool dependencies we need.
            if token.pos_ != "VERB":
                continue
            object_tokens = _first_object_span(token)
            tool_tokens = _prep_object_span(token, _TOOL_PREPS)
            material_tokens = _prep_object_span(token, _MATERIAL_PREPS)
            action_lemma = normalize_term(token.lemma_ or token.text)
            if not action_lemma:
                continue
            triples.append(
                ActionTriple(
                    record_type=segment.record_type,
                    record_id=segment.record_id,
                    source_field=segment.source_field,
                    action=normalize_term(token.text),
                    action_lemma=action_lemma,
                    action_text=token.text,
                    object_text=_clean_span_text(object_tokens),
                    object_lemmas=_content_lemmas(object_tokens),
                    tool_text=_clean_span_text(tool_tokens),
                    tool_lemmas=_content_lemmas(tool_tokens),
                    material_text=_clean_span_text(material_tokens),
                    material_lemmas=_content_lemmas(material_tokens),
                    negated=_is_negated(token),
                    sentence=sentence.text.strip(),
                    token_start=token.i,
                    token_end=token.i + 1,
                    extraction_method="spacy_dependency_v1",
                    confidence=round(_verb_confidence(token, object_tokens, tool_tokens), 4)
                    * segment.source_confidence,
                )
            )
    return triples


def extract_triples(segments: Iterable[TextSegment], model_name: str) -> list[ActionTriple]:
    nlp = load_spacy_model(model_name)
    segment_list = list(segments)
    parser_texts = [
        segment.text[:1].lower() + segment.text[1:]
        if segment.source_field in {"title", "query"} and segment.text
        else segment.text
        for segment in segment_list
    ]
    docs = nlp.pipe(parser_texts, batch_size=64)
    rows: list[ActionTriple] = []
    for segment, doc in zip(segment_list, docs, strict=True):
        rows.extend(triples_from_doc(segment, doc))
    return rows


def triples_to_lookup(triples: Iterable[ActionTriple]) -> dict[tuple[str, str], list[ActionTriple]]:
    lookup: dict[tuple[str, str], list[ActionTriple]] = {}
    for triple in triples:
        lookup.setdefault((triple.record_type, triple.record_id), []).append(triple)
    return lookup


def triple_dict_for_analysis(triple: ActionTriple) -> dict[str, Any]:
    data = triple.model_dump(mode="json")
    data["object_lemmas_joined"] = ";".join(triple.object_lemmas)
    data["tool_lemmas_joined"] = ";".join(triple.tool_lemmas)
    data["context_tool_lemmas_joined"] = ";".join(triple.context_tool_lemmas)
    data["material_lemmas_joined"] = ";".join(triple.material_lemmas)
    data["context_material_lemmas_joined"] = ";".join(triple.context_material_lemmas)
    return data
