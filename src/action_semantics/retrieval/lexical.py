"""Small, explicit TF-IDF baseline shared by search and experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from action_semantics.models import ClipRecord


PRODUCTION_CANDIDATE_FIELDS = [
    "clip.title (2x weight)",
    "clip.description",
    "clip.goal",
    "clip.tools",
    "clip.supplies",
    "video.title",
    "video.summary",
    "video.goal",
    "video.category",
]


@dataclass(frozen=True)
class TfidfIndex:
    """One fitted candidate matrix that can score many queries cheaply."""

    clip_ids: list[str]
    vectorizer: TfidfVectorizer
    candidate_matrix: Any

    @classmethod
    def from_clips(
        cls,
        clips: list[ClipRecord],
        *,
        documents: list[str] | None = None,
    ) -> "TfidfIndex":
        candidate_documents = documents or [production_candidate_text(clip) for clip in clips]
        if len(candidate_documents) != len(clips):
            raise ValueError("The number of TF-IDF documents must equal the number of clips.")
        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True
        )
        candidate_matrix = vectorizer.fit_transform(candidate_documents)
        return cls(
            clip_ids=[clip.clip_id for clip in clips],
            vectorizer=vectorizer,
            candidate_matrix=candidate_matrix,
        )

    def scores(self, query_text: str) -> dict[str, float]:
        query_vector = self.vectorizer.transform([query_text])
        scores = (self.candidate_matrix @ query_vector.T).toarray().ravel()
        return {
            clip_id: float(score)
            for clip_id, score in zip(self.clip_ids, scores, strict=True)
        }


def _strings(values: Iterable[object]) -> list[str]:
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def production_candidate_text(clip: ClipRecord) -> str:
    """Build the documented text view used by production lexical search."""
    clip_metadata = clip.gemini_metadata.get("clip", {})
    video_metadata = clip.gemini_metadata.get("source_video", {})
    if not isinstance(clip_metadata, dict):
        clip_metadata = {}
    if not isinstance(video_metadata, dict):
        video_metadata = {}
    title = clip.title or ""
    inventory: list[object] = []
    for value_key, item_key in (("tools", "tool_items"), ("supplies", "supply_items")):
        values = clip_metadata.get(value_key, [])
        inventory.extend(values if isinstance(values, list) else [values])
        items = clip_metadata.get(item_key, [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                inventory.append(item.get("name"))
                alternatives = item.get("alternatives", [])
                if isinstance(alternatives, list):
                    inventory.extend(alternatives)
    category = video_metadata.get("category")
    category_name = category.get("name") if isinstance(category, dict) else category
    return " ".join(
        _strings(
            [
                title,
                title,
                clip.description,
                clip.summary,
                *inventory,
                video_metadata.get("title"),
                video_metadata.get("summary"),
                video_metadata.get("goal"),
                video_metadata.get("category_name") or category_name,
            ]
        )
    )


def tfidf_scores(
    query_text: str,
    clips: list[ClipRecord],
    *,
    documents: list[str] | None = None,
) -> dict[str, float]:
    """Fit a deterministic candidate-only TF-IDF baseline and score one query."""
    if not clips:
        return {}
    return TfidfIndex.from_clips(clips, documents=documents).scores(query_text)
