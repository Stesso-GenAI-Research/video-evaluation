"""Small, explicit TF-IDF baseline shared by search and experiments."""

from __future__ import annotations

from collections.abc import Iterable

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
    tools = clip_metadata.get("tools", [])
    supplies = clip_metadata.get("supplies", [])
    inventory = [
        *(tools if isinstance(tools, list) else [tools]),
        *(supplies if isinstance(supplies, list) else [supplies]),
    ]
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
    candidate_documents = documents or [production_candidate_text(clip) for clip in clips]
    if len(candidate_documents) != len(clips):
        raise ValueError("The number of TF-IDF documents must equal the number of clips.")
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    candidate_matrix = vectorizer.fit_transform(candidate_documents)
    query_vector = vectorizer.transform([query_text])
    scores = (candidate_matrix @ query_vector.T).toarray().ravel()
    return {
        clip.clip_id: float(score)
        for clip, score in zip(clips, scores, strict=True)
    }
