from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BaseRecord(BaseModel):
    source_row_sha256: str | None = None


class ClipRecord(BaseRecord):
    clip_id: str
    project_id: str | None = None
    step_id: str | None = None
    video_id: str | None = None
    url: str | None = None
    title: str | None = None
    description: str | None = None
    summary: str | None = None
    transcript: str | None = None
    automatic_captions: str | None = None
    whisper_transcript: str | None = None
    gemini_metadata: dict[str, Any] = Field(default_factory=dict)
    dense_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("clip_id")
    @classmethod
    def clip_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clip_id is blank")
        return value.strip()


class StepRecord(BaseRecord):
    step_id: str
    project_id: str | None = None
    step_index: int | None = None
    title: str | None = None
    description: str | None = None
    tools: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    dense_embeddings: dict[str, list[float]] = Field(default_factory=dict)

    @field_validator("step_id")
    @classmethod
    def step_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("step_id is blank")
        return value.strip()


class PairwiseComparison(BaseRecord):
    comparison_id: str
    step_id: str
    clip_a_id: str
    clip_b_id: str
    winner_clip_id: str
    loser_clip_id: str | None = None
    annotator_id: str | None = None
    source: str | None = None
    created_at: datetime | None = None

    @model_validator(mode="after")
    def winner_must_be_a_or_b(self) -> "PairwiseComparison":
        if self.winner_clip_id not in {self.clip_a_id, self.clip_b_id}:
            raise ValueError("winner_clip_id must equal clip_a_id or clip_b_id")
        if self.loser_clip_id is None:
            self.loser_clip_id = self.clip_b_id if self.winner_clip_id == self.clip_a_id else self.clip_a_id
        return self


class TextSegment(BaseModel):
    record_type: Literal["clip", "step"]
    record_id: str
    source_field: str
    text: str
    source_confidence: float = 1.0


class ActionTriple(BaseModel):
    record_type: Literal["clip", "step"]
    record_id: str
    source_field: str
    action: str
    action_lemma: str
    action_text: str
    object_text: str | None = None
    object_lemmas: list[str] = Field(default_factory=list)
    tool_text: str | None = None
    tool_lemmas: list[str] = Field(default_factory=list)
    context_tool_lemmas: list[str] = Field(default_factory=list)
    material_text: str | None = None
    material_lemmas: list[str] = Field(default_factory=list)
    context_material_lemmas: list[str] = Field(default_factory=list)
    negated: bool = False
    sentence: str
    token_start: int | None = None
    token_end: int | None = None
    extraction_method: str
    confidence: float = 1.0


class VerbNetMapping(BaseModel):
    action_lemma: str
    verbnet_classes: list[str] = Field(default_factory=list)
    wordnet_synsets: list[str] = Field(default_factory=list)
    has_mapping: bool


class FrameNetMapping(BaseModel):
    action_lemma: str
    frames: list[str] = Field(default_factory=list)
    lexical_units: list[str] = Field(default_factory=list)
    has_mapping: bool


class SrlRole(BaseModel):
    record_type: Literal["clip", "step"]
    record_id: str
    source_field: str
    predicate_lemma: str
    predicate_text: str
    agent: str | None = None
    patient: str | None = None
    instrument: str | None = None
    location_or_scope: str | None = None
    sentence: str
    extraction_method: str = "dependency_srl_v1"
    confidence: float = 1.0


class TaxonomyAssignment(BaseModel):
    action_lemma: str
    cluster_id: int
    cluster_label: str
    support_count: int
    representative_objects: list[str] = Field(default_factory=list)
    representative_tools: list[str] = Field(default_factory=list)
    representative_materials: list[str] = Field(default_factory=list)


class ScoreRow(BaseModel):
    step_id: str
    clip_id: str
    dense_score: float | None = None
    structured_score: float | None = None
    hybrid_score: float | None = None
    action_match: float | None = None
    object_match: float | None = None
    tool_match: float | None = None
    material_match: float | None = None
    taxonomy_match: float | None = None
    framenet_match: float | None = None
    verbnet_match: float | None = None


class PairwiseEvaluationRow(BaseModel):
    comparison_id: str
    step_id: str
    clip_a_id: str
    clip_b_id: str
    winner_clip_id: str
    model_name: str
    score_a: float | None
    score_b: float | None
    predicted_winner_clip_id: str | None
    correct: bool | None
    tie: bool
