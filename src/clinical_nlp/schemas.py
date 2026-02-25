from __future__ import annotations

from typing import Literal
from typing import Optional
from pydantic import BaseModel, Field


EntityLabel = Literal["DIAGNOSIS", "MEDICATION", "DOSAGE", "PROCEDURE"]


class Span(BaseModel):
    start: int
    end: int
    text: str


class Entity(Span):
    label: EntityLabel
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "hybrid"


class PHISpan(Span):
    category: str
    score: float = Field(default=1.0, ge=0.0, le=1.0)


class ProcessRequest(BaseModel):
    text: str
    redact: bool = True


class BatchProcessRequest(BaseModel):
    documents: list[str]
    redact: bool = True


class ProcessResponse(BaseModel):
    normalized_text: str
    redacted_text: Optional[str] = None
    entities: list[Entity] = Field(default_factory=list)
    phi: list[PHISpan] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    entity_summary: Optional[dict] = None


class BatchProcessResponse(BaseModel):
    results: list[ProcessResponse]
    total_entities: int = 0
    total_phi_spans: int = 0
