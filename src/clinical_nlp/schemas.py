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
    redacted_text: Optional[str]
    entities: list[Entity]
    phi: list[PHISpan]
    sections: list[str]


class BatchProcessResponse(BaseModel):
    results: list[ProcessResponse]
