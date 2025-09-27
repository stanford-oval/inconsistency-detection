"""Pydantic schemas used across the fact extraction pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TopicFacts(BaseModel):
    """Facts grouped by a shared topic or theme."""

    topic: str = Field(..., description="Topic or theme identified in the passage.")
    facts: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit statements associated with the topic, expressed as concise bulleted facts."
        ),
    )


class FactExtractionProcess(BaseModel):
    """Schema capturing the intermediate reasoning prior to providing final facts."""

    key_topics: list[str] = Field(
        default_factory=list,
        description="Primary topics or themes derived from the title and passage.",
    )
    explicit_facts_by_topic: list[TopicFacts] = Field(
        default_factory=list,
        description="Explicit facts grouped by topic for traceability.",
    )
    direct_inferences_considered: list[str] = Field(
        default_factory=list,
        description=(
            "Potential inferences evaluated for inclusion, limited to those "
            "directly supported by explicit statements."
        ),
    )
    atomicity_and_self_containment_review: str = Field(
        ...,
        description=(
            "Summary of how each fact was confirmed to be atomic, self-contained, "
            "and independently verifiable."
        ),
    )
    redundancy_check: str = Field(
        ...,
        description="Explanation of how redundant or overlapping facts were removed.",
    )


class FactWithSpan(BaseModel):
    """Atomic fact with supporting text span."""

    fact: str = Field(..., description="Atomic fact, ready for fact-checking.")
    text_span: str = Field(
        ...,
        description="Exact sentence or phrase from the passage that supports the fact.",
    )


class FactExtractionResponse(BaseModel):
    """Structured payload returned by the fact extraction model."""

    fact_extraction_process: FactExtractionProcess = Field(
        ...,
        description="Detailed breakdown of the process used to derive the final facts.",
    )
    facts_with_spans: list[FactWithSpan] = Field(
        default_factory=list,
        description="Final list of atomic facts paired with their supporting text spans.",
    )


class FactFilterResponse(BaseModel):
    """Decision on whether a fact requires verification."""

    reasoning: str = Field(
        ...,
        description=(
            "Step-by-step justification referencing objectivity, non-triviality, "
            "relevance, and contextual sufficiency."
        ),
    )
    determination: Literal["worthy", "not_worthy"] = Field(
        ...,
        description=(
            "Final decision on whether the fact is worthy of verification. Use "
            '"worthy" for statements that should be verified and "not_worthy" '
            "otherwise."
        ),
    )


__all__ = [
    "FactExtractionProcess",
    "FactExtractionResponse",
    "FactFilterResponse",
    "FactWithSpan",
    "TopicFacts",
]
