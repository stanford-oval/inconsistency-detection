"""Data models and agent state definitions for inconsistency detection."""

from __future__ import annotations

from typing import Annotated, Literal, Sequence

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from retrieval.document_block import Block


class InconsistencyReportResponse(BaseModel):
    """Structured response returned by the report generation model."""

    verdict: Literal["consistent", "inconsistent"] = Field(
        description=(
            "Overall determination of the investigated claim. Use 'consistent' if the claim is supported, "
            "'inconsistent' if at least one piece of evidence contradicts it."
        ),
    )
    wording_feedback: str = Field(
        description="Guidance on whether the claims' wording in the original passage should be revised and why.",
    )
    explanation: str = Field(
        description="Detailed narrative explaining the verdict with citations in [n] format.",
    )


class InconsistencyReport(InconsistencyReportResponse):
    """Used for returning the final report along with all search results."""

    claim_text: str = Field(
        description="The claim text that was investigated.",
    )
    claim_passage: str = Field(
        description="The passage from which the claim was extracted.",
    )
    search_results: list[Block] = Field(
        description="List of all search results retrieved during the investigation.",
    )


class InconsistencyAgentState(AgentState):
    """Runtime state for the inconsistency detection agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    passage: str
    claim: str
    all_search_results: list[Block]
    inconsistency_report: InconsistencyReportResponse | None


__all__ = [
    "InconsistencyAgentState",
    "InconsistencyReport",
    "InconsistencyReportResponse",
]
