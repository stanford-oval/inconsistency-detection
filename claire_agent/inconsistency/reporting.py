"""Utilities for summarising the agent run into a final inconsistency report."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

from retrieval.document_block import Block
from utils.logger import logger

from .models import InconsistencyAgentState, InconsistencyReportResponse
from .prompts import REPORT_SYSTEM_PROMPT


async def generate_report(*, model: Any, state: InconsistencyAgentState) -> None:
    """Populate the agent state with a structured inconsistency report."""
    messages = list(state.get("messages", []))
    if not messages:
        logger.warning("No messages available for explanation generation.")
        return

    final_message = next(
        (msg for msg in reversed(messages) if isinstance(msg, BaseMessage)),
        None,
    )
    if final_message is None:
        logger.warning("Unable to locate final message for explanation generation.")
        return

    final_message_text = final_message.text()
    all_search_results = list(state.get("all_search_results", []))

    unique_results: dict[str, Block] = {}
    for block in all_search_results:
        if block.combined_text not in unique_results:
            unique_results[block.combined_text] = block
    all_search_results = list(unique_results.values())

    passage = state.get("passage", "")
    claim_text = state.get("claim", "")

    structured_model = model.with_structured_output(InconsistencyReportResponse)

    inconsistency_report_response: InconsistencyReportResponse = await structured_model.ainvoke(
        [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original Passage:\n{passage}\n\n"
                    f"Investigated Claim:\n{claim_text}\n\n"
                    f"Search results:\n{Block.block_list_to_string(all_search_results)}\n\n"
                    f"Final Verdict Summary: {final_message_text}\n"
                ),
            },
        ]
    )

    state["inconsistency_report"] = inconsistency_report_response
    state["all_search_results"] = all_search_results


__all__ = ["generate_report"]
