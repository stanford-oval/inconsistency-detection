"""Tooling utilities used by the inconsistency agent."""

from __future__ import annotations

from typing import Annotated, Any

from langchain.agents.tool_node import InjectedState
from langchain_core.tools import tool

from retrieval import search_wikipedia as sw
from retrieval.document_block import Block

from .prompts import (
    CLARIFY_SYSTEM_PROMPT,
    CLARIFY_TOOL_DESCRIPTION,
    EXPLAIN_SYSTEM_PROMPT,
    EXPLAIN_TOOL_DESCRIPTION,
    SEARCH_TOOL_DESCRIPTION,
)


class InconsistencyToolset:
    """Builds the tool set exposed to the underlying ReAct agent."""

    def __init__(self, *, model: Any, num_results_per_query: int) -> None:
        self._model = model
        self._num_results_per_query = num_results_per_query

    def build_tools(self) -> list[Any]:
        explain_tool = tool(
            self._explain,
            description=EXPLAIN_TOOL_DESCRIPTION,
        )
        explain_tool.name = "explain"

        clarify_tool = tool(
            self._clarify_entity,
            description=CLARIFY_TOOL_DESCRIPTION,
        )
        clarify_tool.name = "clarify_entity"

        search_tool = tool(
            self._search_wikipedia,
            description=SEARCH_TOOL_DESCRIPTION,
        )
        search_tool.name = "search_wikipedia"

        return [explain_tool, clarify_tool, search_tool]

    async def _explain(self, topic: str) -> str:
        llm_output = await self._model.ainvoke(
            [
                {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": f"Topic: {topic}"},
            ]
        )
        return llm_output.text()

    async def _clarify_entity(self, entity_name_and_description: str) -> str:
        search_results = await sw(
            entity_name_and_description,
            num_results=15,
            do_client_side_reranking=False,
        )
        search_results_string = Block.block_list_to_string(search_results)

        llm_output = await self._model.ainvoke(
            [
                {"role": "system", "content": CLARIFY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""Entity: {entity_name_and_description}\n\n{search_results_string}""",
                },
            ]
        )
        return llm_output.text()

    async def _search_wikipedia(
        self,
        state: Annotated[dict[str, Any], InjectedState],
        query: str,
    ) -> str:
        search_results = await sw(
            search_query=query,
            num_results=self._num_results_per_query,
            do_client_side_reranking=True,
            num_results_before_reranking=20,
            reranking_engine="gpt-4o",
        )
        result_string = Block.block_list_to_string(
            search_results,
            start_index=len(state["all_search_results"]) + 1,
        )
        state["all_search_results"].extend(search_results)

        return result_string


__all__ = ["InconsistencyToolset"]
