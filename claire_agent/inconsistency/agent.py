"""High-level wrapper around the Claire inconsistency detection agent."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from utils.async_parallel import run_async_in_parallel
from utils.logger import logger

from ..fact_extraction import ClaimExtractor
from .models import InconsistencyAgentState, InconsistencyReport
from .prompts import AGENT_SYSTEM_PROMPT
from .reporting import generate_report
from .tools import InconsistencyToolset


class InconsistencyAgent:
    """High-level wrapper around the inconsistency detection agent."""

    def __init__(
        self,
        engine: str,
        model_provider: str,
        num_results_per_query: int,
        reasoning_effort: str | None = None,
    ) -> None:
        if not isinstance(engine, str) or not engine:
            raise ValueError("engine must be a non-empty string")
        if not isinstance(model_provider, str) or not model_provider:
            raise ValueError("model_provider must be a non-empty string")
        if not isinstance(num_results_per_query, int) or num_results_per_query <= 0:
            raise ValueError("num_results_per_query must be a positive integer")

        init_kwargs: dict[str, Any] = {
            "model": engine,
            "model_provider": model_provider,
        }
        if reasoning_effort is not None:
            init_kwargs["reasoning"] = {"effort": reasoning_effort}

        self._model = init_chat_model(**init_kwargs)
        toolset = InconsistencyToolset(
            model=self._model,
            num_results_per_query=num_results_per_query,
        )
        self._tools = toolset.build_tools()

        self._react_agent = create_agent(
            model=self._model,
            state_schema=InconsistencyAgentState,
            tools=self._tools,
            prompt=AGENT_SYSTEM_PROMPT,
        )

        self._claim_extractor = ClaimExtractor(
            engine=engine,
            model_provider=model_provider,
            reasoning_effort=reasoning_effort,
        )

    async def analyze_claim(
        self,
        claim_text: str,
        passage: str,
    ) -> InconsistencyReport:
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("passage must be a non-empty string")

        state = self._build_initial_state(claim_text, passage)
        await self._react_agent.ainvoke(state)
        await generate_report(model=self._model, state=state)

        report = state.get("inconsistency_report")
        assert report is not None
        all_search_results = state.get("all_search_results", [])
        return InconsistencyReport(
            claim_text=claim_text,
            claim_passage=passage,
            verdict=report.verdict,
            wording_feedback=report.wording_feedback,
            explanation=report.explanation,
            search_results=all_search_results,
        )

    async def analyze_passage_for_inconsistencies(
        self,
        passage: str,
    ) -> list[InconsistencyReport]:
        """Extract claims from a passage and generate inconsistency reports."""
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("passage must be a non-empty string")

        claims = await self._claim_extractor.extract(passage=passage)

        passage_preview = passage.strip().splitlines()[0][:80] if passage.strip() else ""

        if not claims:
            logger.info(
                f"No claims extracted for passage '{passage_preview}' when generating inconsistency reports.",
            )
            return []
        logger.info(
            f"Extracted {len(claims)} claims from passage '{passage_preview}'",
        )

        per_claim_inconsistency_reports = await run_async_in_parallel(
            self.analyze_claim,
            claims,
            [passage] * len(claims),
            max_concurrency=10,
        )

        logger.info(
            f"Generated inconsistency reports for {len(claims)} claims",
        )

        return per_claim_inconsistency_reports

    def _build_initial_state(
        self,
        claim_text: str,
        passage: str,
    ) -> InconsistencyAgentState:
        initial_message = HumanMessage(
            content=(
                "You are investigating the following claim extracted from a passage.\n\n"
                f"Passage:\n{passage}\n\n"
                f"Claim:\n{claim_text}"
            )
        )
        return {
            "messages": [initial_message],
            "all_search_results": [],
            "inconsistency_report": None,
            "passage": passage,
            "claim": claim_text,
        }


__all__ = ["InconsistencyAgent"]
