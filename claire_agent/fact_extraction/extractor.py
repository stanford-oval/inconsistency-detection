"""LLM-backed claim extraction implementation."""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model

from utils.logger import logger

from .prompts import EXTRACTION_PROMPT_TEMPLATE, FACT_FILTER_PROMPT_TEMPLATE
from .schemas import FactExtractionResponse, FactFilterResponse


class ClaimExtractor:
    """Encapsulates the LLM-backed pipeline for deriving atomic claims from passages."""

    def __init__(
        self,
        *,
        engine: str,
        model_provider: str,
        reasoning_effort: str | None = None,
    ) -> None:
        if not isinstance(engine, str) or not engine.strip():
            raise ValueError("engine must be provided.")

        if not isinstance(model_provider, str) or not model_provider.strip():
            raise ValueError("model_provider must be provided.")

        self._engine = engine.strip()
        self._model_provider = model_provider.strip()
        self._reasoning_effort = reasoning_effort

    async def extract(self, passage: str) -> list[str]:
        """Extract atomic claims for the provided passage."""
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("passage must be a non-empty string")

        model = self._build_model()
        formatted_prompt = EXTRACTION_PROMPT_TEMPLATE.format(passage=passage).strip()
        messages = [{"role": "user", "content": formatted_prompt}]

        structured_model = model.with_structured_output(FactExtractionResponse)

        try:
            structured_response: FactExtractionResponse = await structured_model.ainvoke(messages)
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception(
                "Structured fact extraction failed; returning empty result",
            )
            return []

        deduped_facts: list[str] = []
        seen: set[str] = set()

        for entry in structured_response.facts_with_spans:
            normalized_fact = entry.fact.strip()
            if normalized_fact and normalized_fact not in seen:
                deduped_facts.append(normalized_fact)
                seen.add(normalized_fact)

        if not deduped_facts:
            logger.warning("No atomic facts extracted from passage")

        filtered_facts = await self._filter_facts(model=model, facts=deduped_facts)

        return filtered_facts

    def _build_model(self) -> Any:
        init_kwargs: dict[str, Any] = {
            "model": self._engine,
            "model_provider": self._model_provider,
        }
        if self._reasoning_effort is not None:
            init_kwargs["reasoning"] = {"effort": self._reasoning_effort}

        return init_chat_model(**init_kwargs)

    async def _filter_facts(self, *, model: Any, facts: list[str]) -> list[str]:
        if not facts:
            return []

        structured_filter_model = model.with_structured_output(FactFilterResponse)

        filter_messages: list[list[dict[str, str]]] = []
        for fact in facts:
            formatted_prompt = FACT_FILTER_PROMPT_TEMPLATE.format(statement=fact.strip()).strip()
            filter_messages.append([{"role": "user", "content": formatted_prompt}])

        try:
            responses: list[FactFilterResponse] = await structured_filter_model.abatch(
                filter_messages
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception(
                "Fact filtering stage failed; returning all extracted facts",
                exc_info=exc,
            )
            return facts

        filtered: list[str] = []
        for fact, response in zip(facts, responses, strict=False):
            if response.determination == "worthy":
                filtered.append(fact)
            else:
                logger.debug(f"Filtered out fact: {fact}. Output: {response}")

        return filtered


__all__ = ["ClaimExtractor"]
