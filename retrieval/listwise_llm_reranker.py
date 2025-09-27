"""LLM Listwise Reranker.

This module provides a listwise reranking component inspired by RankGPT:
https://github.com/sunnweiwei/RankGPT

Primary Public API:
    * llm_rerank(query: str, query_result: list[Block], query_context: Optional[str] = None, *, engine: Optional[str] = None) -> list[Block]

Backward Compatibility:
    * initialize_llm_reranker(engine: str) is retained but no longer required; first call to llm_rerank can pass engine instead.

Behavioral Notes:
    * Uses overlapping sliding windows from the tail of the result list backward.
    * Within each window the LLM outputs 1-indexed passage ids; only those are kept (implicit filtering).
    * Dropped documents accelerate the backward step to cover more earlier results.
    * Parsing is defensive to handle partially malformed JSON outputs.
"""

from typing import Final, Optional, cast

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from retrieval.document_block import Block
from utils.logger import logger


# Publicly mutable only through initialize_llm_reranker; keep internal reference typed.
llm_reranker: Optional["ListwiseLLMReranker"] = None

__all__ = []  # No public exports; use higher-level helpers instead.


class RankedPassages(BaseModel):
    """Structured response specifying passage ranking."""

    ranked_passage_ids: list[int] = Field(
        ..., description="1-indexed passage identifiers ranked from most to least useful"
    )


class ListwiseLLMReranker:
    """Listwise LLM-based reranker using overlapping sliding windows.

    Strategy:
      * Process tail windows backward so higher-ranked later docs can surface while
        preserving earlier context ordering.
      * Within each window, only documents explicitly returned by the LLM are kept
        (implicit filtering of low relevance). Missing indices accelerate backward stepping.
    """

    LLM_RERANKER_SLIDING_WINDOW_SIZE: Final[int] = 20
    LLM_RERANKER_SLIDING_WINDOW_STEP: Final[int] = 10

    def __init__(
        self,
        engine: str,
        *,
        sliding_window_size: Optional[int] = None,
        sliding_window_step: Optional[int] = None,
    ) -> None:
        """Configure the listwise reranker.

        Args:
            engine: LangChain chat model identifier (usually an Azure OpenAI
                deployment name).
            sliding_window_size: Optional override for how many retrieved results
                are considered per LLM call. Falls back to
                ``LLM_RERANKER_SLIDING_WINDOW_SIZE`` when unset or non-positive.
            sliding_window_step: Optional override for how far to step backwards
                after processing a window. Falls back to
                ``LLM_RERANKER_SLIDING_WINDOW_STEP`` when unset or non-positive.

        Raises:
            RuntimeError: If the chat model cannot be created with the supplied
                engine.
        """
        self.engine = engine  # store for idempotent init reuse
        # Initialize LangChain Chat Model. We assume Azure OpenAI provider by default (matches existing usage).
        # If another provider is desired, caller should adapt engine or extend this class.
        try:
            base_model = init_chat_model(engine, model_provider="azure_openai")
            # Wrap with structured output schema
            self.model = base_model.with_structured_output(RankedPassages)
        except Exception as e:  # pragma: no cover - defensive; surface clear error
            raise RuntimeError(f"Failed to initialize chat model for engine '{engine}': {e}") from e
        self.sliding_window_size = (
            sliding_window_size
            if sliding_window_size and sliding_window_size > 0
            else ListwiseLLMReranker.LLM_RERANKER_SLIDING_WINDOW_SIZE
        )
        self.sliding_window_step = (
            sliding_window_step
            if sliding_window_step and sliding_window_step > 0
            else ListwiseLLMReranker.LLM_RERANKER_SLIDING_WINDOW_STEP
        )

    async def rerank(
        self,
        query: str,
        query_result: list[Block],
        query_context: Optional[str] = None,
    ) -> list[Block]:
        """Apply listwise reranking to the incoming retrieval results.

        Processes overlapping windows from the tail of ``query_result`` to allow later
        but highly relevant passages to bubble up while still honoring earlier context.

        Args:
            query: Natural-language statement or question being fact-checked.
            query_result: Container of ranked retrieval hits to refine.
            query_context: Optional extra background information injected into the
                LLM prompt.

        Returns:
            list[Block]: Copy of the input with results reordered (and possibly
                filtered) according to the LLM judgment.
        """
        # Fast path: nothing to rerank or not enough results for windowing
        if not query_result:
            return query_result

        results = list(query_result)  # copy for local mutation
        total = len(results)
        window_size = min(self.sliding_window_size, total)

        # Start from the tail and move backwards with overlapping windows
        end_idx = total
        start_idx = end_idx - window_size

        while end_idx > 0:
            if start_idx < 0:
                start_idx = 0

            window_slice = results[start_idx:end_idx]
            if not window_slice:
                break

            reranked_rel_indices = await self._llm_rerank_window(query, window_slice, query_context)

            if reranked_rel_indices:
                reordered_window = [
                    window_slice[i] for i in reranked_rel_indices if 0 <= i < len(window_slice)
                ]
            else:
                logger.warning(
                    "LLM returned no indices for window %s-%s; retaining original order",
                    start_idx,
                    end_idx,
                )
                reordered_window = window_slice

            effective_count = (
                len(reranked_rel_indices) if reranked_rel_indices else len(window_slice)
            )

            results = results[:start_idx] + reordered_window + results[end_idx:]

            if start_idx == 0:
                break

            window_length = end_idx - start_idx
            step = self._compute_backward_step(window_length, effective_count)
            end_idx -= step
            start_idx -= step

        assert len(results) <= len(query_result)
        return results

    async def _llm_rerank_window(
        self,
        query: str,
        search_results: list[Block],
        query_context: Optional[str] = None,
    ) -> list[int]:
        """Ask the LLM to order passages inside a single window.

        Args:
            query: User claim or question under evaluation.
            search_results: Slice of retrieval blocks to score.
            query_context: Optional additional evidence to include in the prompt.

        Returns:
            list[int]: Zero-based indices (relative to ``single_search_results``)
                retained by the LLM, ordered from most to least useful. The list may
                be shorter than the window size if the model discards low-value
                passages.
        """
        try:
            llm_inputs = self._build_llm_messages(query, search_results, query_context)
            response = await self.model.ainvoke(llm_inputs)
        except Exception as e:  # pragma: no cover - network/LLM failures
            logger.error(f"LLM invocation error during rerank window: {e}")
            return []
        return self._normalize_reranked_indices(response, len(search_results))

    def _build_llm_messages(
        self,
        query: str,
        search_results: list[Block],
        query_context: Optional[str],
    ) -> list[dict[str, str]]:
        """Create the chat messages fed to the structured-output model.

        Args:
            query: User claim or question under evaluation.
            search_results: Slice of retrieval outputs being reranked.
            query_context: Optional supplementary context supplied to the model.

        Returns:
            list[dict[str, str]]: Chat messages ready for the LangChain model.
        """
        passages = Block.block_list_to_string(search_results)
        context_section = (
            f"\n\nAdditional context to consider:\n<context>\n{query_context}\n</context>"
            if query_context
            else ""
        )
        return [
            {
                "role": "system",
                "content": """You are an intelligent assistant tasked with ranking passages based on their relevance to a given query, and their usefulness in refuting a false claim. Your goal is to provide an accurate ranking of the passages in descending order of usefulness.

To complete this task, follow these steps:
1. Carefully read the claim and all the passages.

2. For each passage, analyze its content and determine its relevance and usefulness for refuting the given claim. Consider the following factors:
   a. Relevance: How closely does the passage relate to the topic of the claim?
   b. Specificity: Does the passage provide specific facts, figures, or details that directly contradict the claim?

3. Rank the passages based on their overall usefulness for fact-checking the claim. The most useful passage should be ranked first, and the least useful passage should be ranked last.

4. Present your final ranking in the given format. For example, [1, 2, 4, 3] would indicate that passage [1] is the most useful, followed by [2], then [4], and finally [3] is the least useful.
Only provide the ranking result. Do not include any explanations.""",
            },
            {
                "role": "user",
                "content": """False claim to refute:
<false_claim>
{query}
</false_claim>{context}

There are {num_passages} passages:
<passages>
{passages}
</passages>

Provide ranked_passage_ids (1-indexed) from most to least useful. Do not include explanations.""".format(
                    query=query,
                    context=context_section,
                    num_passages=len(search_results),
                    passages=passages,
                ),
            },
        ]

    def _normalize_reranked_indices(
        self,
        response: object,
        window_size: int,
    ) -> list[int]:
        """Convert the structured response into de-duplicated zero-based indices.

        Args:
            response: Structured model response containing ranked passage IDs.
            window_size: Number of passages provided to the LLM in the window.

        Returns:
            list[int]: Zero-based ordered indices retained from the window.
        """
        # Convert 1-indexed to 0-indexed while preserving order and uniqueness
        reranked_indices: list[int] = []
        seen: set[int] = set()
        structured = cast(RankedPassages, response)
        for val in structured.ranked_passage_ids:
            zero = val - 1
            if zero >= 0 and zero < window_size and zero not in seen:
                seen.add(zero)
                reranked_indices.append(zero)

        # Safety: never exceed available results
        assert len(reranked_indices) <= window_size
        return reranked_indices

    def _compute_backward_step(self, window_length: int, kept_count: int) -> int:
        """Determine how far to move the sliding window towards the head.

        Args:
            window_length: Number of passages inspected in the current window.
            kept_count: Passages retained by the LLM for the next ordering phase.

        Returns:
            int: Number of items to step backward by, guaranteeing progress even if
                the LLM keeps everything.
        """
        if window_length <= 0:
            return self.sliding_window_step

        dropped = max(window_length - kept_count, 0)
        step = max(self.sliding_window_step, dropped)
        return step or self.sliding_window_step


async def llm_rerank(
    query: str,
    query_result: list[Block],
    query_context: Optional[str] = None,
    engine: Optional[str] = None,
) -> list[Block]:
    """Entry point for listwise reranking of retrieval results.

    The first invocation constructs a shared :class:`ListwiseLLMReranker` using the
    supplied ``engine``; subsequent calls reuse the cached instance unless a
    different engine is provided (callers must handle engine switches manually by
    passing the new value).

    Args:
        query: Natural-language claim or question the passages should address.
        query_result: Initial retrieval output whose ``results`` will be reordered.
        query_context: Optional extra background to incorporate into the LLM prompt.
        engine: Identifier for the underlying chat model. Mandatory on the first call
            when the module-level reranker has not yet been initialized.

    Returns:
        list[Block]: Copy of ``query_result`` with passages reranked (and potentially
            filtered) according to the LLM response.

    Raises:
        ValueError: If called before initialization and ``engine`` is omitted.
    """
    global llm_reranker
    if llm_reranker is None:
        if not engine:
            raise ValueError("engine must be provided on llm_rerank call")
        llm_reranker = ListwiseLLMReranker(engine)

    return await llm_reranker.rerank(query, query_result, query_context)
