"""Utilities for working with LangChain LLM usage tracking."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Final, Optional, TypedDict

from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tracers.context import register_configure_hook

from utils.logger import logger


class ModelPricing(TypedDict):
	"""Pricing information for a single OpenAI model."""

	input: float
	cached_input: float
	output: float


_TOKENS_PER_MILLION: Final = 1_000_000

_OPENAI_PRICING_PER_M_TOKEN: Final[dict[str, ModelPricing]] = {
	"gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
	"gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
	"gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.4},
	"gpt-4.1": {"input": 2.0, "cached_input": 0.5, "output": 8.0},
	"gpt-4.1-mini": {"input": 0.4, "cached_input": 0.1, "output": 1.6},
	"gpt-4.1-nano": {"input": 0.1, "cached_input": 0.025, "output": 0.4},
	"gpt-4o": {"input": 2.5, "cached_input": 1.25, "output": 10.0},
	"gpt-4o-2024-11-20": {"input": 2.5, "cached_input": 1.25, "output": 10.0},
	"gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.6},
}


_GLOBAL_USAGE_HANDLER_VAR: ContextVar[
	Optional[UsageMetadataCallbackHandler]
] = ContextVar("global_usage_metadata_handler", default=None)

# Ensure LangChain picks up the handler from the context once configured.
register_configure_hook(
	_GLOBAL_USAGE_HANDLER_VAR,
	inheritable=True,
	handle_class=UsageMetadataCallbackHandler,
)


def enable_global_usage_tracking() -> UsageMetadataCallbackHandler:
	"""Ensure a usage metadata handler is installed and return it."""
	handler = _GLOBAL_USAGE_HANDLER_VAR.get()
	if handler is None:
		handler = UsageMetadataCallbackHandler()
		_GLOBAL_USAGE_HANDLER_VAR.set(handler)
	return handler


def _extract_cached_input_tokens(metadata: UsageMetadata) -> int:
	"""Return the total cached input tokens recorded in usage metadata."""
	input_details = metadata.get("input_token_details")
	if not isinstance(input_details, dict):
		return 0

	total_cached = 0
	for key, value in input_details.items():
		if key.startswith("cache_read") and isinstance(value, (int, float)):
			total_cached += int(value)
	return total_cached


def _calculate_cost(model_name: str, metadata: UsageMetadata) -> Optional[float]:
	"""Calculate the dollar cost for a single model's usage metadata."""
	pricing = _OPENAI_PRICING_PER_M_TOKEN.get(model_name)
	if pricing is None:
		return None

	input_tokens = metadata.get("input_tokens", 0)
	cached_input_tokens = _extract_cached_input_tokens(metadata)
	output_tokens = metadata.get("output_tokens", 0)
	paid_input_tokens = max(input_tokens - cached_input_tokens, 0)
	cost = (
		paid_input_tokens * pricing["input"]
		+ cached_input_tokens * pricing["cached_input"]
		+ output_tokens * pricing["output"]
	) / _TOKENS_PER_MILLION
	return cost


def _aggregate_usage(usage_metadata: dict[str, UsageMetadata]) -> UsageMetadata:
	total: UsageMetadata = UsageMetadata(
		input_tokens=0,
		output_tokens=0,
		total_tokens=0,
	)
	for metadata in usage_metadata.values():
		total["input_tokens"] += metadata.get("input_tokens", 0)
		total["output_tokens"] += metadata.get("output_tokens", 0)
		total["total_tokens"] += metadata.get("total_tokens", 0)
	return total


def log_global_token_usage(prefix: str = "LLM token usage summary") -> None:
	"""Log aggregated token usage across all tracked LLM calls."""
	handler = _GLOBAL_USAGE_HANDLER_VAR.get()
	if handler is None:
		logger.info(f"{prefix}: handler not configured")
		return

	usage_metadata = handler.usage_metadata
	if not usage_metadata:
		logger.info(f"{prefix}: no LLM calls recorded")
		return

	aggregate = _aggregate_usage(usage_metadata)
	total_cached_input_tokens = 0
	total_cost = 0.0
	model_logs: list[str] = []
	for model_name, metadata in usage_metadata.items():
		cached_input_tokens = _extract_cached_input_tokens(metadata)
		total_cached_input_tokens += cached_input_tokens
		cost = _calculate_cost(model_name, metadata)
		if cost is not None:
			total_cost += cost
		cost_display = f"${cost:.6f}" if cost is not None else "unknown"
		cost_display = f"${cost:,.6f}" if cost is not None else "unknown"

		input_tokens = metadata.get("input_tokens", 0)
		output_tokens = metadata.get("output_tokens", 0)
		total_tokens = metadata.get("total_tokens", 0)

		model_logs.append(
			(
				f"{prefix}\n"
				f"  model={model_name}\n"
				f"  input={input_tokens:,}\n"
				f"  cached_input={cached_input_tokens:,}\n"
				f"  output={output_tokens:,}\n"
				f"  total={total_tokens:,}\n"
				f"  cost={cost_display}"
			)
		)
	logger.info(
		(
			f"{prefix} totals\n"
			f"  total_input={aggregate['input_tokens']:,}\n"
			f"  total_cached_input={total_cached_input_tokens:,}\n"
			f"  total_output={aggregate['output_tokens']:,}\n"
			f"  total_tokens={aggregate['total_tokens']:,}\n"
			f"  total_cost=${total_cost:,.6f}"
		)
	)

	for log_line in model_logs:
		logger.info(log_line)



