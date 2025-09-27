import argparse
import asyncio
import json

from claire_agent import InconsistencyAgent
from claire_agent.claim import Claim
from retrieval.document_block import Block
from utils.async_parallel import run_async_in_parallel
from utils.logger import logger
from utils.report_rendering import render_inconsistency_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="gpt-5")
    parser.add_argument(
        "--model_provider",
        type=str,
        default="azure_openai",
        choices=[
            "xai",
            "ollama",
            "bedrock",
            "ibm",
            "anthropic",
            "cohere",
            "fireworks",
            "google_vertexai",
            "groq",
            "openai",
            "azure_ai",
            "together",
            "mistralai",
            "google_anthropic_vertex",
            "google_genai",
            "azure_openai",
            "deepseek",
            "bedrock_converse",
            "huggingface",
            "perplexity",
        ],
        help="Model provider. The options are from LangChain.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=0,
        help="Number of examples to process. 0 means all.",
    )
    parser.add_argument(
        "--input_offset",
        type=int,
        default=0,
        help="Offset to start processing examples from. Useful for debugging.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikicollide_dataset/dev.json",
        help="Dataset path to load from",
    )
    parser.add_argument(
        "--num_results_per_query",
        type=int,
        default=5,
        help="Number of search results to retrieve per query",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning effort level for the LLM. Only used for reasoning models.",
    )
    return parser.parse_args()


async def main(args):
    dataset = json.load(open(args.dataset, "r"))
    if args.input_size > 0:
        dataset = dataset[args.input_offset : args.input_offset + args.input_size]
    else:
        dataset = dataset[args.input_offset :]

    claims: list[Claim] = []
    for d in dataset:
        claims.append(
            Claim(
                claim_id=d["claim"]["claim_id"],
                claim_text=d["claim"]["claim_text"],
                context_block=Block(**d["claim"]["claim_context_block"]),
            )
        )
    logger.info(f"Loaded {len(claims)} example(s) from {args.dataset}")

    if not args.engine:
        raise ValueError("engine must be a non-empty string")
    if not args.model_provider:
        raise ValueError("model_provider must be a non-empty string")
    if args.num_results_per_query <= 0:
        raise ValueError("num_results_per_query must be a positive integer")

    agent = InconsistencyAgent(
        args.engine,
        args.model_provider,
        reasoning_effort=args.reasoning_effort,
        num_results_per_query=args.num_results_per_query,
    )

    # run the entire dataset
    inconsistency_reports = await run_async_in_parallel(
        agent.analyze_claim,
        [c.claim_text for c in claims],
        [c.context_block.combined_text for c in claims],
        max_concurrency=5,
        desc="Analyzing claims",
    )
    for report in inconsistency_reports:
        render_inconsistency_report(report)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
