from typing import Optional

from retrieval.document_block import Block
from retrieval.listwise_llm_reranker import llm_rerank
from retrieval.retriever_client import retrieve_via_api


async def search_wikipedia(
    search_query: str,
    num_results: int,
    do_client_side_reranking: bool,
    num_results_before_reranking: Optional[int] = None,
    article_to_exclude: Optional[str] = None,
    reranking_engine: Optional[str] = None,
) -> list[Block]:
    """Never does a server-side reranking. Only does client-side reranking if do_client_side_reranking is True."""
    if num_results_before_reranking is None:
        num_results_before_reranking = num_results

    assert num_results_before_reranking >= num_results
    if not do_client_side_reranking:
        assert num_results_before_reranking == num_results, (
            "If not doing client-side reranking, num_results_before_reranking must equal num_results_per_query"
        )

    search_results: list[Block] = await retrieve_via_api(
        query=search_query,
        retriever_endpoint="https://search.genie.stanford.edu/wikipedia_20250320",
        num_results=num_results_before_reranking,
    )

    if do_client_side_reranking:
        search_results = await llm_rerank(
            search_query,
            search_results,
            engine=reranking_engine,
        )

    if article_to_exclude:
        search_results = [
            result for result in search_results if result.document_title != article_to_exclude
        ]

    assert len(search_results) <= num_results_before_reranking

    if len(search_results) > num_results:
        search_results = search_results[:num_results]

    return search_results
