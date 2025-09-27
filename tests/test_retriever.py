import pytest

from retrieval import search_wikipedia


@pytest.mark.asyncio
async def test_search_wikipedia_live_call():
    """Live integration test of the public search_wikipedia API."""
    results = await search_wikipedia(
        search_query="Albert Einstein birthplace",
        num_results=3,
        do_client_side_reranking=False,  # avoid LLM dependency here
    )

    assert results, "Expected at least one result from live call"


@pytest.mark.asyncio
async def test_search_wikipedia_with_rerank():
    """Ensure reranking path works through public search_wikipedia."""
    results = await search_wikipedia(
        search_query="Albert Einstein theory of relativity",
        num_results=10,
        do_client_side_reranking=True,
        reranking_engine="gpt-4o",
    )

    assert results, "Expected results after reranking"
    # Basic shape assertions
    assert all(r.document_title for r in results)
