import asyncio
import weakref

import httpx

from retrieval.document_block import Block
from utils.disk_cache import diskcache_cache
import utils.logger as logger


logger = logger.logger

_CLIENT_TIMEOUT = httpx.Timeout(30.0)
_CLIENT_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=20)
_loop_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
    weakref.WeakKeyDictionary()
)
_loop_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)


async def _get_http_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    lock = _loop_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _loop_locks[loop] = lock

    async with lock:
        client = _loop_clients.get(loop)
        if client is None or getattr(client, "is_closed", False):
            _loop_clients[loop] = httpx.AsyncClient(
                timeout=_CLIENT_TIMEOUT,
                limits=_CLIENT_LIMITS,
            )
            client = _loop_clients[loop]
        return client


async def close_retriever_http_client() -> None:
    """Gracefully close the shared HTTP client for the current loop."""
    loop = asyncio.get_running_loop()
    lock = _loop_locks.get(loop)
    if lock is None:
        return

    async with lock:
        client = _loop_clients.pop(loop, None)
        if client is not None and not getattr(client, "is_closed", False):
            await client.aclose()


@diskcache_cache
async def retrieve_via_api(
    query: str,
    retriever_endpoint: str,
    num_results: int,
) -> list[Block]:
    """Internal helper: retrieve search results from a retriever API for a single query.

    Args:
        query (str): The query string to be sent to the retriever.
        retriever_endpoint (str): The endpoint URL of the retriever API.
        num_results (int): Number of blocks to consider before reranking.

    Returns:
    list[Block]: List of retrieved blocks for the query.

    Raises:
        Exception: If the rate limit is reached or if there is an error with the retriever API request.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a str")

    try:
        client = await _get_http_client()
        response = await client.post(
            retriever_endpoint,
            json={
                "query": [query],  # API previously expected a list; keep list wrapper
                "rerank": False,  # We handle reranking ourselves if needed
                "num_blocks_to_rerank": num_results,
                "num_blocks": num_results,
                "search_filters": [
                    {
                        "field_name": "language",
                        "filter_type": "eq",
                        "field_value": "en",
                    }
                ],
            },
        )
    except httpx.RequestError as exc:
        logger.error(
            "Error encountered when sending this request to retriever endpoint={} error={}",
            retriever_endpoint,
            exc,
        )
        raise

    if response.status_code == 429:
        raise Exception(
            "You have reached your rate limit for the retrieval server. Please wait and try later, or host your own retriever."
        )
    if response.status_code != 200:
        logger.error(f"Error encountered when sending this request to retriever: {response.text}")

    results = response.json()
    assert isinstance(results, list) and len(results) == 1, (
        f"Expected exactly one result for the single query. Got: {results}"
    )

    search_results = [
        Block(
            document_title=r["document_title"],
            section_title=r["section_title"],
            content=r["content"],
            last_edit_date=r["last_edit_date"],
            url=r["url"],
        )
        for r in results[0]["results"]
    ]  # convert to pydantic object
    return search_results
