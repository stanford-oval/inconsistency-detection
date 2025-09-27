"""Retrieval package public interface.

This package provides:

Classes:
    Block: Base retrieval/indexing unit.

Functions:
    search_wikipedia: High-level helper for Wikipedia retrieval with optional client-side reranking (and optional internal LLM rerank step).

Only symbols listed in __all__ are considered stable. Internal modules should not be imported
from outside the package; prefer importing from `retrieval` directly:

    from retrieval import Block, llm_rerank, search_wikipedia

"""

from .document_block import Block
from .wikipedia_search import search_wikipedia


__all__ = [
    "Block",
    "search_wikipedia",
]
