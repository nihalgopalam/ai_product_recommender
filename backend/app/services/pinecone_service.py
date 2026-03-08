import asyncio
import logging

from pinecone import Pinecone

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton — None until init_client() is called in lifespan.
# Tests monkeypatch this directly.
index = None


class PineconeUnavailableError(RuntimeError):
    """Raised when the Pinecone index is not initialized."""


def init_client() -> None:
    global index
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    logger.info(
        "pinecone: client initialized index=%s dimension=%d",
        settings.pinecone_index_name,
        settings.pinecone_dimension,
    )


def _reset_for_testing() -> None:
    """Reset module state between tests."""
    global index
    index = None


async def query(
    embedding: list[float],
    top_k: int = 5,
    filter: dict | None = None,
) -> list:
    """
    Query Pinecone index by vector similarity.
    Returns a list of match objects with .id, .score, and .metadata.
    """
    result = await asyncio.to_thread(
        index.query,
        vector=embedding,
        top_k=top_k,
        filter=filter,
        include_metadata=True,
    )
    logger.debug("pinecone: query top_k=%d matches=%d", top_k, len(result.matches))
    return result.matches


async def upsert(vectors: list[dict]) -> None:
    """
    Upsert vectors into Pinecone.
    Each dict must have keys: id (str), values (list[float]), metadata (dict).
    """
    await asyncio.to_thread(index.upsert, vectors=vectors)
    logger.debug("pinecone: upserted %d vectors", len(vectors))
