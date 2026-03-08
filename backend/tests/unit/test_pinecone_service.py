from unittest.mock import MagicMock

import pytest

from app.services.pinecone_service import query, upsert


# --- query ---

@pytest.mark.asyncio
async def test_query_returns_matches(mock_pinecone):
    mock_match = MagicMock()
    mock_match.id = "prod-1"
    mock_pinecone.query.return_value = MagicMock(matches=[mock_match])

    results = await query([0.1] * 1536)

    assert len(results) == 1
    assert results[0].id == "prod-1"
    mock_pinecone.query.assert_called_once_with(
        vector=[0.1] * 1536, top_k=5, filter=None, include_metadata=True
    )


@pytest.mark.asyncio
async def test_query_passes_top_k_and_filter(mock_pinecone):
    mock_pinecone.query.return_value = MagicMock(matches=[])

    await query([0.1] * 1536, top_k=3, filter={"category": {"$eq": "laptop"}})

    call_kwargs = mock_pinecone.query.call_args.kwargs
    assert call_kwargs["top_k"] == 3
    assert call_kwargs["filter"] == {"category": {"$eq": "laptop"}}


@pytest.mark.asyncio
async def test_query_empty_results(mock_pinecone):
    mock_pinecone.query.return_value = MagicMock(matches=[])

    results = await query([0.0] * 1536)

    assert results == []


# --- upsert ---

@pytest.mark.asyncio
async def test_upsert_calls_index(mock_pinecone):
    vectors = [{"id": "prod-1", "values": [0.1] * 1536, "metadata": {"name": "Test"}}]

    await upsert(vectors)

    mock_pinecone.upsert.assert_called_once_with(vectors=vectors)


@pytest.mark.asyncio
async def test_upsert_multiple_vectors(mock_pinecone):
    vectors = [
        {"id": f"prod-{i}", "values": [float(i)] * 1536, "metadata": {}}
        for i in range(3)
    ]

    await upsert(vectors)

    mock_pinecone.upsert.assert_called_once()
    assert len(mock_pinecone.upsert.call_args.kwargs["vectors"]) == 3
