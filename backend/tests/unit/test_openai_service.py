import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIError

from app.services.openai_service import (
    OpenAIUnavailableError,
    chat_completion,
    embed,
    get_call_count,
)

# Minimal httpx.Request used to construct openai.APIError in tests
_MOCK_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _api_error(msg: str = "test error") -> APIError:
    return APIError(msg, _MOCK_REQUEST, body=None)


# --- chat_completion ---

@pytest.mark.asyncio
async def test_chat_completion_returns_content(mock_openai):
    result = await chat_completion([{"role": "user", "content": "hello"}])
    assert result == "mock response"
    mock_openai.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completion_increments_call_count(mock_openai):
    assert get_call_count() == 0
    await chat_completion([{"role": "user", "content": "hello"}])
    await chat_completion([{"role": "user", "content": "hello"}])
    assert get_call_count() == 2


@pytest.mark.asyncio
async def test_chat_completion_uses_default_model(mock_openai):
    await chat_completion([{"role": "user", "content": "test"}])
    call_kwargs = mock_openai.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_chat_completion_uses_explicit_model(mock_openai):
    await chat_completion([{"role": "user", "content": "test"}], model="gpt-4o-mini")
    call_kwargs = mock_openai.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_chat_completion_retries_once(mock_openai, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    mock_openai.chat.completions.create.side_effect = [
        _api_error(),
        MagicMock(choices=[MagicMock(message=MagicMock(content="retry success"))]),
    ]
    result = await chat_completion([{"role": "user", "content": "test"}])
    assert result == "retry success"
    assert mock_openai.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_chat_completion_raises_after_two_failures(mock_openai, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    mock_openai.chat.completions.create.side_effect = _api_error()
    with pytest.raises(OpenAIUnavailableError):
        await chat_completion([{"role": "user", "content": "test"}])
    assert mock_openai.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_chat_completion_does_not_increment_on_failure(mock_openai, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    mock_openai.chat.completions.create.side_effect = _api_error()
    with pytest.raises(OpenAIUnavailableError):
        await chat_completion([{"role": "user", "content": "test"}])
    assert get_call_count() == 0


# --- embed ---

@pytest.mark.asyncio
async def test_embed_returns_vector(mock_openai):
    result = await embed("gaming laptop")
    assert result == [0.1] * 1536
    mock_openai.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_embed_cache_hit(mock_openai):
    await embed("same text")
    await embed("same text")
    # Second call should be served from cache, not the API
    assert mock_openai.embeddings.create.call_count == 1


@pytest.mark.asyncio
async def test_embed_cache_miss_different_texts(mock_openai):
    await embed("text one")
    await embed("text two")
    assert mock_openai.embeddings.create.call_count == 2


@pytest.mark.asyncio
async def test_embed_raises_after_two_failures(mock_openai, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    mock_openai.embeddings.create.side_effect = _api_error()
    with pytest.raises(OpenAIUnavailableError):
        await embed("some text")
    assert mock_openai.embeddings.create.call_count == 2


@pytest.mark.asyncio
async def test_embed_does_not_cache_on_failure(mock_openai, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    mock_openai.embeddings.create.side_effect = [
        _api_error(),
        _api_error(),
        MagicMock(data=[MagicMock(embedding=[0.5] * 1536)]),
    ]
    with pytest.raises(OpenAIUnavailableError):
        await embed("some text")
    # After failure, a fresh call should hit the API again (not a bad cached value)
    mock_openai.embeddings.create.side_effect = None
    mock_openai.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.5] * 1536)]
    )
    result = await embed("some text")
    assert result == [0.5] * 1536
