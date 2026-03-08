import hashlib
import logging
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton — None until init_client() is called in lifespan.
# Tests monkeypatch this directly.
client: AsyncOpenAI | None = None

_embed_cache: dict[str, list[float]] = {}
_call_count: int = 0


class OpenAIUnavailableError(RuntimeError):
    """Raised when OpenAI API fails after all retries."""


def init_client() -> None:
    global client
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    logger.info(
        "openai: client initialized model=%s embedding_model=%s",
        settings.openai_chat_model,
        settings.openai_embedding_model,
    )


def get_call_count() -> int:
    return _call_count


def _reset_for_testing() -> None:
    """Reset module state between tests."""
    global _call_count
    _call_count = 0
    _embed_cache.clear()


# --- Private retry wrappers ---
# Separated from the public functions so monkeypatching `client` works at call time.

@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((APIError, APITimeoutError)),
    reraise=True,
)
async def _chat_with_retry(messages: list[dict[str, Any]], model: str):
    return await client.chat.completions.create(model=model, messages=messages)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((APIError, APITimeoutError)),
    reraise=True,
)
async def _embed_with_retry(text: str, model: str):
    return await client.embeddings.create(input=text, model=model)


# --- Public API ---

async def chat_completion(messages: list[dict[str, Any]], model: str | None = None) -> str:
    """
    Call OpenAI chat completion. Retries once with 2s backoff on API errors.
    Raises OpenAIUnavailableError if both attempts fail.
    Increments the module-level call counter on success.
    """
    global _call_count
    settings = get_settings()
    model = model or settings.openai_chat_model
    logger.debug("chat_completion: model=%s messages=%d", model, len(messages))
    try:
        response = await _chat_with_retry(messages, model)
    except (APIError, APITimeoutError) as exc:
        logger.error("chat_completion: failed after retries model=%s", model, exc_info=True)
        raise OpenAIUnavailableError("OpenAI chat service unavailable") from exc
    _call_count += 1
    return response.choices[0].message.content


async def embed(text: str) -> list[float]:
    """
    Embed text using text-embedding-3-small. Results are cached by MD5 of input.
    Retries once with 2s backoff on API errors.
    Raises OpenAIUnavailableError if both attempts fail.
    """
    settings = get_settings()
    key = hashlib.md5(text.encode()).hexdigest()
    if key in _embed_cache:
        logger.debug("embed: cache hit key=%s", key)
        return _embed_cache[key]

    logger.debug("embed: cache miss text_len=%d", len(text))
    try:
        response = await _embed_with_retry(text, settings.openai_embedding_model)
    except (APIError, APITimeoutError) as exc:
        logger.error("embed: failed after retries text_len=%d", len(text), exc_info=True)
        raise OpenAIUnavailableError("OpenAI embedding service unavailable") from exc

    vector = response.data[0].embedding
    _embed_cache[key] = vector
    return vector
