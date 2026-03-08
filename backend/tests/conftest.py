from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.services.openai_service.init_client", lambda: None)
    monkeypatch.setattr("app.services.pinecone_service.init_client", lambda: None)
    monkeypatch.setattr("app.db.models.init_db", lambda: None)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_openai(monkeypatch):
    mock = AsyncMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="mock response"))]
    )
    mock.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    monkeypatch.setattr("app.services.openai_service.client", mock)
    return mock


@pytest.fixture
def mock_pinecone(monkeypatch):
    mock = MagicMock()
    mock.query.return_value = MagicMock(matches=[])
    monkeypatch.setattr("app.services.pinecone_service.index", mock)
    return mock


@pytest.fixture(autouse=True)
def reset_openai_state():
    """Reset openai_service module state between every test."""
    from app.services import openai_service
    openai_service._reset_for_testing()
    yield
    openai_service._reset_for_testing()


@pytest.fixture(autouse=True)
def reset_pinecone_state():
    """Reset pinecone_service module state between every test."""
    from app.services import pinecone_service
    pinecone_service._reset_for_testing()
    yield
    pinecone_service._reset_for_testing()
