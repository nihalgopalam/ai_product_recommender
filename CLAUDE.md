# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Conversational AI shopping assistant. Users describe what they want; the backend runs a LangGraph agent pipeline to extract intent, search Pinecone, collect feature feedback, run web search, and return ranked recommendations via SSE stream.

Full implementation spec: `shopping-assistant-plan-v2.md`. Refer to it for all architectural decisions, data contracts, agent graph definition, error handling, and cost controls.

**Current branch:** `langchain-rework` — Phase 1 in progress. Scaffold + logging + openai_service done. Pinecone service, SQLite/db layer, and agent graph not yet built.

---

## Commands

```bash
# Run backend (from backend/)
cd backend && uvicorn app.main:app --reload

# Run tests (from backend/)
cd backend && pytest
cd backend && pytest tests/unit/
cd backend && pytest -k "spec_extractor"
cd backend && pytest -v --tb=short
```

Backend runs on `http://localhost:8000`. Verify all routes: `GET /api/routes`.

---

## Architecture

### What exists

| Path | Status |
|------|--------|
| `backend/app/main.py` | FastAPI app, CORS, lifespan, `/api/health`, `/api/routes` |
| `backend/app/config.py` | `Settings` via `pydantic-settings`, loads from `.env`, `@lru_cache` singleton |
| `backend/app/models/schemas.py` | All Pydantic models: `Feature`, `ProductSpec`, `Product`, `UserProfile`, `ChatMessage`, `ChatRequest`, `ProfilePatch` |
| `backend/app/routers/chat.py` | `POST /api/chat/{user_id}` — stub (501) |
| `backend/app/routers/users.py` | `GET/PATCH /api/users/{user_id}/profile`, `DELETE /api/users/{user_id}/session` — stubs (501) |
| `backend/app/routers/products.py` | `GET /api/products/search` — stub (501) |
| `backend/app/logging_config.py` | `configure_logging()` — call first in lifespan |
| `backend/app/services/openai_service.py` | `chat_completion()`, `embed()`, retry, cache, `init_client()` |
| `backend/tests/conftest.py` | `client`, `mock_openai`, `reset_openai_state` (autouse) fixtures |
| `backend/tests/unit/test_openai_service.py` | 13 unit tests |

### What must be built (per `shopping-assistant-plan-v2.md`)

| Path | Purpose |
|------|---------|
| `backend/app/agents/orchestrator.py` | LangGraph `StateGraph` wiring all 9 nodes |
| `backend/app/agents/spec_extractor.py` | Extracts `ProductSpec` from user message (gpt-4o-mini) |
| `backend/app/agents/clarifier.py` | Generates clarifying question for missing spec fields |
| `backend/app/agents/feature_extractor.py` | Extracts 4–6 features per product (gpt-4o-mini) |
| `backend/app/agents/preference_gate.py` | Aggregates feature feedback into `preference_vector` |
| `backend/app/agents/web_searcher.py` | DuckDuckGo search via `langchain_community` |
| `backend/app/agents/db_updater.py` | Embeds + upserts web results into Pinecone |
| `backend/app/agents/profile_updater.py` | EMA merge of session prefs into SQLite `UserProfile` |
| `backend/app/agents/synthesizer.py` | Ranks products, generates explanations (gpt-4o) |
| `backend/app/services/pinecone_service.py` | Pinecone query/upsert |
| `backend/app/services/user_profile.py` | SQLite CRUD for `UserProfile` |
| `backend/app/db/` | SQLAlchemy ORM models (`users`, `user_profiles`) |
| `backend/tests/` | pytest unit + integration tests (see §12 of plan) |
| `frontend/` | React + Vite SPA (Phase 9) |

### LangGraph graph flow

```
spec_extractor → [clarify → END] | [proceed → vector_query]
vector_query → feature_extractor → END   (pause for feature feedback)
preference_gate → [more_feedback → END] | [proceed → web_searcher]
web_searcher → db_updater → profile_updater → synthesizer → END
```

Graph pauses at `clarifier` and `feature_extractor` nodes. The chat endpoint resumes by re-injecting user input into `AgentState` and calling `graph.ainvoke()` again.

### Session state

`AgentState` is stored in `app.state.sessions` (plain `dict`, keyed by `user_id`, 2hr TTL). Access from a request handler via `request.app.state.sessions`.

### Key decisions (do not revisit without strong reason)

- **SSE not WebSocket** for `POST /api/chat/{user_id}` — returns `text/event-stream`
- **SQLite not PostgreSQL** — SQLAlchemy with `create_all()`, file at `DATABASE_PATH`
- **DuckDuckGo** for web search — no API key required (`langchain_community`)
- **gpt-4o-mini** for spec + feature extraction; **gpt-4o** for synthesizer only
- **Always upsert** to Pinecone — unconditional, idempotent
- **Pinecone `top_k=5`** — caps downstream LLM calls
- **Embedding cache** — in-memory `dict[str, list[float]]` keyed by content hash
- **EMA alpha=0.3** for merging session preferences into long-term `UserProfile`

### Dependencies

All Phase 1 deps added to `requirements.txt`. Run `pip install -r requirements.txt` from `backend/`.

### Logging

`configure_logging()` is in `backend/app/logging_config.py`, called first in lifespan. Each module uses `logging.getLogger(__name__)`. Silenced loggers: `httpx`, `httpcore`, `openai`, `pinecone`, `langchain_core`, `langgraph`.

### Testing patterns

- Use `@pytest.mark.asyncio` on all async tests (no pytest.ini asyncio_mode)
- `monkeypatch(asyncio, "sleep", AsyncMock())` — suppresses tenacity's 2s retry wait in tests
- `openai.APIError` requires `httpx.Request` in constructor: `APIError("msg", httpx.Request("POST", "https://api.openai.com"), body=None)`
- Service modules expose `_reset_for_testing()` to clear module-level state. `conftest.py` calls it via `autouse=True` fixture.
- Mock service clients by monkeypatching the module-level variable: `monkeypatch.setattr("app.services.openai_service.client", mock)`

### Code conventions / gotchas

- Do NOT use `from __future__ import annotations` in files with Pydantic models — breaks Pydantic v2 runtime annotation inspection
- Use `datetime.now(timezone.utc)` not `datetime.utcnow()` — deprecated since Python 3.12
- Service clients (`AsyncOpenAI`, Pinecone index) are `None` at module level, initialized via `init_client()` called in lifespan — never at import time
