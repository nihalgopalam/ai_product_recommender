# Shopping Assistant — Implementation Plan
**Stack:** Python (FastAPI) · React (Vite) · Pinecone · OpenAI GPT-4o · v1.0

---

This document provides a complete, agent-ready implementation plan for the Shopping Assistant — a conversational product recommendation system with semantic search, real-time web research, and a persisted user preference model. All phases, file structures, data contracts, and API surfaces are defined with enough specificity for a coding agent to implement without further clarification.

---

## 1. Project Structure

The project is split into two top-level workspaces: a Python FastAPI backend and a React/Vite frontend.

| Path | Purpose |
|------|---------|
| `backend/` | FastAPI application root |
| `backend/app/main.py` | FastAPI app + CORS + router registration |
| `backend/app/routers/chat.py` | SSE + REST chat endpoints |
| `backend/app/agents/orchestrator.py` | Main LangGraph agent graph |
| `backend/app/agents/spec_extractor.py` | NLU node — extracts specs from user message |
| `backend/app/agents/clarifier.py` | Generates clarifying questions |
| `backend/app/agents/feature_extractor.py` | LLM feature extraction per product |
| `backend/app/agents/web_searcher.py` | Web search tool node |
| `backend/app/agents/db_updater.py` | Upsert new products into Pinecone |
| `backend/app/agents/synthesizer.py` | Final ranking + recommendation generation |
| `backend/app/services/pinecone_service.py` | Pinecone client wrapper (shared catalog) |
| `backend/app/services/openai_service.py` | OpenAI client wrapper (chat + embeddings) |
| `backend/app/services/user_profile.py` | User profile CRUD (SQLite) |
| `backend/app/models/` | Pydantic schemas for all data shapes |
| `backend/app/db/` | SQLAlchemy models (SQLite) |
| `backend/.env.example` | Required environment variables |
| `frontend/` | React + Vite application root |
| `frontend/src/components/Chat/` | ChatWindow, MessageBubble, InputBar |
| `frontend/src/components/Feedback/` | FeatureSelector — like/dislike UI |
| `frontend/src/components/Products/` | ProductCard, ProductGrid |
| `frontend/src/hooks/useChat.ts` | SSE + REST state management |
| `frontend/src/store/sessionStore.ts` | Zustand store for session state |
| `frontend/src/types/` | TypeScript interfaces matching backend models |

---

## 2. Environment Variables

All secrets are loaded from `backend/.env`. Never commit this file. Provide `backend/.env.example` with these keys:

```env
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o
PINECONE_API_KEY=
PINECONE_INDEX_NAME=shopping-assistant-products
PINECONE_DIMENSION=1536
DATABASE_PATH=./data/shopping_assistant.db
CORS_ORIGINS=http://localhost:5173
```

---

## 3. Data Models

These Pydantic models define the contracts between all layers. TypeScript equivalents must be generated or mirrored in `frontend/src/types/`.

### 3.1 ProductSpec

Represents the structured intent extracted from the user's message.

```python
class ProductSpec(BaseModel):
    category: str                    # e.g. 'laptop', 'headphones'
    constraints: dict[str, Any]      # e.g. {'price_max': 1500, 'brand': 'Sony'}
    keywords: list[str]              # free-text search terms
    is_complete: bool                # True when no required fields are missing
    missing_fields: list[str]        # fields that triggered clarifying questions
```

### 3.2 Product

A product record stored in Pinecone (vector) and referenced by ID.

```python
class Product(BaseModel):
    id: str                          # Pinecone vector ID
    name: str
    description: str
    price: float | None
    url: str
    source: str                      # 'vector_db' | 'web_search'
    features: list[Feature]          # LLM-extracted feature list
    embedding: list[float] | None    # omitted when returning to client
```

### 3.3 Feature

A single LLM-extracted product attribute surfaced in the preference selector.

```python
class Feature(BaseModel):
    name: str                        # e.g. 'battery_life'
    display_name: str                # e.g. 'Battery Life'
    value: str                       # e.g. '15 hours'
    sentiment: Literal['liked', 'disliked', 'neutral'] = 'neutral'
```

### 3.4 UserProfile

Persisted user model. Updated at every preference feedback interaction.

```python
class UserProfile(BaseModel):
    user_id: str
    feature_preferences: dict[str, float]  # feature_name → score [-1.0, 1.0]
    disliked_products: list[str]           # product IDs to exclude
    liked_products: list[str]
    session_history: list[str]             # past query summaries
    updated_at: datetime
```

### 3.5 ChatMessage

```python
class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str
    message_type: Literal['text', 'clarify', 'feature_select', 'products', 'recommendations']
    payload: dict[str, Any] | None   # products / features / etc.
```

### 3.6 AgentState (LangGraph)

The shared mutable state passed between all agent nodes.

```python
class AgentState(TypedDict):
    user_id: str
    messages: list[ChatMessage]
    spec: ProductSpec | None
    candidate_products: list[Product]
    feature_feedback: list[Feature]         # user-annotated features
    preference_vector: dict[str, float]     # aggregated this session
    web_results: list[Product]
    user_profile: UserProfile
    next_action: str                        # routing signal
```

---

## 4. Agent Orchestration (LangGraph)

The backend orchestration layer is a LangGraph `StateGraph`. Each node is a pure function that reads from and writes to `AgentState`. Conditional edges handle the two user-facing loops.

### 4.1 Graph Definition (`orchestrator.py`)

```python
graph = StateGraph(AgentState)

graph.add_node('spec_extractor',    spec_extractor_node)
graph.add_node('clarifier',         clarifier_node)
graph.add_node('vector_query',      vector_query_node)
graph.add_node('feature_extractor', feature_extractor_node)
graph.add_node('preference_gate',   preference_gate_node)
graph.add_node('web_searcher',      web_searcher_node)
graph.add_node('db_updater',        db_updater_node)
graph.add_node('profile_updater',   profile_updater_node)
graph.add_node('synthesizer',       synthesizer_node)

graph.set_entry_point('spec_extractor')

graph.add_conditional_edges('spec_extractor', route_spec,
    { 'clarify': 'clarifier', 'proceed': 'vector_query' })
graph.add_edge('clarifier',         END)            # pause; resume on user reply
graph.add_edge('vector_query',      'feature_extractor')
graph.add_edge('feature_extractor', END)            # pause; send features to UI

graph.add_conditional_edges('preference_gate', route_prefs,
    { 'more_feedback': END, 'proceed': 'web_searcher' })

graph.add_edge('web_searcher',      'db_updater')
graph.add_edge('db_updater',        'profile_updater')
graph.add_edge('profile_updater',   'synthesizer')
graph.add_edge('synthesizer',       END)
```

> The graph pauses at `clarifier` and `feature_extractor`. The REST/SSE layer resumes it by injecting the user's reply back into `AgentState` and calling `graph.ainvoke()` again with the updated state.

### 4.2 Node Specifications

---

#### Step 1 — `spec_extractor_node`

**Input:** `AgentState` with latest user message. **Output:** `AgentState.spec` populated.

- Call OpenAI chat completion with a system prompt instructing it to extract a `ProductSpec` JSON from the user message.
- Use `gpt-4o-mini` for this node to reduce cost.
- Set `spec.is_complete = False` and populate `missing_fields` if category is absent or constraints are ambiguous.
- Set `next_action = 'clarify'` or `'proceed'` accordingly.

---

#### Step 2 — `clarifier_node`

**Input:** `AgentState.spec.missing_fields`. **Output:** `ChatMessage` of type `'clarify'`.

- Generate a natural-language clarifying question covering all missing fields in a single message.
- Append message to `AgentState.messages` and set `next_action = END`.
- The client resumes the graph when the user answers via a new `POST /api/chat` request.

---

#### Step 3 — `vector_query_node`

**Input:** `AgentState.spec`. **Output:** `AgentState.candidate_products` (top 5 results).

- Embed the spec (join category + keywords + constraints as text) using `text-embedding-3-small`.
- Cache embeddings by spec text hash — don't re-embed identical queries.
- Query Pinecone index with the embedding. Apply metadata filters for `price_max`, `category`, `brand` if present in `spec.constraints`.
- Deserialize results into `Product` objects. Store in `AgentState.candidate_products`.

---

#### Step 4a — `feature_extractor_node`

**Input:** `AgentState.candidate_products` (max 5). **Output:** Each `Product.features` populated.

- For each candidate product, call OpenAI with the prompt: *"Extract the 4–6 most important purchasable features of this product as a JSON array of Feature objects."*
- Use `gpt-4o-mini` for this node to reduce cost.
- Features must be concrete and value-bearing (e.g. `price: $999`, `battery: 15hr`, `display: OLED 4K`).
- Avoid generic features like `quality` or `design`. Prefer measurable attributes.
- Attach extracted features to each Product. Emit a `ChatMessage` of type `'feature_select'` with the products payload.
- Set `next_action = END` to pause and wait for user feedback.

---

#### Step 4b — `preference_gate_node`

**Input:** `AgentState.feature_feedback` (user-annotated). **Output:** Updated `preference_vector`.

- Aggregate `feature_feedback` into `AgentState.preference_vector`: liked features → `+1.0`, disliked → `-1.0`, neutral → `0`.
- Average with existing `preference_vector` if this is not the first feedback round.
- Route to `'more_feedback'` if fewer than 3 features have been rated, else `'proceed'`.

---

#### Step 5 — `web_searcher_node`

**Input:** `AgentState.spec` + `AgentState.preference_vector`. **Output:** `AgentState.web_results`.

Use DuckDuckGo search via `langchain_community.tools.DuckDuckGoSearchRun`. This is free and requires no API key.

- Build a search query string from `spec.keywords` + top-weighted liked features.
- Call DuckDuckGo search. Retrieve top 10 results.
- Parse each result into a `Product` object. Set `source = 'web_search'`.
- Do not embed at this stage — `db_updater` handles that.

---

#### Step 6 — `db_updater_node`

**Input:** `AgentState.web_results`. **Output:** Pinecone upsert + `AgentState.web_results` with IDs.

Always upsert unconditionally. Pinecone upserts are idempotent and cheap — no need to fetch existing vectors to check for changes.

- For each web result: embed the product description using `text-embedding-3-small`.
- Cache embeddings by product URL hash — skip re-embedding products already in cache.
- Upsert into Pinecone using product URL hash as a stable ID.
- Store metadata: `name`, `price`, `url`, `category`, `source`, `features_json` (as JSON string), `updated_at` (epoch).

---

#### Step 7 — `profile_updater_node`

**Input:** `AgentState.preference_vector` + `AgentState.user_profile`. **Output:** Persisted `UserProfile`.

- Merge `AgentState.preference_vector` into `user_profile.feature_preferences` using exponential moving average (`alpha=0.3`) to prevent single-session overwriting of long-term prefs.
- Append liked/disliked product IDs. Append a 1-sentence summary of this session's query to `session_history`.
- Write updated profile to SQLite via `user_profile` service. This persists across sessions.

---

#### Step 8 — `synthesizer_node`

**Input:** `AgentState.web_results` + `AgentState.user_profile` + `AgentState.preference_vector`. **Output:** Final ranked `ChatMessage`.

- Score each product: base score from Pinecone similarity + `preference_vector` dot product against product features.
- Apply `disliked_products` exclusion filter from `user_profile`.
- Return top 5 products sorted by score.
- Call OpenAI to generate a 1–2 sentence explanation for each product's recommendation based on the user's stated preferences.
- Emit a `ChatMessage` of type `'recommendations'` with ranked `Product` list as payload.

---

## 5. API Surface (FastAPI)

### 5.1 REST Chat — `POST /api/chat/{user_id}`

Primary endpoint for all agent interaction. Returns SSE stream for real-time updates.

| Direction | Format |
|-----------|--------|
| Client → Server | `POST` with JSON body: `{ type: 'message', content: string }` or `{ type: 'feature_feedback', features: Feature[] }` |
| Server → Client | SSE stream of `ChatMessage` objects as JSON events. Content-Type: `text/event-stream`. |
| Session state | `AgentState` stored server-side in memory, keyed by `user_id` with 2hr TTL. |
| Resume logic | On `'message'` receipt: re-inject into `AgentState.messages` and call `graph.ainvoke()`. On `'feature_feedback'`: populate `AgentState.feature_feedback` and invoke `preference_gate_node`. |

### 5.2 REST Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/api/chat/{user_id}` | Send message or feedback, returns SSE stream |
| `GET` | `/api/users/{user_id}/profile` | Fetch persisted `UserProfile` for session init |
| `PATCH` | `/api/users/{user_id}/profile` | Partial update to `UserProfile` (admin/debug) |
| `DELETE` | `/api/users/{user_id}/session` | Clear active `AgentState` from memory |
| `GET` | `/api/products/search` | Direct vector search (query params: `q`, `top_k`) |
| `GET` | `/api/health` | Liveness probe |

---

## 6. Frontend (React + Vite)

The frontend is a standalone React SPA. It communicates via REST + SSE for all agent interaction, and via REST for profile fetch on load.

### 6.1 Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `ChatWindow` | Renders message history. Dispatches user input via `useChat` hook. |
| `MessageBubble` | Renders a single `ChatMessage`. Switches on `message_type` to render text, `ProductGrid`, or `FeatureSelector` inline. |
| `FeatureSelector` | Displays extracted features as cards. Each card has a thumbs-up / thumbs-down toggle. On submit, sends `{ type: 'feature_feedback', features }` via POST. |
| `ProductCard` | Renders a Product: image, name, price, URL, LLM-generated explanation, feature pills. |
| `ProductGrid` | Responsive grid of `ProductCard`s. Used for both candidate suggestions (step 3) and final recommendations (step 8). |
| `useChat` (hook) | Owns SSE + REST lifecycle. Exposes `sendMessage(content)`, `sendFeedback(features)`, `messages[]`, `isLoading`. Handles SSE reconnection. |
| `sessionStore` (Zustand) | Holds `user_id` (UUID generated on first visit, stored in `localStorage`), `UserProfile`, and current `AgentState` snapshot for display. |

### 6.2 `message_type` Rendering Rules

`MessageBubble` must switch on `message_type` to render the correct UI:

- `'text'` → plain text bubble
- `'clarify'` → text bubble with a distinct amber left-border style
- `'feature_select'` → render `FeatureSelector` component with `payload.products`
- `'products'` → render `ProductGrid` with `payload.products` (DB suggestions, step 3)
- `'recommendations'` → render `ProductGrid` with `payload.products` (final ranked, step 8)

---

## 7. Pinecone Index Setup

One shared index for all users. All per-user preference filtering happens at query time via the `preference_vector`, not via namespace isolation.

| Setting | Value |
|---------|-------|
| Index name | `shopping-assistant-products` |
| Dimension | `1536` (text-embedding-3-small) |
| Metric | `cosine` |
| Pod type | `p1.x1` (or serverless) |
| Metadata fields | `name` (str), `price` (float), `url` (str), `category` (str), `source` (str), `features_json` (str), `updated_at` (int epoch) |

Query-time metadata filtering example (price cap from `spec.constraints`):

```python
index.query(
    vector=embedding,
    top_k=5,
    filter={
        'price':    { '$lte': spec.constraints['price_max'] },
        'category': { '$eq':  spec.category }
    }
)
```

---

## 8. SQLite Schema

Using SQLite instead of PostgreSQL. No migration tool needed — use SQLAlchemy with `create_all()` for v1. SQLite is zero-config, file-based, and sufficient for single-server deployment with no auth.

Two tables required.

```sql
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id              TEXT PRIMARY KEY REFERENCES users(user_id),
    feature_preferences  TEXT    NOT NULL DEFAULT '{}',    -- JSON string
    liked_products       TEXT             DEFAULT '[]',    -- JSON array string
    disliked_products    TEXT             DEFAULT '[]',    -- JSON array string
    session_history      TEXT             DEFAULT '[]',    -- JSON array string
    updated_at           TEXT             DEFAULT (datetime('now'))
);
```

On session start: fetch `user_profile` by `user_id`. If no row exists, insert a blank profile. Pass the loaded `UserProfile` into `AgentState` as the starting preference signal for the synthesizer node.

---

## 9. Error Handling Strategy

This section defines fallback behavior when external services fail.

| Failure | Behavior |
|---------|----------|
| **OpenAI API down/timeout** | Retry once with 2s backoff. On second failure, return a canned message: "I'm having trouble processing right now. Please try again in a moment." Do not crash the session — preserve `AgentState`. |
| **Pinecone returns 0 results** | Skip `feature_extractor_node`. Route directly to `web_searcher_node` with a note to the user: "I don't have matching products in my database yet — searching the web for you." |
| **DuckDuckGo search fails** | Return whatever `candidate_products` exist from the vector query. Append a note: "Web search is temporarily unavailable. Showing results from my existing catalog." If no candidates either, ask the user to refine their query. |
| **Malformed user input** | `spec_extractor_node` sets `is_complete=False` with a clarifying question. Never crash on bad input. |
| **SQLite write failure** | Log the error. The session continues without persisting the profile update — preferences are still held in `AgentState` for the current session. |
| **Embedding generation fails** | Skip the product that failed embedding. Continue with remaining products. Log which product was skipped. |

---

## 10. Rate Limiting & Cost Controls

Every user turn can trigger multiple LLM and embedding calls. These guardrails prevent runaway costs.

| Control | Value | Rationale |
|---------|-------|-----------|
| Max products for feature extraction | 5 | Each product = 1 LLM call. 5 products = 5 calls max. |
| Embedding cache | By content hash (spec text or product URL) | Don't re-embed identical queries or known products. Use an in-memory `dict[str, list[float]]`. |
| Model for spec extraction | `gpt-4o-mini` | Structured extraction doesn't need full `gpt-4o` capability. |
| Model for feature extraction | `gpt-4o-mini` | Same reasoning — JSON extraction task. |
| Model for synthesis | `gpt-4o` | Final recommendation quality matters. Keep the best model here. |
| Per-session LLM call budget | 50 calls | Track in `AgentState`. After 50 calls, return a message asking the user to start a new session. |
| Pinecone `top_k` | 5 | Downstream feature extraction cost scales linearly with result count. |

---

## 11. Implementation Phases

Recommended build order. Each phase should be independently testable before proceeding.

| Phase | Name | Deliverables |
|-------|------|-------------|
| 1 | Foundation | FastAPI project scaffold, `.env` loading, Pinecone + OpenAI + SQLite clients, `/api/health` |
| 2 | Spec Extraction | `spec_extractor_node` + `clarifier_node`, LangGraph graph skeleton, unit tests with mock LLM |
| 3 | Vector Pipeline | `vector_query_node` + `db_updater_node`, Pinecone index creation script. Seed dynamically via web search, not manually. |
| 4 | Feature Feedback | `feature_extractor_node`, `preference_gate_node`, `AgentState` `preference_vector` accumulation |
| 5 | Web Search | `web_searcher_node` with DuckDuckGo, integration into graph, `db_updater` triggered post-search |
| 6 | Profile Persistence | `profile_updater_node`, SQLite `user_profiles` table, `UserProfile` CRUD service, EMA merge logic |
| 7 | Synthesizer | `synthesizer_node`, scoring logic (Pinecone sim + `preference_vector` dot product), OpenAI explanation generation |
| 8 | SSE Layer | SSE endpoint `POST /api/chat/{user_id}`, in-memory `AgentState` with TTL, pause/resume logic on `END` nodes |
| 9 | React Frontend | Vite scaffold, `useChat` hook, `ChatWindow`, `MessageBubble` with `message_type` routing, `FeatureSelector`, `ProductGrid`/`Card` |
| 10 | Integration + Polish | E2E test full flow, error handling (per Section 9), cost control enforcement (per Section 10), loading states, reconnect logic, CORS, deployment config |

---

## 12. Key Constraints & Decisions

| Constraint | Decision |
|------------|----------|
| **Spec clarity threshold** | A spec is considered complete when `category` is present AND at least one constraint or 2+ keywords exist. Customize in `spec_extractor_node`. |
| **Preference sufficiency** | `preference_gate` routes to `'proceed'` when >= 3 features have been rated (liked or disliked). Tunable per category. |
| **Profile EMA alpha** | `0.3` — new session preferences update stored prefs at 30% weight. Prevents a single bad session from overwriting the long-term model. |
| **User identity** | `user_id` is a UUID generated client-side on first visit, stored in `localStorage`. No authentication required in v1. |
| **Pinecone namespace** | Use a single default namespace (shared catalog). User preferences influence ranking at query/synthesis time only. |
| **Feature count** | Feature extractor targets 4–6 features per product. Fewer = insufficient signal. More = UI overload. |
| **Web search provider** | DuckDuckGo via LangChain — free, no API key required. |
| **AgentState persistence** | In-memory `dict` with 2hr TTL. On TTL expiry, a new session starts fresh (profile still loaded from SQLite). |
| **LLM cost model** | `gpt-4o-mini` for extraction tasks, `gpt-4o` for synthesis only. Max 5 products per extraction round. 50 LLM calls per session. |
| **Pinecone upsert strategy** | Always upsert. No conditional checks. |
