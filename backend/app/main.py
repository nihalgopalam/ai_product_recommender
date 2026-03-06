from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, products, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    # In-memory AgentState store: {user_id: AgentState}. Phase 1 adds Pinecone/OpenAI/SQLite here.
    app.state.sessions = {}  # dict[str, AgentState]
    yield
    app.state.sessions.clear()


settings = get_settings()

app = FastAPI(
    title="Shopping Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(users.router)
app.include_router(products.router)


@app.get("/api/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok", "version": app.version}

@app.get("/api/routes", tags=["routes"])
async def routes()-> dict:
    routes = [(r.methods, r.path) for r in app.routes if hasattr(r, 'path') and hasattr(r, 'methods') and r.methods]
    return {'routes': routes}
