from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat/{user_id}")
async def chat(user_id: str, body: ChatRequest) -> StreamingResponse:
    """
    Primary chat endpoint. Accepts a user message or feature feedback.
    Returns an SSE stream of ChatMessage events.

    body.type == 'message'          → inject into AgentState, invoke graph
    body.type == 'feature_feedback' → populate feature_feedback, invoke preference_gate
    """
    raise HTTPException(status_code=501, detail="Not implemented — agent graph not wired yet")
