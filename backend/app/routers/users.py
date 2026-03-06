from fastapi import APIRouter, HTTPException

from app.models.schemas import ProfilePatch, UserProfile

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/{user_id}/profile", response_model=UserProfile)
async def get_profile(user_id: str) -> UserProfile:
    """Fetch persisted UserProfile. Called on session init to seed AgentState."""
    raise HTTPException(status_code=501, detail="Not implemented — SQLite service not wired yet")


@router.patch("/{user_id}/profile", response_model=UserProfile)
async def patch_profile(user_id: str, body: ProfilePatch) -> UserProfile:
    """Partial update to UserProfile. Intended for admin/debug use."""
    raise HTTPException(status_code=501, detail="Not implemented — SQLite service not wired yet")


@router.delete("/{user_id}/session", status_code=204)
async def clear_session(user_id: str) -> None:
    """Clear the in-memory AgentState for this user (force fresh session)."""
    raise HTTPException(status_code=501, detail="Not implemented — session store not wired yet")
