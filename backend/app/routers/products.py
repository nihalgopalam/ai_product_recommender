from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import Product

router = APIRouter(prefix="/api/products", tags=["products"])


@router.get("/search", response_model=list[Product])
async def search_products(
    q: str = Query(..., description="Search query text"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return"),
) -> list[Product]:
    """Direct Pinecone vector search. Bypasses the agent graph."""
    raise HTTPException(status_code=501, detail="Not implemented — Pinecone service not wired yet")
