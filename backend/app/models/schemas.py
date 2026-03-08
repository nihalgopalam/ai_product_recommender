from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class Feature(BaseModel):
    name: str
    display_name: str
    value: str
    sentiment: Literal["liked", "disliked", "neutral"] = "neutral"


class ProductSpec(BaseModel):
    category: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    is_complete: bool = False
    missing_fields: list[str] = Field(default_factory=list)


class Product(BaseModel):
    id: str
    name: str
    description: str
    price: float | None = None
    url: str
    source: Literal["vector_db", "web_search"]
    features: list[Feature] = Field(default_factory=list)
    embedding: list[float] | None = Field(default=None, exclude=True)


class UserProfile(BaseModel):
    user_id: str
    feature_preferences: dict[str, float] = Field(default_factory=dict)
    disliked_products: list[str] = Field(default_factory=list)
    liked_products: list[str] = Field(default_factory=list)
    session_history: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    message_type: Literal["text", "clarify", "feature_select", "products", "recommendations"] = "text"
    payload: dict[str, Any] | None = None


# --- Request/response bodies for API endpoints ---

class ChatRequest(BaseModel):
    type: Literal["message", "feature_feedback"]
    content: str | None = None
    features: list[Feature] | None = None


class ProfilePatch(BaseModel):
    feature_preferences: dict[str, float] | None = None
    disliked_products: list[str] | None = None
    liked_products: list[str] | None = None
