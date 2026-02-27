"""Type definitions for image generation data structures."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class ImageGeneration(Protocol):
    """Protocol for image generation data with common fields."""

    id: str
    prompt: str


class ChatGPTImageGeneration(BaseModel):
    """Image generation from ChatGPT DALL-E."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    id: str
    conversation_id: str
    message_id: str
    asset_pointer: str
    url: str
    prompt: str = ""


class SoraImageGeneration(BaseModel):
    """Image generation from Sora."""

    model_config = ConfigDict(extra="forbid")

    created_at: str | None = None
    id: str
    task_id: str | None = None
    url: str | None = None
    prompt: str = ""
