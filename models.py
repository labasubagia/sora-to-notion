"""Type definitions for image generation data structures."""

from typing import Protocol, TypedDict


class ImageGeneration(Protocol):
    """Protocol for image generation data with common fields."""

    id: str
    prompt: str


class ChatGPTImageGeneration(TypedDict):
    """Image generation from ChatGPT DALL-E."""

    created_at: str
    id: str
    conversation_id: str
    message_id: str
    asset_pointer: str
    url: str
    prompt: str


class SoraImageGeneration(TypedDict):
    """Image generation from Sora."""

    created_at: str
    id: str
    task_id: str
    url: str
    prompt: str
