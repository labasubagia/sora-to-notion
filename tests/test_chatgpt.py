"""
Integration tests for chatgpt.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_mock_response

from chatgpt import (
    delete_conversation,
    get_conversation_details,
    get_conversations,
    get_headers,
    get_image_generations,
    get_prompt_from_image_node_in_conversation,
    get_conversation_mapping_key_by_asset_pointer,
)


@pytest.mark.integration
class TestChatGPTHeaders:
    """Tests for ChatGPT headers generation."""

    def test_headers_contain_auth(self, mock_env_vars):
        """Headers should contain Authorization."""
        headers = get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_headers_contain_user_agent(self, mock_env_vars):
        """Headers should contain User-Agent."""
        headers = get_headers()
        assert "User-Agent" in headers

    def test_headers_contain_content_type(self, mock_env_vars):
        """Headers should contain Content-Type."""
        headers = get_headers()
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    def test_headers_decode_cookie(self, monkeypatch):
        """Should decode base64 cookie."""
        # Skip in CI - real env vars may override test values
        import os
        if os.getenv("CI"):
            pytest.skip("Skipping in CI - env vars may override test values")
        
        # Set up test env vars directly to ensure isolation
        monkeypatch.setenv("CHATGPT_COOKIE_STRING_BASE64", "dGVzdF9jb29raWU=")
        monkeypatch.setenv("CHATGPT_AUTHORIZATION_TOKEN", "test_token")
        monkeypatch.setenv("CHATGPT_USER_AGENT", "TestAgent/1.0")
        
        headers = get_headers()
        # dGVzdF9jb29raWU= decodes to "test_cookie"
        assert "Cookie" in headers
        assert headers["Cookie"] == "test_cookie"


@pytest.mark.integration
class TestChatGPTConversations:
    """Tests for ChatGPT conversation operations."""

    async def test_get_conversations(self, mock_aiohttp_session):
        """Should fetch conversations."""
        mock_aiohttp_session._responses = [
            make_mock_response({"items": [{"id": "conv_1"}]})
        ]

        result = await get_conversations(mock_aiohttp_session, limit=5)
        assert "items" in result
        assert len(result["items"]) == 1

    async def test_get_conversations_with_params(self, mock_aiohttp_session):
        """Should fetch conversations with filters."""
        mock_aiohttp_session._responses = [
            make_mock_response({"items": []})
        ]

        result = await get_conversations(
            mock_aiohttp_session, limit=5, is_archived=False, is_starred=False
        )
        assert "items" in result

    async def test_get_conversation_details(self, mock_aiohttp_session):
        """Should fetch conversation details."""
        mock_aiohttp_session._responses = [
            make_mock_response({"mapping": {"node1": {}}})
        ]

        result = await get_conversation_details(mock_aiohttp_session, "conv_abc")
        assert "mapping" in result


@pytest.mark.integration
class TestChatGPTImageGenerations:
    """Tests for ChatGPT image generation operations."""

    async def test_get_image_generations(self, mock_aiohttp_session):
        """Should fetch recent image generations."""
        mock_aiohttp_session._responses = [
            make_mock_response({"items": [{"id": "img_1"}]})
        ]

        result = await get_image_generations(mock_aiohttp_session, limit=5)
        assert "items" in result
        assert len(result["items"]) == 1

    async def test_delete_conversation(self, mock_aiohttp_session):
        """Should delete/hide conversation."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await delete_conversation(mock_aiohttp_session, "conv_test")
        assert result is not None


class TestChatGPTPromptExtraction:
    """Tests for prompt extraction logic."""

    def test_get_prompt_from_user_message(self):
        """Should extract prompt from user message."""
        data = {
            "mapping": {
                "node1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["A beautiful sunset"]},
                    },
                    "parent": None,
                }
            }
        }

        prompt = get_prompt_from_image_node_in_conversation(
            data, "node1", "asset_ghi"
        )
        assert prompt == "A beautiful sunset"

    def test_get_prompt_walks_to_parent(self):
        """Should walk to parent node if current has no prompt."""
        data = {
            "mapping": {
                "node1": {
                    "message": None,
                    "parent": "node2",
                },
                "node2": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Parent prompt"]},
                    },
                    "parent": None,
                },
            }
        }

        prompt = get_prompt_from_image_node_in_conversation(
            data, "node1", "asset_ghi"
        )
        assert prompt == "Parent prompt"

    def test_get_prompt_returns_none_if_not_found(self):
        """Should return None if no prompt found."""
        data = {
            "mapping": {
                "node1": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Not a user message"]},
                    },
                    "parent": None,
                }
            }
        }

        prompt = get_prompt_from_image_node_in_conversation(
            data, "node1", "asset_ghi"
        )
        assert prompt is None

    def test_get_conversation_mapping_key_by_asset_pointer(self):
        """Should find node by asset_pointer."""
        data = {
            "mapping": {
                "node1": {
                    "message": {
                        "content": {
                            "parts": [{"asset_pointer": "asset_123"}]
                        }
                    }
                }
            }
        }

        key = get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert key == "node1"

    def test_get_conversation_mapping_key_returns_none(self):
        """Should return None if asset_pointer not found."""
        data = {"mapping": {"node1": {"message": None}}}

        key = get_conversation_mapping_key_by_asset_pointer(data, "asset_456")
        assert key is None
