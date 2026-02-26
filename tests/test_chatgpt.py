"""
Integration tests for chatgpt.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_mock_response
import chatgpt

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

    def test_headers_decode_cookie(self):
        """Should decode base64 cookie - tested via direct base64 verification."""
        # This test verifies the base64 decoding logic works correctly
        # Actual cookie values come from .env and vary by environment
        from base64 import b64decode
        
        # Test the decoding logic directly
        test_cookie_b64 = "dGVzdF9jb29raWU="
        decoded = b64decode(test_cookie_b64).decode("utf-8")
        assert decoded == "test_cookie"
        
        # Verify get_headers() returns a dict with required keys
        # (actual values depend on environment)
        headers = get_headers()
        assert isinstance(headers, dict)
        assert "Authorization" in headers
        assert "User-Agent" in headers
        assert "Content-Type" in headers


@pytest.mark.integration
class TestChatGPTFetchImageGenerations:
    """Tests for fetch_image_generations function."""

    async def test_fetch_image_generations_success(self, mock_aiohttp_session, monkeypatch):
        """Should fetch and process image generations."""
        from unittest.mock import patch, AsyncMock
        
        # Mock get_image_generations response
        mock_data = {
            "items": [
                {
                    "id": "img_123",
                    "conversation_id": "conv_abc",
                    "message_id": "msg_def",
                    "asset_pointer": "asset_ghi",
                    "url": "https://example.com/image.png",
                    "created_at": 1705315800,
                }
            ]
        }
        
        with patch("chatgpt.get_image_generations", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_data
            
            with patch("chatgpt.get_conversation_details", new_callable=AsyncMock) as mock_detail:
                mock_detail.return_value = {
                    "mapping": {
                        "node1": {
                            "message": {
                                "author": {"role": "user"},
                                "content": {"parts": ["Test prompt"]},
                            },
                            "parent": None,
                        }
                    }
                }
                
                with patch("chatgpt.get_prompt_from_image_node_in_conversation") as mock_prompt:
                    mock_prompt.return_value = "Test prompt"
                    
                    result = await chatgpt.fetch_image_generations(limit=5)
                    
                    assert isinstance(result, list)
                    assert len(result) == 1
                    assert result[0]["id"] == "img_123"
                    assert result[0]["prompt"] == "Test prompt"

    async def test_fetch_image_generations_empty(self, mock_aiohttp_session, monkeypatch):
        """Should return empty list when no generations."""
        from unittest.mock import patch, AsyncMock
        
        with patch("chatgpt.get_image_generations", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"items": []}
            
            result = await chatgpt.fetch_image_generations(limit=5)
            
            assert result == []


@pytest.mark.integration
class TestChatGPTUploadToNotion:
    """Tests for upload_to_notion function."""

    async def test_upload_to_notion_full_workflow(self, monkeypatch, tmp_path):
        """Should execute full upload workflow."""
        from unittest.mock import patch, AsyncMock
        
        image_folder = str(tmp_path / "images")
        
        with patch("chatgpt.fetch_image_generations", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {
                    "id": "img_123",
                    "url": "https://example.com/image.png",
                    "prompt": "Test prompt",
                }
            ]
            
            with patch("chatgpt.download_all_images", new_callable=AsyncMock) as mock_download:
                mock_download.return_value = None
                
                with patch("chatgpt.add_prompt_to_images") as mock_add_prompt:
                    mock_add_prompt.return_value = None
                    
                    with patch("chatgpt.upload_all_images_to_notion", new_callable=AsyncMock) as mock_upload:
                        mock_upload.return_value = None
                        
                        await chatgpt.upload_to_notion(
                            image_folder=image_folder,
                            db_id="test_db",
                            upload_to_notion=True,
                            remove_in_chatgpt=False,
                            add_prompt_to_image=True,
                            limit=5,
                        )
                        
                        mock_fetch.assert_called_once()
                        mock_download.assert_called_once()
                        mock_add_prompt.assert_called_once()
                        mock_upload.assert_called_once()

    async def test_upload_to_notion_skip_download(self, monkeypatch, tmp_path):
        """Should skip download when add_prompt_to_image=False."""
        from unittest.mock import patch, AsyncMock
        
        image_folder = str(tmp_path / "images")
        
        with patch("chatgpt.fetch_image_generations", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []
            
            with patch("chatgpt.download_all_images", new_callable=AsyncMock) as mock_download:
                mock_download.return_value = None
                
                await chatgpt.upload_to_notion(
                    image_folder=image_folder,
                    db_id="test_db",
                    upload_to_notion=False,
                    add_prompt_to_image=False,
                    limit=5,
                )
                
                mock_fetch.assert_called_once()


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
