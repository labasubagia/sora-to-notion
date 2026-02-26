"""
Pytest configuration and shared fixtures for Sora CLI tests.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def tmp_output_dir():
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_generation():
    """Sample Sora/ChatGPT generation data."""
    return {
        "created_at": "2024-01-15T10:30:00.000000+00:00",
        "id": "gen_test123abc",
        "task_id": "task_xyz789",
        "conversation_id": "conv_abc123",
        "message_id": "msg_def456",
        "asset_pointer": "asset_ghi789",
        "url": "https://example.com/image.png",
        "prompt": "A beautiful sunset over mountains, photorealistic, 4K",
    }


@pytest.fixture
def sample_generations(sample_generation):
    """List of sample generations."""
    return [sample_generation] * 3


@pytest.fixture
def sample_notion_response():
    """Sample Notion API response for page creation."""
    return {
        "id": "page_123abc",
        "created_time": "2024-01-15T10:30:00.000Z",
        "properties": {
            "Name": {
                "title": [{"text": {"content": "gen_test123abc.png"}}]
            },
            "Image": {
                "files": [
                    {
                        "type": "file_upload",
                        "file_upload": {"id": "upload_456def"},
                    }
                ]
            },
            "Prompt": {
                "rich_text": [{"text": {"content": "Test prompt"}}]
            },
            "Model": {"select": {"name": "Sora"}},
            "Face": {"select": {"name": "_original_"}},
        },
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_test_notion_key")
    monkeypatch.setenv("NOTION_DATABASE_ID", "test_database_123")
    monkeypatch.setenv("CHATGPT_AUTHORIZATION_TOKEN", "test_token_abc")
    monkeypatch.setenv("CHATGPT_USER_AGENT", "TestAgent/1.0")
    monkeypatch.setenv("CHATGPT_COOKIE_STRING_BASE64", "dGVzdF9jb29raWU=")


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session for testing."""
    # Create proper async context manager mock
    class MockResponse:
        def __init__(self, json_data, status=200):
            self._json_data = json_data
            self.status = status
        
        async def json(self):
            return self._json_data
        
        def raise_for_status(self):
            if self.status >= 400:
                raise Exception(f"HTTP {self.status}")
    
    class MockSession:
        def __init__(self):
            self._responses = []
        
        def _mock_method(self, method, url, **kwargs):
            class ContextManager:
                async def __aenter__(ctx_self):
                    if self._responses:
                        return self._responses.pop(0)
                    return MockResponse({})
                
                async def __aexit__(ctx_self, *args):
                    pass
            return ContextManager()
        
        def get(self, url, **kwargs):
            return self._mock_method('GET', url, **kwargs)
        
        def post(self, url, **kwargs):
            return self._mock_method('POST', url, **kwargs)
        
        def delete(self, url, **kwargs):
            return self._mock_method('DELETE', url, **kwargs)
        
        def patch(self, url, **kwargs):
            return self._mock_method('PATCH', url, **kwargs)
    
    return MockSession()


def make_mock_response(json_data, status=200):
    """Helper to create mock HTTP response."""
    class MockResp:
        def __init__(self, data, s):
            self._data = data
            self.status = s
        
        async def json(self):
            return self._data
        
        def raise_for_status(self):
            pass
    
    return MockResp(json_data, status)


@pytest.fixture
def tmp_env_file(tmp_path):
    """Create a temporary .env file for testing."""
    env_file = tmp_path / ".env"
    env_file.write_text("""
NOTION_API_KEY=test_notion_key_123
NOTION_DATABASE_ID=test_db_456
CHATGPT_AUTHORIZATION_TOKEN=test_token_abc
CHATGPT_USER_AGENT=TestAgent/1.0
CHATGPT_COOKIE_STRING_BASE64=dGVzdF9jb29raWU=
""")
    return env_file


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset module caches before each test."""
    # Import here to avoid circular imports
    import notion
    
    # Clear caches
    notion._db_data_sources_cache.clear()
    notion._db_page_cache.clear()
    
    yield
    
    # Cleanup after test
    notion._db_data_sources_cache.clear()
    notion._db_page_cache.clear()


@pytest.fixture
def sample_image_bytes():
    """Return minimal valid PNG image bytes."""
    # Minimal 1x1 transparent PNG
    return bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,  # IEND chunk
        0x42, 0x60, 0x82,
    ])
