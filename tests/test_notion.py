"""
Integration tests for notion.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_mock_response
from notion import (
    _db_data_sources_cache,
    _db_page_cache,
    add_page_to_db,
    create_upload_img,
    get_db_data_sources,
    get_headers,
    is_page_exists_in_db,
    query_data_source,
    send_upload_img,
)


@pytest.mark.integration
class TestNotionHeaders:
    """Tests for Notion headers generation."""

    def test_headers_contain_auth(self, mock_env_vars):
        """Headers should contain Authorization."""
        headers = get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_headers_contain_notion_version(self, mock_env_vars):
        """Headers should contain Notion-Version."""
        headers = get_headers()
        assert "Notion-Version" in headers
        assert headers["Notion-Version"] == "2025-09-03"

    def test_headers_contain_content_type(self, mock_env_vars):
        """Headers should contain Content-Type."""
        headers = get_headers()
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"


@pytest.mark.integration
class TestNotionDatabase:
    """Tests for Notion database operations."""

    async def test_get_db_data_sources(self, mock_aiohttp_session):
        """Should fetch database data sources."""
        mock_aiohttp_session._responses = [
            make_mock_response({"data_sources": [{"id": "ds_123"}]})
        ]

        sources = await get_db_data_sources(mock_aiohttp_session, "test_db_123")
        assert isinstance(sources, list)
        assert len(sources) == 1
        assert sources[0]["id"] == "ds_123"

    async def test_get_db_data_sources_uses_cache(self, mock_aiohttp_session):
        """Should use cache for repeated calls."""
        _db_data_sources_cache.clear()
        _db_data_sources_cache["test_db"] = [{"id": "cached_ds"}]

        sources = await get_db_data_sources(mock_aiohttp_session, "test_db")
        
        assert sources == [{"id": "cached_ds"}]

    async def test_query_data_source(self, mock_aiohttp_session):
        """Should query data source."""
        mock_aiohttp_session._responses = [
            make_mock_response({"results": [{"id": "page_1"}]})
        ]

        result = await query_data_source(mock_aiohttp_session, "ds_123", "test.png")
        assert "results" in result
        assert len(result["results"]) == 1

    async def test_is_page_exists_in_db(self, mock_aiohttp_session):
        """Should check if page exists in database."""
        with patch("notion.get_db_data_sources", new_callable=AsyncMock) as mock_get_ds:
            mock_get_ds.return_value = [{"id": "ds_123"}]
            
            with patch("notion.query_data_source", new_callable=AsyncMock) as mock_query:
                mock_query.return_value = {
                    "results": [
                        {"properties": {"Name": {"title": [{"text": {"content": "test.png"}}]}}}
                    ]
                }

                exists = await is_page_exists_in_db(
                    mock_aiohttp_session, "test_db_123", "test.png"
                )
                assert exists is True

    async def test_is_page_exists_not_found(self, mock_aiohttp_session):
        """Should return False if page not found."""
        with patch("notion.get_db_data_sources", new_callable=AsyncMock) as mock_get_ds:
            mock_get_ds.return_value = [{"id": "ds_123"}]
            
            with patch("notion.query_data_source", new_callable=AsyncMock) as mock_query:
                mock_query.return_value = {"results": []}

                exists = await is_page_exists_in_db(
                    mock_aiohttp_session, "test_db_123", "nonexistent.png"
                )
                assert exists is False

    async def test_is_page_exists_uses_cache(self, mock_aiohttp_session):
        """Should use cache for repeated checks."""
        _db_page_cache.clear()
        _db_page_cache.add("cached_image.png")

        exists = await is_page_exists_in_db(
            mock_aiohttp_session, "test_db_123", "cached_image.png"
        )
        assert exists is True


@pytest.mark.integration
class TestNotionUpload:
    """Tests for Notion upload operations."""

    async def test_create_upload_img(self, mock_aiohttp_session, sample_image_bytes, tmp_path):
        """Should create file upload."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        mock_aiohttp_session._responses = [
            make_mock_response({"id": "upload_123", "filename": "test.png"})
        ]

        result = await create_upload_img(mock_aiohttp_session, str(img_path))
        assert "id" in result
        assert result["id"] == "upload_123"

    async def test_create_upload_img_file_not_found(self, mock_aiohttp_session):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            await create_upload_img(mock_aiohttp_session, "/nonexistent/file.png")

    async def test_send_upload_img(self, mock_aiohttp_session, sample_image_bytes, tmp_path):
        """Should send file upload."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        mock_aiohttp_session._responses = [
            make_mock_response({"id": "upload_123", "status": "complete"})
        ]

        result = await send_upload_img(mock_aiohttp_session, "upload_123", str(img_path))
        assert result is not None

    async def test_add_page_to_db(self, mock_aiohttp_session, sample_image_bytes, tmp_path):
        """Should add page to database."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        mock_aiohttp_session._responses = [
            make_mock_response({"id": "upload_123"}),  # create_upload_img
            make_mock_response({"id": "upload_123", "status": "complete"}),  # send_upload_img
            make_mock_response({  # add_page_to_db
                "id": "page_123",
                "properties": {
                    "Prompt": {"rich_text": [{"text": {"content": "Test prompt"}}]}
                }
            }),
        ]

        result = await add_page_to_db(
            mock_aiohttp_session,
            "test_db_123",
            str(img_path),
            "Test prompt",
            model="Sora",
        )
        assert "id" in result
        assert result["id"] == "page_123"

    async def test_add_page_to_db_file_not_found(self, mock_aiohttp_session):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            await add_page_to_db(
                mock_aiohttp_session,
                "test_db_123",
                "/nonexistent/file.png",
                "Test prompt",
            )


class TestNotionCaching:
    """Tests for Notion caching behavior."""

    def test_data_sources_cache_populated(self, mock_env_vars):
        """Data sources cache should be populated."""
        _db_data_sources_cache.clear()
        _db_data_sources_cache["test_db"] = [{"id": "ds_123"}]

        assert "test_db" in _db_data_sources_cache

    def test_page_cache_populated(self, mock_env_vars):
        """Page cache should be populated."""
        _db_page_cache.clear()
        _db_page_cache.add("test_image.png")

        assert "test_image.png" in _db_page_cache

    def test_cache_cleared_by_fixture(self):
        """Caches should be cleared by reset_caches fixture."""
        assert len(_db_data_sources_cache) == 0
        assert len(_db_page_cache) == 0
