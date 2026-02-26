"""
Integration tests for sora.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""
from unittest.mock import AsyncMock

import pytest

from tests.conftest import make_mock_response
from sora import (
    archive_generation,
    archive_task,
    delete_generation,
    delete_task,
    fetch_all_lists_tasks,
    fetch_list_tasks,
    fetch_recent_tasks,
    get_generations_from_tasks,
    get_headers,
)


@pytest.mark.integration
class TestSoraHeaders:
    """Tests for Sora headers generation."""

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


@pytest.mark.integration
class TestSoraTasks:
    """Tests for Sora task operations."""

    async def test_fetch_recent_tasks(self, mock_aiohttp_session):
        """Should fetch recent tasks."""
        mock_aiohttp_session._responses = [
            make_mock_response({"task_responses": [{"id": "task_1"}]})
        ]

        result = await fetch_recent_tasks(mock_aiohttp_session, limit=5)
        assert "task_responses" in result
        assert len(result["task_responses"]) == 1

    async def test_fetch_list_tasks(self, mock_aiohttp_session):
        """Should fetch list of tasks."""
        mock_aiohttp_session._responses = [
            make_mock_response({"task_responses": [{"id": "task_1"}]})
        ]

        result = await fetch_list_tasks(mock_aiohttp_session, limit=5)
        assert "task_responses" in result

    async def test_fetch_all_lists_tasks(self, monkeypatch):
        """Should fetch all tasks with pagination."""
        from unittest.mock import AsyncMock, patch
        
        # Mock the entire function to avoid real API calls
        mock_tasks = [
            {"id": "task_1", "generations": []},
            {"id": "task_2", "generations": []},
        ]
        
        with patch("sora.fetch_list_tasks", new_callable=AsyncMock) as mock_fetch:
            # First call returns has_more=True
            # Second call returns has_more=False
            mock_fetch.side_effect = [
                {"task_responses": [mock_tasks[0]], "has_more": True, "last_id": "task_1"},
                {"task_responses": [mock_tasks[1]], "has_more": False, "last_id": None},
            ]
            
            tasks = await fetch_all_lists_tasks(limit=5)
            assert isinstance(tasks, list)
            assert len(tasks) == 2
            assert mock_fetch.call_count == 2

    async def test_archive_task(self, mock_aiohttp_session):
        """Should archive/trash a task."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await archive_task(mock_aiohttp_session, "task_test")
        assert result is not None

    async def test_delete_task(self, mock_aiohttp_session):
        """Should delete a task."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await delete_task(mock_aiohttp_session, "task_test")
        assert result is not None


@pytest.mark.integration
class TestSoraGenerations:
    """Tests for Sora generation operations."""

    async def test_archive_generation(self, mock_aiohttp_session):
        """Should archive/trash a generation."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await archive_generation(mock_aiohttp_session, "gen_test")
        assert result is not None

    async def test_delete_generation(self, mock_aiohttp_session):
        """Should delete a generation."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await delete_generation(mock_aiohttp_session, "gen_test")
        assert result is not None


class TestSoraDataProcessing:
    """Tests for Sora data processing functions."""

    def test_get_generations_from_tasks(self):
        """Should extract generations from tasks."""
        tasks = [
            {
                "id": "task_123",
                "created_at": "2024-01-15T10:30:00Z",
                "generations": [
                    {
                        "id": "gen_abc",
                        "task_id": "task_123",
                        "url": "https://example.com/img1.png",
                        "prompt": "First prompt",
                    },
                    {
                        "id": "gen_def",
                        "task_id": "task_123",
                        "url": "https://example.com/img2.png",
                        "prompt": "Second prompt",
                    },
                ],
            }
        ]

        generations = get_generations_from_tasks(tasks)

        assert len(generations) == 2
        assert generations[0]["id"] == "gen_abc"
        assert generations[0]["prompt"] == "First prompt"
        assert generations[1]["id"] == "gen_def"

    def test_get_generations_sorted_by_created_at(self):
        """Should sort generations by created_at."""
        tasks = [
            {
                "id": "task_1",
                "created_at": "2024-01-15T12:00:00Z",
                "generations": [
                    {"id": "gen_later", "task_id": "task_1", "url": "url2", "prompt": "p2"}
                ],
            },
            {
                "id": "task_2",
                "created_at": "2024-01-15T10:00:00Z",
                "generations": [
                    {"id": "gen_earlier", "task_id": "task_2", "url": "url1", "prompt": "p1"}
                ],
            },
        ]

        generations = get_generations_from_tasks(tasks)

        assert len(generations) == 2
        assert generations[0]["id"] == "gen_earlier"
        assert generations[1]["id"] == "gen_later"

    def test_get_generations_empty_tasks(self):
        """Should handle tasks with no generations."""
        tasks = [
            {"id": "task_1", "created_at": "2024-01-15T10:00:00Z", "generations": []},
            {"id": "task_2", "created_at": "2024-01-15T11:00:00Z", "generations": []},
        ]

        generations = get_generations_from_tasks(tasks)
        assert len(generations) == 0

    def test_get_generations_missing_fields(self):
        """Should handle missing fields gracefully."""
        tasks = [
            {
                "id": "task_1",
                "created_at": "2024-01-15T10:00:00Z",
                "generations": [
                    {"id": "gen_1"},
                ],
            }
        ]

        generations = get_generations_from_tasks(tasks)

        assert len(generations) == 1
        assert generations[0]["task_id"] is None
        assert generations[0]["url"] is None
        assert generations[0]["prompt"] is None
