"""
CLI tests for main.py using Typer's test utilities.
"""
import re
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from main import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    return ansi_escape.sub('', text)


class TestCLIHelp:
    """Tests for CLI help messages."""

    def test_main_help(self):
        """Should show main help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "sora-upload-to-notion" in result.stdout
        assert "chatgpt-upload-to-notion" in result.stdout
        assert "sora-cleanup-trash" in result.stdout
        assert "sora-cleanup-tasks" in result.stdout
        assert "clean-output-path" in result.stdout

    def test_sora_upload_help(self):
        """Should show sora-upload-to-notion help."""
        result = runner.invoke(app, ["sora-upload-to-notion", "--help"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--image-folder" in clean_output
        assert "--db-id" in clean_output
        assert "--trash-in-sora" in clean_output

    def test_chatgpt_upload_help(self):
        """Should show chatgpt-upload-to-notion help."""
        result = runner.invoke(app, ["chatgpt-upload-to-notion", "--help"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--image-folder" in clean_output
        assert "--limit" in clean_output


class TestCLIValidation:
    """Tests for CLI input validation."""

    def test_invalid_db_id_too_short(self):
        """Should reject database IDs that are too short."""
        result = runner.invoke(
            app,
            ["sora-upload-to-notion", "--db-id", "short"],
        )
        assert result.exit_code != 0
        # Error message may be in stdout or stderr
        output = result.stdout + str(result.stderr)
        assert "Notion database ID must be a valid ID" in output or "Invalid value" in output

    def test_invalid_db_id_empty(self):
        """Should reject empty database IDs."""
        result = runner.invoke(
            app,
            ["sora-upload-to-notion", "--db-id", ""],
        )
        assert result.exit_code != 0


class TestCLICommands:
    """Tests for CLI command execution."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_env_vars, tmp_output_dir, monkeypatch):
        """Setup test environment."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        # Mock env var validation to pass in tests
        monkeypatch.setattr("util.validate_env_vars", lambda x: None)
        yield

    @patch("sora.upload_to_notion", new_callable=AsyncMock)
    def test_sora_upload_to_notion(self, mock_upload, mock_env_vars):
        """Should call sora.upload_to_notion."""
        result = runner.invoke(
            app,
            [
                "sora-upload-to-notion",
                "--image-folder", "test_images",
                "--db-id", "test_db_12345678901234567890",
                "--no-trash-in-sora",
                "--no-remove-in-sora",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()

    @patch("chatgpt.upload_to_notion", new_callable=AsyncMock)
    def test_chatgpt_upload_to_notion(self, mock_upload, mock_env_vars):
        """Should call chatgpt.upload_to_notion."""
        result = runner.invoke(
            app,
            [
                "chatgpt-upload-to-notion",
                "--image-folder", "test_images",
                "--db-id", "test_db_12345678901234567890",
                "--limit", "10",
                "--no-remove-in-chatgpt",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()

    @patch("sora.cleanup_trash", new_callable=AsyncMock)
    def test_sora_cleanup_trash(self, mock_cleanup, mock_env_vars):
        """Should call sora.cleanup_trash."""
        result = runner.invoke(app, ["sora-cleanup-trash"])
        assert result.exit_code == 0
        mock_cleanup.assert_called_once()

    @patch("sora.cleanup_tasks", new_callable=AsyncMock)
    def test_sora_cleanup_tasks(self, mock_cleanup, mock_env_vars):
        """Should call sora.cleanup_tasks."""
        result = runner.invoke(app, ["sora-cleanup-tasks"])
        assert result.exit_code == 0
        mock_cleanup.assert_called_once()

    def test_clean_output_path(self, tmp_output_dir, monkeypatch):
        """Should clean output path."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))

        # Create test files
        (tmp_output_dir / "test.txt").write_text("test")
        (tmp_output_dir / ".gitkeep").write_text("keep")

        result = runner.invoke(app, ["clean-output-path"])
        assert result.exit_code == 0
        assert "Cleaning output path" in result.stdout
        assert "Output path cleaned" in result.stdout

        # Verify files cleaned
        assert not (tmp_output_dir / "test.txt").exists()
        assert (tmp_output_dir / ".gitkeep").exists()


class TestCLIEnvValidation:
    """Tests for environment variable validation."""

    def test_missing_env_vars_sora(self):
        """Should fail if required env vars missing for Sora."""
        with patch("util.validate_env_vars") as mock_validate:
            mock_validate.side_effect = ValueError("Missing NOTION_API_KEY")
            result = runner.invoke(
                app,
                ["sora-upload-to-notion", "--db-id", "test_db_12345678901234567890"],
            )
            assert result.exit_code != 0
            assert "Missing" in result.stdout or result.exception is not None

    def test_missing_env_vars_chatgpt(self):
        """Should fail if required env vars missing for ChatGPT."""
        with patch("util.validate_env_vars") as mock_validate:
            mock_validate.side_effect = ValueError("Missing CHATGPT_COOKIE")
            result = runner.invoke(
                app,
                [
                    "chatgpt-upload-to-notion",
                    "--db-id", "test_db_12345678901234567890",
                ],
            )
            assert result.exit_code != 0


class TestCLIDefaults:
    """Tests for CLI default values."""

    def test_sora_default_image_folder(self):
        """Should use default image folder."""
        result = runner.invoke(app, ["sora-upload-to-notion", "--help"])
        assert result.exit_code == 0
        assert "[default: sora_images]" in result.stdout

    def test_chatgpt_default_image_folder(self):
        """Should use default image folder."""
        result = runner.invoke(app, ["chatgpt-upload-to-notion", "--help"])
        assert result.exit_code == 0
        # Default is shown split across lines in rich formatting
        assert "chatgpt_images" in result.stdout

    def test_chatgpt_default_limit(self):
        """Should use default limit."""
        result = runner.invoke(app, ["chatgpt-upload-to-notion", "--help"])
        assert result.exit_code == 0
        assert "[default: 100]" in result.stdout
