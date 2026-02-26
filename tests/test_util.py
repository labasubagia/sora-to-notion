"""
Unit tests for util.py - pure functions, no external dependencies.
"""
import os
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest

from util import (
    MAX_RETRIES,
    clean_output_path,
    get_http_timeout,
    get_output_path,
    http_retryable,
    save_to_dataset,
    should_retry_http,
    validate_env_vars,
)


class TestGetOutputPath:
    """Tests for get_output_path function."""

    def test_relative_path_allowed(self, tmp_output_dir, monkeypatch):
        """Relative paths within output directory should work."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("images/test.png")
        assert path.name == "test.png"
        assert "images" in str(path)

    def test_absolute_path_rejected(self):
        """Absolute paths should be rejected."""
        with pytest.raises(ValueError, match="Absolute paths are not allowed"):
            get_output_path("/etc/passwd")

    def test_path_traversal_rejected(self, tmp_output_dir, monkeypatch):
        """Path traversal attempts should be rejected."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        with pytest.raises(ValueError, match="attempts to escape"):
            get_output_path("../../etc/passwd")

    def test_creates_parent_directories(self, tmp_output_dir, monkeypatch):
        """Parent directories should be created automatically."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("nested/deep/path/file.png")
        assert path.parent.exists()

    def test_is_dir_flag(self, tmp_output_dir, monkeypatch):
        """is_dir=True should create directory."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("test_dir", is_dir=True)
        assert path.is_dir()


class TestHttpRetryable:
    """Tests for http_retryable function."""

    def test_429_retryable(self):
        """Rate limit should be retryable."""
        assert http_retryable(429) is True

    def test_5xx_retryable(self):
        """Server errors should be retryable."""
        assert http_retryable(500) is True
        assert http_retryable(503) is True
        assert http_retryable(504) is True

    def test_4xx_not_retryable(self):
        """Client errors should not be retryable."""
        assert http_retryable(400) is False
        assert http_retryable(404) is False

    def test_2xx_not_retryable(self):
        """Success should not be retryable."""
        assert http_retryable(200) is False

    def test_none_not_retryable(self):
        """None status code should not be retryable."""
        assert http_retryable(None) is False


class TestShouldRetryHttp:
    """Tests for should_retry_http function."""

    def test_client_response_error_429(self):
        """429 status should retry."""
        err = aiohttp.ClientResponseError(None, None, status=429)
        assert should_retry_http(err) is True

    def test_client_response_error_503(self):
        """503 status should retry."""
        err = aiohttp.ClientResponseError(None, None, status=503)
        assert should_retry_http(err) is True

    def test_client_response_error_404(self):
        """404 status should not retry."""
        err = aiohttp.ClientResponseError(None, None, status=404)
        assert should_retry_http(err) is False

    def test_client_connector_error(self):
        """Connector errors should retry."""
        import errno
        os_error = OSError(errno.ECONNREFUSED, "Connection refused")
        err = aiohttp.ClientConnectorError(None, os_error)
        assert should_retry_http(err) is True

    def test_client_timeout(self):
        """Timeout errors should retry."""
        err = aiohttp.ClientTimeout()
        assert should_retry_http(err) is True

    def test_other_exception(self):
        """Other exceptions should not retry by default."""
        err = ValueError("test")
        assert should_retry_http(err) is False


class TestGetHttpTimeout:
    """Tests for get_http_timeout function."""

    def test_returns_client_timeout(self):
        """Should return aiohttp.ClientTimeout."""
        timeout = get_http_timeout()
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == 30


class TestValidateEnvVars:
    """Tests for validate_env_vars function."""

    def test_missing_vars_raises_error(self):
        """Missing variables should raise ValueError."""
        with patch("dotenv.dotenv_values", return_value={"EXISTING_VAR": "value"}):
            with pytest.raises(ValueError, match="Missing required"):
                validate_env_vars(["MISSING_VAR"])

    def test_all_vars_present(self):
        """All variables present should pass."""
        with patch("dotenv.dotenv_values", return_value={"VAR1": "val1", "VAR2": "val2"}):
            # Should not raise
            validate_env_vars(["VAR1", "VAR2"])

    def test_empty_var_treated_as_missing(self):
        """Empty variables should be treated as missing."""
        with patch("dotenv.dotenv_values", return_value={"VAR1": "", "VAR2": "  "}):
            with pytest.raises(ValueError, match="Missing required"):
                validate_env_vars(["VAR1", "VAR2"])


class TestSaveToDataset:
    """Tests for save_to_dataset function."""

    def test_none_dataset_skips(self, capsys):
        """None dataset should skip silently."""
        save_to_dataset(None, [{"id": "test"}])
        captured = capsys.readouterr()
        assert "Saved dataset" not in captured.out

    def test_empty_data_skips(self, capsys):
        """Empty data should skip."""
        save_to_dataset("test.csv", [])
        captured = capsys.readouterr()
        assert "No generations to save" in captured.out

    def test_saves_csv(self, tmp_output_dir, monkeypatch):
        """Should save to CSV file."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        data = [{"id": "test123", "prompt": "test prompt"}]
        save_to_dataset("test_dataset.csv", data)
        
        csv_path = tmp_output_dir / "test_dataset.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "id" in content
        assert "test123" in content


class TestCleanOutputPath:
    """Tests for clean_output_path function."""

    def test_removes_files(self, tmp_output_dir, monkeypatch):
        """Should remove all files except .gitkeep."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        
        # Create test files
        (tmp_output_dir / "test.txt").write_text("test")
        (tmp_output_dir / ".gitkeep").write_text("keep")
        
        clean_output_path()
        
        assert not (tmp_output_dir / "test.txt").exists()
        assert (tmp_output_dir / ".gitkeep").exists()

    def test_removes_directories(self, tmp_output_dir, monkeypatch):
        """Should remove directories."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        
        # Create test directory
        test_dir = tmp_output_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")
        
        clean_output_path()
        
        assert not test_dir.exists()

    def test_nonexistent_path(self, tmp_path, monkeypatch):
        """Should handle non-existent output path."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "nonexistent"))
        # Should not raise
        clean_output_path()
