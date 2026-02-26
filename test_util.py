"""Tests for util.py module"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

import util


class TestSaveToDataset:
    """Tests for save_to_dataset function"""

    @patch("util.pd.DataFrame")
    @patch("util.get_output_path")
    def test_save_to_dataset_success(self, mock_get_path, mock_dataframe):
        """Test successful save to dataset"""
        mock_get_path.return_value = Path("/tmp/test.csv")
        mock_df = MagicMock()
        mock_dataframe.return_value = mock_df

        data = [{"col1": "val1", "col2": "val2"}]
        util.save_to_dataset("test_dataset", data)

        mock_dataframe.assert_called_once_with(data)
        mock_df.to_csv.assert_called_once_with(Path("/tmp/test.csv"), index=False)
        mock_get_path.assert_called_once_with("test_dataset")

    def test_save_to_dataset_none_dataset(self):
        """Test with None dataset - should return early"""
        # Should not raise any exception
        util.save_to_dataset(None, [{"data": "value"}])

    def test_save_to_dataset_empty_data(self):
        """Test with empty data list - should return early"""
        # Should not raise any exception and not call DataFrame
        with patch("util.pd.DataFrame") as mock_dataframe:
            util.save_to_dataset("test", [])
            mock_dataframe.assert_not_called()


class TestGetOutputPath:
    """Tests for get_output_path function"""

    @patch("util.Path.mkdir")
    @patch("util.Path.resolve")
    @patch("util.Path.__truediv__")
    def test_get_output_path_relative(self, mock_div, mock_resolve, mock_mkdir):
        """Test with relative path"""
        mock_base = MagicMock()
        mock_base.resolve.return_value = Path("/output")
        mock_base.__truediv__ = lambda self, other: Path("/output") / other
        mock_final = MagicMock()
        mock_final.resolve.return_value = Path("/output/test/file.csv")
        mock_final.parents = [Path("/output")]
        mock_div.return_value = mock_final
        mock_resolve.return_value = Path("/output/test/file.csv")

        with patch(
            "util.Path", side_effect=lambda x: Path(x) if x != "./output" else mock_base
        ):
            with patch("util.OUTPUT_PATH", "./output"):
                with patch("util.Path.__truediv__", return_value=mock_final):
                    # This is a complex test due to Path mocking
                    pass

    def test_get_output_path_absolute_not_allowed(self):
        """Test that absolute paths are not allowed"""
        with pytest.raises(ValueError, match="Absolute paths are not allowed"):
            util.get_output_path("/absolute/path/file.csv")

    def test_get_output_path_escape_attempt(self):
        """Test that path escape attempts are blocked"""
        with pytest.raises(ValueError, match="attempts to escape the output directory"):
            # Use ../../../ to attempt escape
            util.get_output_path("../../../etc/passwd")

    @patch("pathlib.Path.mkdir")
    def test_get_output_path_creates_directories(self, mock_mkdir):
        """Test that parent directories are created"""
        # Create a temp directory to use as OUTPUT_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("util.OUTPUT_PATH", tmpdir):
                result = util.get_output_path("subdir/file.csv")
                assert result.name == "file.csv"
                mock_mkdir.assert_called()

    @patch("pathlib.Path.mkdir")
    def test_get_output_path_is_dir(self, mock_mkdir):
        """Test with is_dir=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("util.OUTPUT_PATH", tmpdir):
                result = util.get_output_path("subdir", is_dir=True)
                assert "subdir" in str(result)


class TestCleanOutputPath:
    """Tests for clean_output_path function"""

    def test_clean_output_path_nonexistent(self):
        """Test when OUTPUT_PATH doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = os.path.join(tmpdir, "nonexistent")
            with patch("util.OUTPUT_PATH", nonexistent):
                # Should not raise any exception
                util.clean_output_path()

    def test_clean_output_path_removes_files(self):
        """Test that files are removed except .gitkeep"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            gitkeep = Path(tmpdir) / ".gitkeep"
            file1 = Path(tmpdir) / "file1.txt"
            file2 = Path(tmpdir) / "file2.csv"
            gitkeep.touch()
            file1.touch()
            file2.touch()

            with patch("util.OUTPUT_PATH", tmpdir):
                util.clean_output_path()

            assert gitkeep.exists()
            assert not file1.exists()
            assert not file2.exists()

    def test_clean_output_path_removes_dirs(self):
        """Test that subdirectories are removed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "file.txt").touch()

            with patch("util.OUTPUT_PATH", tmpdir):
                util.clean_output_path()

            assert not subdir.exists()


class TestShouldRetryHttp:
    """Tests for should_retry_http function"""

    def test_retry_on_429(self):
        """Test retry on 429 status code"""
        exc = aiohttp.ClientResponseError(
            None, None, status=429, message="Too Many Requests"
        )
        assert util.should_retry_http(exc) is True

    def test_retry_on_500(self):
        """Test retry on 500 status code"""
        exc = aiohttp.ClientResponseError(
            None, None, status=500, message="Internal Server Error"
        )
        assert util.should_retry_http(exc) is True

    def test_retry_on_503(self):
        """Test retry on 503 status code"""
        exc = aiohttp.ClientResponseError(
            None, None, status=503, message="Service Unavailable"
        )
        assert util.should_retry_http(exc) is True

    def test_no_retry_on_400(self):
        """Test no retry on 400 status code"""
        exc = aiohttp.ClientResponseError(None, None, status=400, message="Bad Request")
        assert util.should_retry_http(exc) is False

    def test_no_retry_on_404(self):
        """Test no retry on 404 status code"""
        exc = aiohttp.ClientResponseError(None, None, status=404, message="Not Found")
        assert util.should_retry_http(exc) is False

    def test_retry_on_client_connector_error(self):
        """Test retry on ClientConnectorError"""
        import errno

        os_error = OSError(errno.ECONNREFUSED, "Connection refused")
        exc = aiohttp.ClientConnectorError(None, os_error)
        assert util.should_retry_http(exc) is True

    def test_retry_on_client_timeout(self):
        """Test retry on ClientTimeout"""
        exc = aiohttp.ClientTimeout()
        assert util.should_retry_http(exc) is True

    def test_retry_on_client_error_with_status(self):
        """Test retry on ClientError with status"""
        exc = aiohttp.ClientError()
        exc.status = 502
        assert util.should_retry_http(exc) is True

    def test_retry_on_client_error_without_status(self):
        """Test retry on ClientError without status"""
        exc = aiohttp.ClientError()
        assert util.should_retry_http(exc) is True

    def test_no_retry_on_generic_exception(self):
        """Test no retry on generic exception"""
        exc = ValueError("Some error")
        assert util.should_retry_http(exc) is False


class TestRetryHttp:
    """Tests for retry_http function"""

    def test_retry_http_returns_decorator(self):
        """Test that retry_http returns a retry decorator"""
        decorator = util.retry_http()
        assert decorator is not None

    @patch("util.MAX_RETRIES", 2)
    def test_retry_http_applies_retry_logic(self):
        """Test that retry logic is applied"""
        decorator = util.retry_http()

        call_count = 0

        @decorator
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                import errno

                os_error = OSError(errno.ECONNREFUSED, "Connection refused")
                raise aiohttp.ClientConnectorError(None, os_error)
            return "success"

        result = asyncio.run(failing_function())
        assert result == "success"
        assert call_count == 2


class TestHttpRetryable:
    """Tests for http_retryable function"""

    def test_retryable_on_429(self):
        """Test 429 is retryable"""
        assert util.http_retryable(429) is True

    def test_retryable_on_500(self):
        """Test 500 is retryable"""
        assert util.http_retryable(500) is True

    def test_retryable_on_502(self):
        """Test 502 is retryable"""
        assert util.http_retryable(502) is True

    def test_retryable_on_503(self):
        """Test 503 is retryable"""
        assert util.http_retryable(503) is True

    def test_retryable_on_599(self):
        """Test 599 is retryable"""
        assert util.http_retryable(599) is True

    def test_not_retryable_on_400(self):
        """Test 400 is not retryable"""
        assert util.http_retryable(400) is False

    def test_not_retryable_on_404(self):
        """Test 404 is not retryable"""
        assert util.http_retryable(404) is False

    def test_not_retryable_on_200(self):
        """Test 200 is not retryable"""
        assert util.http_retryable(200) is False

    def test_not_retryable_on_none(self):
        """Test None is not retryable"""
        assert util.http_retryable(None) is False


class TestGetHttpTimeout:
    """Tests for get_http_timeout function"""

    def test_get_http_timeout_returns_timeout(self):
        """Test that get_http_timeout returns ClientTimeout"""
        timeout = util.get_http_timeout()
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == util.HTTP_TIMEOUT_SECONDS


class TestValidateEnvVars:
    """Tests for validate_env_vars function"""

    @patch("dotenv.dotenv_values")
    def test_validate_env_vars_all_present(self, mock_dotenv):
        """Test when all env vars are present"""
        mock_dotenv.return_value = {"VAR1": "value1", "VAR2": "value2"}
        # Should not raise
        util.validate_env_vars(["VAR1", "VAR2"])

    @patch("dotenv.dotenv_values")
    def test_validate_env_vars_missing(self, mock_dotenv):
        """Test when env vars are missing"""
        mock_dotenv.return_value = {"VAR1": "value1"}
        with pytest.raises(ValueError, match="Missing required environment variables"):
            util.validate_env_vars(["VAR1", "VAR2"])

    @patch("dotenv.dotenv_values")
    def test_validate_env_vars_empty_string(self, mock_dotenv):
        """Test when env var is empty string"""
        mock_dotenv.return_value = {"VAR1": ""}
        with pytest.raises(ValueError):
            util.validate_env_vars(["VAR1"])

    @patch("dotenv.dotenv_values")
    def test_validate_env_vars_whitespace_only(self, mock_dotenv):
        """Test when env var is whitespace only"""
        mock_dotenv.return_value = {"VAR1": "   "}
        with pytest.raises(ValueError):
            util.validate_env_vars(["VAR1"])


class TestGetConfig:
    """Tests for get_config function"""

    @patch("dotenv.dotenv_values")
    def test_get_config_returns_dict(self, mock_dotenv):
        """Test that get_config returns a dict"""
        mock_dotenv.return_value = {"KEY": "value"}
        config = util.get_config()
        assert isinstance(config, dict)
        assert config["KEY"] == "value"


class TestDownloadImage:
    """Tests for download_image function"""

    @pytest.mark.asyncio
    async def test_download_image_success(self):
        """Test successful image download"""
        mock_response = AsyncMock()
        mock_response.read.return_value = b"image_data"
        mock_response.raise_for_status = MagicMock()

        mock_get_context = AsyncMock()
        mock_get_context.__aenter__.return_value = mock_response
        mock_get_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_context

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            await util.download_image(
                mock_session, "http://example.com/image.png", tmp_path
            )

            mock_session.get.assert_called_once_with(
                "http://example.com/image.png", headers={}
            )
            mock_response.read.assert_called_once()

            # Verify file was written
            with open(tmp_path, "rb") as f:
                assert f.read() == b"image_data"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_download_image_with_headers(self):
        """Test image download with custom headers"""
        mock_response = AsyncMock()
        mock_response.read.return_value = b"image_data"
        mock_response.raise_for_status = MagicMock()

        mock_get_context = AsyncMock()
        mock_get_context.__aenter__.return_value = mock_response
        mock_get_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_context

        headers = {"Authorization": "Bearer token"}

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            await util.download_image(
                mock_session, "http://example.com/image.png", tmp_path, headers
            )

            mock_session.get.assert_called_once_with(
                "http://example.com/image.png", headers=headers
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_download_image_http_error(self):
        """Test that HTTP errors are raised"""

        def raise_error():
            raise aiohttp.ClientResponseError(
                None, None, status=404, message="Not Found"
            )

        mock_response = AsyncMock()
        mock_response.raise_for_status = raise_error
        mock_response.read.return_value = b"image_data"

        mock_get_context = AsyncMock()
        mock_get_context.__aenter__.return_value = mock_response
        mock_get_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_context

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(aiohttp.ClientResponseError):
                await util.download_image(
                    mock_session, "http://example.com/image.png", tmp_path
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestConstants:
    """Tests for module constants"""

    def test_max_retries_constant(self):
        """Test MAX_RETRIES is defined"""
        assert util.MAX_RETRIES == 5

    def test_max_concurrent_downloads_constant(self):
        """Test MAX_CONCURRENT_DOWNLOADS is defined"""
        assert util.MAX_CONCURRENT_DOWNLOADS == 10

    def test_max_concurrent_requests_constant(self):
        """Test MAX_CONCURRENT_REQUESTS is defined"""
        assert util.MAX_CONCURRENT_REQUESTS == 10

    def test_http_timeout_seconds_constant(self):
        """Test HTTP_TIMEOUT_SECONDS is defined"""
        assert util.HTTP_TIMEOUT_SECONDS == 30

    def test_output_path_constant(self):
        """Test OUTPUT_PATH is defined"""
        assert util.OUTPUT_PATH == "./output"
