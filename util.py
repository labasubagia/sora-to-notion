import asyncio
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import aiohttp
import pandas as pd
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

MAX_RETRIES = 5
MAX_CONCURRENT_DOWNLOADS = 10
MAX_CONCURRENT_REQUESTS = 10
HTTP_TIMEOUT_SECONDS = 30

OUTPUT_PATH = "./output"

logger = logging.getLogger(__name__)


def save_to_dataset(dataset: str, data: Sequence[dict] | Sequence[BaseModel]):
    if dataset is None:
        return
    if len(data) == 0:
        print("No generations to save to dataset.")
        return

    # Convert Pydantic models to dicts if needed
    dict_data: list[dict]
    if data and isinstance(data[0], BaseModel):
        dict_data = [item.model_dump() for item in data]  # type: ignore[union-attr]
    else:
        dict_data = list(data)  # type: ignore[arg-type]

    df = pd.DataFrame(dict_data)
    file_path = get_output_path(dataset)
    df.to_csv(file_path, index=False)
    print(f"✅ Saved dataset to {file_path}\n")


def get_output_path(input_path_str: str, is_dir=False) -> Path:
    input_path = Path(input_path_str)

    if input_path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {input_path_str}")

    base_dir: Path = Path(OUTPUT_PATH).resolve()
    final_path: Path = (base_dir / input_path).resolve()
    if base_dir not in final_path.parents:
        raise ValueError("Path attempts to escape the output directory!")

    if is_dir:
        final_path.mkdir(parents=True, exist_ok=True)
    else:
        final_path.parent.mkdir(parents=True, exist_ok=True)

    return final_path


def clean_output_path() -> None:
    # Except .gitkeep, force remove all files and folders in OUTPUT_PATH
    base_dir: Path = Path(OUTPUT_PATH).resolve()
    if not base_dir.exists():
        return
    for item in base_dir.iterdir():
        if item.is_file() and item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def should_retry_http(exception: Exception) -> bool:
    """Determine if an HTTP exception should be retried"""
    if isinstance(exception, aiohttp.ClientResponseError):
        return http_retryable(exception.status)
    if isinstance(exception, aiohttp.ClientError):
        status = getattr(exception, "status", None)
        if status is not None:
            return http_retryable(status)
        return True
    if isinstance(
        exception,
        aiohttp.ClientConnectorError | aiohttp.ClientTimeout | asyncio.TimeoutError,
    ):
        return True
    return False


def retry_http():
    """Reusable retry decorator for async HTTP requests"""
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(should_retry_http),
        reraise=True,
    )


def http_retryable(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code == 429 or status_code >= 500


def get_http_timeout() -> aiohttp.ClientTimeout:
    """Get default HTTP timeout configuration"""
    return aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)


def validate_env_vars(required_vars: list[str]) -> None:
    """Validate that required environment variables are set"""
    from dotenv import dotenv_values

    config = dotenv_values()
    missing = [var for var in required_vars if not (config.get(var) or "").strip()]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def get_config() -> dict[str, str | None]:
    """Load configuration from .env file"""
    from dotenv import dotenv_values

    return dotenv_values()


async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    file_path: str,
    headers: dict[str, str] | None = None,
) -> None:
    """Download an image from URL to file path"""
    async with session.get(url, headers=headers or {}) as response:
        response.raise_for_status()
        content = await response.read()
        await asyncio.to_thread(lambda: open(file_path, "wb").write(content))
