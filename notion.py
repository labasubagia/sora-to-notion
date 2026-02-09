import asyncio
import os

import aiohttp
import pandas as pd
from dotenv import dotenv_values

from util import MAX_RETRIES, get_output_path, msg_prefix_progress

config = dotenv_values()

BASE_URL = "https://api.notion.com"

API_KEY = config.get("NOTION_API_KEY")
DB_ID = config.get("NOTION_DATABASE_ID")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}

# Cache for database data sources
_db_data_sources_cache = {}
_db_page_cache = set()


async def get_db_data_sources(db_id: str):
    if db_id in _db_data_sources_cache:
        return _db_data_sources_cache[db_id]

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{BASE_URL}/v1/databases/{db_id}", headers=headers
        ) as response:
            response.raise_for_status()
            data = await response.json()
            data_sources = data.get("data_sources", [])
            _db_data_sources_cache[db_id] = data_sources
            return data_sources


async def is_page_exists_in_db(
    db_id: str,
    query: str,
) -> bool:
    if query in _db_page_cache:
        return True
    data_sources = await get_db_data_sources(db_id)
    async with aiohttp.ClientSession() as session:
        for data_source in data_sources:
            async with session.post(
                f"{BASE_URL}/v1/data_sources/{data_source['id']}/query",
                headers=headers,
                json={
                    "filter": {
                        "and": [{"property": "Name", "rich_text": {"equals": query}}]
                    }
                },
            ) as response:
                response.raise_for_status()
                data = await response.json()
                for page in data.get("results", []):
                    name = page["properties"]["Name"]["title"][0]["text"]["content"]
                    if name == query:
                        _db_page_cache.add(query)
                        return True
    return False


async def create_upload_img(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/file_uploads",
            headers=headers,
            json={
                "mode": "single_part",
                "filename": file_name,
                "content_type": "image/png",
            },
        ) as response:
            response.raise_for_status()
            return await response.json()


async def send_upload_img(file_upload_id: str, file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    async with aiohttp.ClientSession() as session:
        with open(file_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=file_name, content_type="image/png")
            async with session.post(
                f"{BASE_URL}/v1/file_uploads/{file_upload_id}/send",
                headers={
                    "Authorization": headers["Authorization"],
                    "Notion-Version": headers["Notion-Version"],
                },
                data=data,
            ) as response:
                response.raise_for_status()
                return await response.json()


async def add_page_to_db(db_id, file_path, prompt, model="Sora", face="_original_"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    create_upload_res = await create_upload_img(file_path)
    send_upload_res = await send_upload_img(create_upload_res["id"], file_path)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/pages",
            headers=headers,
            json={
                "parent": {"database_id": db_id},
                "properties": {
                    "Name": {"title": [{"text": {"content": file_name}}]},
                    "Image": {
                        "files": [
                            {
                                "type": "file_upload",
                                "file_upload": {"id": send_upload_res["id"]},
                            }
                        ]
                    },
                    "Prompt": {"rich_text": [{"text": {"content": prompt or "N/A"}}]},
                    "Model": {"select": {"name": model}},
                    "Face": {"select": {"name": face}},
                },
            },
        ) as response:
            response.raise_for_status()
            _db_page_cache.add(file_name)
            return await response.json()


async def upload_all_images_to_notion(dataset, db_id, image_folder):
    df = pd.read_csv(get_output_path(dataset))
    processed = 0
    total = len(df)

    async def upload(generation_id, prompt):
        nonlocal processed
        file_name = f"{generation_id}.png"
        file_path = get_output_path(os.path.join(image_folder, file_name))
        if not os.path.exists(file_path):
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {file_path} not found, skipped."
            )
            return

        for _ in range(MAX_RETRIES):
            try:
                if await is_page_exists_in_db(db_id, file_name):
                    processed += 1
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {file_path} skipped, already exists",
                    )
                else:
                    await add_page_to_db(db_id, file_path, prompt, model="Sora")
                    processed += 1
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {file_path} uploaded"
                    )
                return
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] {file_path} failed: {e}"
                )
        else:
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {file_path} failed after {MAX_RETRIES} retries"
            )

    await asyncio.gather(
        *[upload(row["id"], row["prompt"]) for (_, row) in df.iterrows()]
    )
