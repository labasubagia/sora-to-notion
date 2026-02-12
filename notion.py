import asyncio
import os

import aiohttp
from dotenv import dotenv_values
from tqdm.asyncio import tqdm

from util import MAX_CONCURRENT_REQUESTS, get_output_path, retry_http

config = dotenv_values()

BASE_URL = "https://api.notion.com"

DB_ID = config.get("NOTION_DATABASE_ID")

headers = {
    "Authorization": f"Bearer {config.get('NOTION_API_KEY')}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}

# Cache for database data sources
_db_data_sources_cache = {}
_db_page_cache = set()


@retry_http()
async def get_db_data_sources(session, db_id: str):
    if db_id in _db_data_sources_cache:
        return _db_data_sources_cache[db_id]

    async with session.get(
        f"{BASE_URL}/v1/databases/{db_id}", headers=headers
    ) as response:
        response.raise_for_status()
        data = await response.json()
        data_sources = data.get("data_sources", [])
        _db_data_sources_cache[db_id] = data_sources
        return data_sources


@retry_http()
async def query_data_source(session, data_source_id: str, query: str):
    async with session.post(
        f"{BASE_URL}/v1/data_sources/{data_source_id}/query",
        headers=headers,
        json={
            "filter": {"and": [{"property": "Name", "rich_text": {"equals": query}}]}
        },
    ) as response:
        response.raise_for_status()
        return await response.json()


async def is_page_exists_in_db(
    session,
    db_id: str,
    query: str,
) -> bool:
    if query in _db_page_cache:
        return True
    data_sources = await get_db_data_sources(session, db_id)
    for data_source in data_sources:
        data = await query_data_source(session, data_source["id"], query)
        for page in data.get("results", []):
            name = page["properties"]["Name"]["title"][0]["text"]["content"]
            if name == query:
                _db_page_cache.add(query)
                return True
    return False


@retry_http()
async def create_upload_img(session, file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

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


@retry_http()
async def send_upload_img(session, file_upload_id: str, file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

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


@retry_http()
async def add_page_to_db(
    session, db_id, file_path, prompt, model="Sora", face="_original_"
):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    create_upload_res = await create_upload_img(session, file_path)
    send_upload_res = await send_upload_img(session, create_upload_res["id"], file_path)

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


async def upload_all_images_to_notion(generations, db_id, image_folder):
    total = len(generations)
    pbar = tqdm(total=total, desc="Uploading to Notion")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:

        async def upload(generation_id, prompt):
            async with semaphore:
                file_name = f"{generation_id}.png"
                file_path = get_output_path(os.path.join(image_folder, file_name))
                if not os.path.exists(file_path):
                    pbar.write(f"⚠️  {file_name} not found, skipped")
                    pbar.update(1)
                    return

                try:
                    if await is_page_exists_in_db(session, db_id, file_name):
                        pbar.write(f"⏭️  {file_name} skipped, already exists")
                    else:
                        await add_page_to_db(
                            session, db_id, file_path, prompt, model="Sora"
                        )
                        pbar.write(f"✅ {file_name} uploaded")
                except Exception as e:
                    pbar.write(f"❌ {file_name} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[upload(row["id"], row["prompt"]) for row in generations])

    pbar.close()
    print()  # Add spacing after progress bar
