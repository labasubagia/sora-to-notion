import asyncio
import os
from base64 import b64decode
from datetime import datetime, timezone

import aiohttp
from dotenv import dotenv_values
from tqdm.asyncio import tqdm

from img import add_prompt_to_images
from notion import is_page_exists_in_db, upload_all_images_to_notion
from util import (
    MAX_CONCURRENT_DOWNLOADS,
    MAX_CONCURRENT_REQUESTS,
    get_output_path,
    retry_http,
    save_to_dataset,
)

config = dotenv_values()

BASE_URL = "https://chatgpt.com/backend-api"


def get_headers():
    headers = {
        "Authorization": f"Bearer {config.get('CHATGPT_AUTHORIZATION_TOKEN', '').strip()}",
        "User-Agent": config.get("CHATGPT_USER_AGENT", "").strip(),
        "Content-Type": "application/json",
    }

    cookie_base64 = config.get("CHATGPT_COOKIE_STRING_BASE64", "").strip()
    if cookie_base64:
        headers["Cookie"] = b64decode(cookie_base64).decode("utf-8").strip()

    return headers


headers = get_headers()


@retry_http()
async def get_conversations(
    session, offset=0, limit=100, is_archived=False, is_starred=False, order="updated"
):
    async with session.get(
        f"{BASE_URL}/conversations",
        headers=headers,
        params={
            "offset": offset,
            "limit": limit,
            "order": order,
            "is_archived": str(is_archived).lower(),
            "is_starred": str(is_starred).lower(),
        },
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data


@retry_http()
async def get_conversation_details(session, conversation_id: str):
    async with session.get(
        f"{BASE_URL}/conversation/{conversation_id}",
        headers=headers,
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data


@retry_http()
async def delete_conversation(session, conversation_id: str):
    async with session.patch(
        f"{BASE_URL}/conversation/{conversation_id}",
        headers=headers,
        json={"is_visible": False},
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data


@retry_http()
async def get_image_generations(session, limit=100):
    async with session.get(
        f"{BASE_URL}/my/recent/image_gen",
        headers=headers,
        params={"limit": limit},
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data


def get_conversation_mapping_key_by_asset_pointer(data, asset_pointer):
    for key, node in data.get("mapping", {}).items():
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and part.get("asset_pointer") == asset_pointer:
                return key
    return None


def get_prompt_from_image_node_in_conversation(data, start_node_id, asset_pointer):
    mapping = data["mapping"]
    current_id = start_node_id
    if current_id not in mapping:
        current_id = get_conversation_mapping_key_by_asset_pointer(data, asset_pointer)
        if current_id is None:
            return None

    while current_id:
        node = mapping[current_id]
        msg = node.get("message")

        if msg:
            author = msg.get("author", {}).get("role")
            if author == "user":
                inputs = msg.get("content", {}).get("parts", [])
                for part in inputs:
                    if isinstance(part, str):
                        return part
        current_id = node.get("parent")

    return None


async def fetch_image_generations(limit=100):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        data = await get_image_generations(session, limit=limit)

        total = len(data.get("items", []))
        pbar = tqdm(total=total, desc="Fetching generation details")

        async def fetch_generation_details(img_gen):
            async with semaphore:
                try:
                    detail = await get_conversation_details(
                        session, img_gen["conversation_id"]
                    )
                    prompt = get_prompt_from_image_node_in_conversation(
                        detail, img_gen["message_id"], img_gen["asset_pointer"]
                    )
                    created_at = datetime.fromtimestamp(
                        img_gen["created_at"], tz=timezone.utc
                    ).isoformat(timespec="microseconds")
                    pbar.write(f"✅ img ID {img_gen['id']}")
                    return {
                        "created_at": created_at,
                        "id": img_gen["id"],
                        "conversation_id": img_gen["conversation_id"],
                        "message_id": img_gen["message_id"],
                        "asset_pointer": img_gen["asset_pointer"],
                        "url": img_gen["url"],
                        "prompt": prompt,
                    }
                except Exception as e:
                    pbar.write(f"❌ img ID {img_gen['id']} failed: {e}")
                    return None
                finally:
                    pbar.update(1)

        results = await asyncio.gather(
            *[fetch_generation_details(img_gen) for img_gen in data.get("items", [])],
            return_exceptions=True,
        )

        pbar.close()
        print()

        # Filter out None and exceptions
        generations = [
            g for g in results if g is not None and not isinstance(g, Exception)
        ]
        generations = sorted(generations, key=lambda x: x["created_at"])
        return generations


@retry_http()
async def download_image(session, url, file_path):
    async with session.get(url, headers=headers) as response:
        response.raise_for_status()
        content = await response.read()
        await asyncio.to_thread(lambda: open(file_path, "wb").write(content))


async def download_all_images(generations, download_folder):
    total = len(generations)
    pbar = tqdm(total=total, desc="Downloading images")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async with aiohttp.ClientSession() as session:

        async def download(row):
            async with semaphore:
                file_name = f"{row['id']}.png"
                file_path = get_output_path(os.path.join(download_folder, file_name))

                if os.path.exists(file_path):
                    pbar.write(f"⏭️  {file_name} skipped, already exists")
                    pbar.update(1)
                    return

                try:
                    await download_image(session, row["url"], file_path)
                    pbar.write(f"✅ {file_name}")
                except Exception as e:
                    pbar.write(f"❌ {file_name} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[download(row) for row in generations])

    pbar.close()
    print()


async def delete_conversation_of_image_generation_uploaded_to_notion(
    generations, db_id
):
    generations = list({gen["conversation_id"]: gen for gen in generations}.values())
    total = len(generations)
    pbar = tqdm(total=total, desc="Deleting conversations")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:

        async def delete(row):
            async with semaphore:
                file_name = f"{row['id']}.png"
                conversation_id = row["conversation_id"]

                try:
                    exists = await is_page_exists_in_db(session, db_id, file_name)
                    if not exists:
                        pbar.write(f"⏭️  {file_name} not found in Notion, skipped")
                        pbar.update(1)
                        return

                    await delete_conversation(session, conversation_id)
                    pbar.write(f"✅ Conversation ID {conversation_id}")
                except Exception as e:
                    pbar.write(f"❌ Conversation ID {conversation_id} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[delete(row) for row in generations])

    pbar.close()
    print()


async def upload_to_notion(
    image_folder: str,
    db_id,
    upload_to_notion=True,
    remove_in_chatgpt=False,
    add_prompt_to_image=True,
    dataset: str = None,
    limit=100,
):
    generations = await fetch_image_generations(limit=limit)

    if dataset:
        save_to_dataset(dataset=dataset, data=generations)

    await download_all_images(generations=generations, download_folder=image_folder)

    if add_prompt_to_image:
        add_prompt_to_images(generations=generations, folder=image_folder)

    if upload_to_notion:
        await upload_all_images_to_notion(
            generations=generations, db_id=db_id, image_folder=image_folder
        )

    if remove_in_chatgpt:
        await delete_conversation_of_image_generation_uploaded_to_notion(
            generations=generations, db_id=db_id
        )
