import asyncio
import os
from datetime import datetime, timezone

import aiohttp
from dotenv import dotenv_values
from tqdm.asyncio import tqdm

from img import add_prompt_to_images
from notion import is_page_exists_in_db, upload_all_images_to_notion
from util import MAX_RETRIES, get_output_path, save_to_dataset

config = dotenv_values()

BASE_URL = "https://chatgpt.com/backend-api"


headers = {
    "Authorization": config.get("CHATGPT_AUTHORIZATION_TOKEN"),
    "User-Agent": config.get("CHATGPT_USER_AGENT"),
    "Cookie": config.get("CHATGPT_COOKIE_STRING"),
    "Content-Type": "application/json",
}


async def get_conversations(
    offset=0, limit=100, is_archived=False, is_starred=False, order="updated"
):
    async with aiohttp.ClientSession() as session:
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


async def get_conversation_details(conversation_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{BASE_URL}/conversation/{conversation_id}",
            headers=headers,
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data


async def delete_conversation(conversation_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.patch(
            f"{BASE_URL}/conversation/{conversation_id}",
            headers=headers,
            json={"is_visible": False},
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data


async def get_image_generations(limit=100):
    async with aiohttp.ClientSession() as session:
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
                return msg["content"]["parts"][0]
        current_id = node.get("parent")

    return None


async def fetch_image_generations(limit=100):

    generations = []
    data = await get_image_generations(limit=limit)
    total = len(data.get("items", []))
    pbar = tqdm(total=total, desc="Fetching generation details")

    async def fetch_generation_details(img_gen):
        for _ in range(MAX_RETRIES):
            try:
                detail = await get_conversation_details(img_gen["conversation_id"])
                prompt = get_prompt_from_image_node_in_conversation(
                    detail, img_gen["message_id"], img_gen["asset_pointer"]
                )
                created_at = datetime.fromtimestamp(
                    img_gen["created_at"], tz=timezone.utc
                ).isoformat(timespec="microseconds")
                pbar.write(f"✅ img ID {img_gen['id']}")
                pbar.update(1)
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
                pbar.write(f"⚠️  img ID {img_gen['id']} fetch error: {e}, retrying...")
        else:
            pbar.write(f"❌ img ID {img_gen['id']} failed after {MAX_RETRIES} retries")
            pbar.update(1)
            raise Exception("Failed to fetch generation details after maximum retries")

    generations = await asyncio.gather(
        *[fetch_generation_details(img_gen) for img_gen in data.get("items", [])]
    )
    pbar.close()
    print()

    generations = sorted(generations, key=lambda x: x["created_at"])
    return generations


async def download_all_images(generations, download_folder):
    total = len(generations)
    pbar = tqdm(total=total, desc="Downloading images")

    async def download_image(session, row):
        file_name = f"{row['id']}.png"
        file_path = get_output_path(os.path.join(download_folder, file_name))

        if os.path.exists(file_path):
            pbar.write(f"⏭️  {file_name} already exists, skipped")
            pbar.update(1)
            return

        for _ in range(MAX_RETRIES):
            try:
                async with session.get(row["url"], headers=headers) as response:
                    response.raise_for_status()
                    with open(file_path, "wb") as f:
                        f.write(await response.read())
                        pbar.write(f"✅ {file_name} downloaded")
                        pbar.update(1)
                        return
            except Exception as e:
                pbar.write(f"⚠️  {file_name} download error: {e}, retrying...")
        else:
            pbar.write(f"❌ {file_name} failed after {MAX_RETRIES} retries")
            pbar.update(1)

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[download_image(session, row) for row in generations])
    pbar.close()
    print()


async def delete_conversation_of_image_generation_uploaded_to_notion(
    generations, db_id
):
    generations = list({gen["conversation_id"]: gen for gen in generations}.values())
    total = len(generations)

    pbar = tqdm(total=total, desc="Deleting conversations")

    async def delete(row):
        file_name = f"{row['id']}.png"
        conversation_id = row["conversation_id"]
        for _ in range(MAX_RETRIES):
            try:
                exists = await is_page_exists_in_db(db_id, file_name)
                if not exists:
                    pbar.write(f"⏭️  {file_name} not found in Notion, skipped")
                    pbar.update(1)
                    return

                await delete_conversation(conversation_id)
                pbar.write(f"✅ Conversation ID {conversation_id}")
                pbar.update(1)
                return
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    pbar.write(
                        f"⏭️  Conversation ID {conversation_id} not found, skipped"
                    )
                    pbar.update(1)
                    return
                pbar.write(
                    f"⚠️  Conversation ID {conversation_id} delete HTTP error: {e.status} {e.message}, retrying..."
                )
            except Exception as e:
                pbar.write(
                    f"⚠️  Conversation ID {conversation_id} delete error: {e}, retrying..."
                )
        else:
            pbar.write(
                f"❌ Conversation ID {conversation_id} failed after {MAX_RETRIES} retries"
            )
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
    dataset: bool = None,  # save to dataset if provided
):
    generations = await fetch_image_generations()
    await download_all_images(generations=generations, download_folder=image_folder)

    if dataset:
        save_to_dataset(dataset=dataset, data=generations)

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
