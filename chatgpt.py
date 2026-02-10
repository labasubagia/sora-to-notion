import asyncio
import os
from datetime import datetime, timezone

import aiohttp
import pandas as pd
from dotenv import dotenv_values

from notion import (
    is_page_exists_in_db,
    upload_all_images_to_notion,
)
from util import MAX_RETRIES, get_output_path, msg_prefix_progress

config = dotenv_values()

BASE_URL = "https://chatgpt.com/backend-api"

API_KEY = config.get("CHATGPT_AUTHORIZATION_TOKEN")
USER_AGENT = config.get("CHATGPT_USER_AGENT")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "User-Agent": USER_AGENT,
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


async def save_dataset_of_image_generations(dataset: str, limit=100):
    """
    Save dataset of image generations to CSV file asynchronously.
    """
    generation_list = []
    data = await get_image_generations(limit=limit)
    total = len(data.get("items", []))
    processed = 0
    print("Fetched", total, "image generations.")

    async def fetch_generation_details(img_gen):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                detail = await get_conversation_details(img_gen["conversation_id"])
                prompt = get_prompt_from_image_node_in_conversation(
                    detail, img_gen["message_id"], img_gen["asset_pointer"]
                )
                created_at = datetime.fromtimestamp(
                    img_gen["created_at"], tz=timezone.utc
                ).isoformat(timespec="microseconds")
                processed += 1
                print(
                    f"[{msg_prefix_progress(processed, total)}] info img ID {img_gen['id']} added."
                )
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
                print(
                    f"[{msg_prefix_progress(processed, total)}] info img ID {img_gen['id']} fetch error: {e}, retrying..."
                )
        else:
            print(
                f"[{msg_prefix_progress(processed, total)}] info img ID {img_gen['id']} failed after {MAX_RETRIES} retries"
            )
            raise Exception("Failed to fetch generation details after maximum retries")

    generation_list = await asyncio.gather(
        *[fetch_generation_details(img_gen) for img_gen in data.get("items", [])]
    )

    if len(generation_list) == 0:
        df = pd.DataFrame(
            columns=[
                "created_at",
                "id",
                "conversation_id",
                "message_id",
                "url",
                "prompt",
            ]
        )
    else:
        df = pd.DataFrame(generation_list)
        df = df.sort_values(by="created_at", ascending=True).reset_index(drop=True)

    df.to_csv(get_output_path(dataset), index=False)


async def download_all_images(
    dataset,
    download_folder,
):
    """
    Download all images from the dataset CSV file asynchronously.
    """
    df = pd.read_csv(get_output_path(dataset))

    total = len(df)
    processed = 0

    async def download_image(session, row):
        nonlocal processed
        file_name = f"{row.id}.png"
        file_path = get_output_path(os.path.join(download_folder, file_name))

        if os.path.exists(file_path):
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {file_path} already exists, skipped."
            )
            return

        for _ in range(MAX_RETRIES):
            try:
                async with session.get(row.url, headers=headers) as response:
                    response.raise_for_status()
                    with open(file_path, "wb") as f:
                        f.write(await response.read())
                        processed += 1
                        print(
                            f"[{msg_prefix_progress(processed, total)}] {file_path} downloaded."
                        )
                        return
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] {file_path} download error: {e}, retrying..."
                )
        else:
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {file_path} failed after {MAX_RETRIES} retries"
            )

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            *[download_image(session, row) for row in df.itertuples(index=False)]
        )


async def delete_conversation_of_image_generation_uploaded_to_notion(dataset, db_id):
    df = pd.read_csv(get_output_path(dataset))
    df = df.drop_duplicates(subset=["conversation_id"])

    total = len(df)
    processed = 0

    async def delete(row):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                file_name = f"{row.id}.png"

                exists = await is_page_exists_in_db(db_id, file_name)
                if not exists:
                    processed += 1
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {file_name} not found in Notion, skipped."
                    )
                    return

                await delete_conversation(row.conversation_id)
                processed += 1
                print(
                    f"[{msg_prefix_progress(processed, total)}] Conversation ID {row.conversation_id} deleted."
                )
                return
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] Conversation ID {row.conversation_id} delete error: {e}, retrying..."
                )
        else:
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] Conversation ID {row.conversation_id} failed after {MAX_RETRIES} retries"
            )

    await asyncio.gather(*[delete(row) for row in df.itertuples(index=False)])


async def upload_to_notion(
    dataset: str,
    image_folder: str,
    db_id,
    upload_to_notion=True,
    remove_in_chatgpt=False,
):
    print("📊 Saving dataset from image generations...")
    await save_dataset_of_image_generations(dataset=dataset)
    print("✅ Dataset saved.\n")
    print("🖼️  Downloading all images...")
    await download_all_images(
        dataset=dataset,
        download_folder=image_folder,
    )
    print("✅ All images downloaded.\n")
    if upload_to_notion:
        print("📤 Uploading all images to Notion...")
        await upload_all_images_to_notion(
            dataset=dataset,
            db_id=db_id,
            image_folder=image_folder,
        )
        print("✅ All images uploaded to Notion.\n")

    if remove_in_chatgpt:
        print("🗑️  Deleting conversations of uploaded image generations...")
        await delete_conversation_of_image_generation_uploaded_to_notion(
            dataset=dataset,
            db_id=db_id,
        )
        print("✅ Conversations deleted.\n")
