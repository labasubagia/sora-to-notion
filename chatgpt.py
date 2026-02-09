from dotenv import dotenv_values
from datetime import datetime, timezone
import pandas as pd
import os
import asyncio
import aiohttp
from util import msg_prefix_progress
from notion import (
    upload_all_images_to_notion,
    is_page_exists_in_db,
    DB_ID as NOTION_DB_ID,
)

config = dotenv_values()

BASE_URL = "https://chatgpt.com/backend-api"

API_KEY = config.get("CHATGPT_AUTHORIZATION_TOKEN")
USER_AGENT = config.get("CHATGPT_USER_AGENT")

headers = {
    "authorization": f"Bearer {API_KEY}",
    "user-agent": USER_AGENT,
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

    df.to_csv(dataset, index=False)


async def download_all_images(
    dataset="chatgpt_generations.csv",
    download_folder="chatgpt_images",
):
    """
    Download all images from the dataset CSV file asynchronously.
    """
    df = pd.read_csv(os.path.join(dataset))

    total = len(df)
    processed = 0

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    async def download_image(session, row):
        nonlocal processed
        file_path = os.path.join(download_folder, f"{row['id']}.png")
        if not os.path.exists(file_path):
            try:
                async with session.get(row["url"], headers=headers) as response:
                    response.raise_for_status()
                    with open(file_path, "wb") as f:
                        f.write(await response.read())
                processed += 1
                print(
                    f"[{msg_prefix_progress(processed, total)}] {file_path} downloaded."
                )
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] {file_path} failed: {e}"
                )
        else:
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {file_path} already exists, skipped."
            )

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            *[download_image(session, row) for _, row in df.iterrows()]
        )


async def delete_conversation_of_image_generation_uploaded_to_notion(
    dataset="chatgpt_generations.csv", db_id=NOTION_DB_ID
):
    df = pd.read_csv(os.path.join(dataset))

    total = len(df)
    processed = 0

    async def delete_if_uploaded(row):
        nonlocal processed
        try:
            file_name = f"{row['id']}.png"
            exists = await is_page_exists_in_db(db_id, file_name)
            if not exists:
                processed += 1
                print(
                    f"[{msg_prefix_progress(processed, total)}] {file_name} not found in Notion, skipped."
                )
                return
            await delete_conversation(row["conversation_id"])
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] Conversation ID {row['conversation_id']} deleted."
            )
        except Exception as e:
            print(
                f"[{msg_prefix_progress(processed, total)}] Conversation ID {row['conversation_id']} failed: {e}"
            )

    await asyncio.gather(*[delete_if_uploaded(row) for _, row in df.iterrows()])


async def chatgpt_upload_to_notion(
    dataset: str, image_folder: str, db_id, upload_to_notion=True
):
    print("📊 Saving dataset from image generations...")
    await save_dataset_of_image_generations(dataset=dataset)
    print("✅ Dataset saved.\n")
    print("🖼️ Downloading all images...")
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
