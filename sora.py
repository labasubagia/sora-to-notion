import asyncio
import json
import os

import aiohttp
import pandas as pd
from dotenv import dotenv_values

from notion import is_page_exists_in_db, upload_all_images_to_notion
from util import MAX_RETRIES, get_output_path, msg_prefix_progress

config = dotenv_values()

BASE_URL = "https://sora.chatgpt.com/backend"
AUTHORIZATION_TOKEN = config.get("CHATGPT_AUTHORIZATION_TOKEN")
USER_AGENT = config.get("CHATGPT_USER_AGENT")

headers = {
    "Authorization": f"Bearer {AUTHORIZATION_TOKEN}",
    "User-Agent": USER_AGENT,
    "Content-Type": "application/json",
}


async def archive_generation(generation_id: str, is_archived=True):
    """
    Trash a generation in Sora,

    is_archived=True means trash, is_archived=False means untrash/restore
    """
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(
            f"{BASE_URL}/generations/{generation_id}", json={"is_archived": is_archived}
        ) as response:
            response.raise_for_status()
            json_data = await response.json()
            return json_data


async def archive_task(task_id: str):
    """
    Trash a task in Sora
    """
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(f"{BASE_URL}/video_gen/{task_id}/archive") as response:
            response.raise_for_status()
            json_data = await response.json()
            return json_data


async def delete_task(task_id: str):
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.delete(f"{BASE_URL}/video_gen/{task_id}") as response:
            response.raise_for_status()
            json_data = await response.json()
            return json_data


async def fetch_list_tasks(
    task_limit=20,
    after_task_id=None,
    archived=False,  # this is trash in sora
):
    params = {"limit": task_limit}
    if archived:
        params["archived"] = "true"
    if after_task_id:
        params["after"] = after_task_id

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(f"{BASE_URL}/v2/list_tasks", params=params) as response:
            response.raise_for_status()
            json_data = await response.json()
            return json_data


async def fetch_all_lists_tasks(
    task_limit=20,
    archived=False,  # this is trash in sora
):
    batch_count = 1
    all_tasks = []
    has_more = True
    last_id = None
    while has_more:
        try:
            data = await fetch_list_tasks(
                task_limit=task_limit,
                after_task_id=last_id,
                archived=archived,
            )
            last_id = data.get("last_id", None)
            has_more = data.get("has_more", False)
            tasks = data.get("task_responses", [])
            all_tasks.extend(tasks)
            print(
                f"Batch {batch_count} fetched, last_id {last_id}, has_more: {has_more}"
            )
            batch_count += 1
        except Exception as e:
            print(f"Batch {batch_count} fetch error, retrying..., error: {e}")

    return all_tasks


async def delete_empty_tasks():
    """
    Delete all tasks that have no generations.
    """
    empty_tasks = []
    tasks = await fetch_all_lists_tasks(task_limit=100, archived=False)
    for task in tasks:
        if len(task.get("generations", [])) == 0:
            task_id = task.get("id")
            empty_tasks.append(task_id)

    processed = 0
    total = len(empty_tasks)
    print(f"Total empty tasks to delete: {total}\n")

    async def delete(task_id):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                archive_data = await archive_task(task_id)
                deleted_data = await delete_task(task_id)
                processed += 1
                print(
                    f"[{msg_prefix_progress(processed, total)}] task {task_id} deleted\n"
                    f"archive: {json.dumps(archive_data)}\n"
                    f"delete: {json.dumps(deleted_data)}\n"
                )
                return
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] task {task_id} failed to delete: {e}, retrying...\n"
                )
        else:
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] task {task_id} failed to delete after {MAX_RETRIES} attempts.\n"
            )

    await asyncio.gather(*[delete(task_id) for task_id in empty_tasks])


async def save_dataset_from_generations(
    dataset: str,  # filename with extension (e.g. generations.csv)
    task_limit=100,
    archived=False,  # this is trash in sora
):
    generation_results = []
    tasks = await fetch_all_lists_tasks(
        task_limit=task_limit,
        archived=archived,
    )
    for task in tasks:
        for generation in task.get("generations", []):
            generation_results.append(
                {
                    "created_at": task.get("created_at"),
                    "id": generation.get("id"),
                    "task_id": generation.get("task_id"),
                    "url": generation.get("url"),
                    "prompt": generation.get("prompt"),
                }
            )

    if len(generation_results) == 0:
        df = pd.DataFrame(columns=["created_at", "id", "task_id", "url", "prompt"])
    else:
        df = pd.DataFrame(generation_results)
        df = df.sort_values(by="created_at", ascending=True).reset_index(drop=True)

    df.to_csv(get_output_path(dataset), index=False)


async def get_generation_download_url(generation_id):
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(
            f"{BASE_URL}/generations/{generation_id}/download"
        ) as response:
            response.raise_for_status()
            json_data = await response.json()
            return json_data.get("url", None)


async def download_generation_image(download_folder, generation_id):
    url = await get_generation_download_url(generation_id)
    # async with aiohttp.ClientSession(headers=headers) as session:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            file_name = f"{generation_id}.png"
            file_path = get_output_path(os.path.join(download_folder, file_name))
            response.raise_for_status()
            content = await response.read()
            with open(file_path, "wb") as f:
                f.write(content)


async def download_all_images(dataset, download_folder="sora_images"):

    df = pd.read_csv(get_output_path(dataset))
    processed = 0
    total = len(df)

    async def download(row):
        nonlocal processed

        generation_id = row.id
        file_name = f"{generation_id}.png"
        file_path = get_output_path(os.path.join(download_folder, file_name))

        if os.path.exists(file_path):
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {file_path} skipped, already exists",
            )
            return

        for _ in range(MAX_RETRIES):
            try:
                # async with aiohttp.ClientSession(headers=headers) as session:
                async with aiohttp.ClientSession() as session:
                    async with session.get(row.url) as response:
                        response.raise_for_status()
                        content = await response.read()
                        with open(file_path, "wb") as f:
                            f.write(content)
                        processed += 1
                        print(
                            f"[{msg_prefix_progress(processed, total)}] {file_path} downloaded",
                        )
                        break
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] {file_path} download error: {e}, retrying..."
                )
        else:
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {file_path} failed to download after {MAX_RETRIES} attempts."
            )

    await asyncio.gather(*[download(row) for row in df.itertuples(index=False)])


async def delete_generation(id: str):
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.delete(f"{BASE_URL}/generations/{id}") as response:
            response.raise_for_status()
            json_data = await response.json()
            return json_data


async def delete_generations(dataset):
    df = pd.read_csv(get_output_path(dataset))
    processed = 0
    total = len(df)

    async def delete(generation_id):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                await delete_generation(generation_id)
                processed += 1
                print(
                    f"[{msg_prefix_progress(processed, total)}] {generation_id} deleted"
                )
                return
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] {generation_id} error: {e}, retrying..."
                )
        else:
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {generation_id} failed to delete after {MAX_RETRIES} attempts."
            )

    await asyncio.gather(*[delete(row.id) for row in df.itertuples(index=False)])


async def delete_generations_already_uploaded_to_notion(
    dataset,
    db_id,
):
    df = pd.read_csv(get_output_path(dataset))
    processed = 0
    total = len(df)

    async def delete(generation_id):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                file_name = f"{generation_id}.png"
                if await is_page_exists_in_db(db_id, file_name):
                    await delete_generation(generation_id)
                    processed += 1
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {generation_id} deleted"
                    )
                else:
                    processed += 1
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {generation_id} skipped, not uploaded to notion yet"
                    )
                return
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] {generation_id} error: {e}, retrying..."
                )

    await asyncio.gather(*[delete(row.id) for row in df.itertuples(index=False)])


async def trash_generations_already_uploaded_to_notion(
    dataset,
    db_id,
):
    df = pd.read_csv(get_output_path(dataset))
    processed = 0
    total = len(df)

    async def trash(generation_id):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                file_name = f"{generation_id}.png"
                if await is_page_exists_in_db(db_id, file_name):
                    await archive_generation(generation_id)
                    processed += 1
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {generation_id} trashed"
                    )
                else:
                    processed += 1
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {generation_id} skipped, not uploaded to notion yet"
                    )
                return
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] {generation_id} error: {e}, retrying..."
                )

    await asyncio.gather(*[trash(row.id) for row in df.itertuples(index=False)])


async def upload_to_notion(
    dataset: str,
    image_folder: str,
    db_id,
    upload_to_notion=True,
    trash_in_sora=False,
    remove_in_sora=False,
):
    if trash_in_sora and remove_in_sora:
        raise ValueError("trash_in_sora and remove_in_sora cannot be both True.")

    print("📊 Saving dataset from generations...")
    await save_dataset_from_generations(dataset=dataset)
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

    if trash_in_sora:
        print("🗑️  Trashing generations already uploaded to Notion...")
        await trash_generations_already_uploaded_to_notion(
            dataset=dataset,
            db_id=db_id,
        )
        print("✅ Trashing completed.\n")

    if remove_in_sora:
        print("🗑️  Deleting generations already uploaded to Notion...")
        await delete_generations_already_uploaded_to_notion(
            dataset=dataset,
            db_id=db_id,
        )
        print("✅ Deletion completed.\n")


async def cleanup_trash(dataset: str):
    print("📊 Saving dataset from generations in trash folder...")
    await save_dataset_from_generations(dataset=dataset, archived=True)
    print("✅ Dataset saved.\n")

    print("🗑️  Deleting all generations in trash...")
    await delete_generations(dataset=dataset)
    print("✅ Cleanup completed.\n")


async def cleanup_tasks():
    print("🗑️  Deleting all empty tasks...")
    await delete_empty_tasks()
    print("✅ All empty tasks deleted.\n")
