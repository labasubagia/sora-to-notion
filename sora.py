import asyncio
import os

import aiohttp
import pandas as pd
from dotenv import dotenv_values
from tqdm.asyncio import tqdm

from notion import is_page_exists_in_db, upload_all_images_to_notion
from util import MAX_RETRIES, get_output_path, http_retryable

config = dotenv_values()

BASE_URL = "https://sora.chatgpt.com/backend"

headers = {
    "Authorization": f"Bearer {config.get('CHATGPT_AUTHORIZATION_TOKEN')}",
    "User-Agent": config.get("CHATGPT_USER_AGENT"),
    "Cookie": config.get("CHATGPT_COOKIE_STRING"),
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


async def fetch_recent_tasks(limit=100, before_task_id: str = None, archived=False):
    params = {"limit": limit}
    if before_task_id:
        params["before"] = before_task_id
    if archived:
        params["archived"] = "true"

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(
            f"{BASE_URL}/v2/recent_tasks", params=params
        ) as response:
            response.raise_for_status()
            data_json = await response.json()
            return data_json


async def fetch_list_tasks(
    limit=20,
    after_task_id=None,
    archived=False,  # this is trash in sora
):
    params = {"limit": limit}
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
    limit=20,
    archived=False,  # this is trash in sora
):
    batch_count = 1
    all_tasks = []
    has_more = True
    last_id = None
    while has_more:
        try:
            data = await fetch_list_tasks(
                limit=limit,
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
        except aiohttp.ClientError as e:
            if not http_retryable(e.status):
                print(f"Batch {batch_count} fetch error, not retryable HTTP error: {e}")
                break
            print(f"Batch {batch_count} fetch HTTP error, retrying..., error: {e}")
        except Exception as e:
            print(f"Batch {batch_count} fetch error, retrying..., error: {e}")

    return all_tasks


async def delete_empty_tasks():
    """
    Delete all tasks that have no generations.
    """
    empty_tasks = []
    tasks = await fetch_all_lists_tasks(limit=100, archived=False)
    for task in tasks:
        if len(task.get("generations", [])) == 0:
            task_id = task.get("id")
            empty_tasks.append(task_id)

    total = len(empty_tasks)
    pbar = tqdm(total=total, desc="Deleting empty tasks")

    async def delete(task_id):
        for _ in range(MAX_RETRIES):
            try:
                _ = await archive_task(task_id)
                _ = await delete_task(task_id)
                pbar.write(
                    f"✅ task {task_id}\n"
                    # f"archive: {json.dumps(archive_data)}\n"
                    # f"delete: {json.dumps(deleted_data)}"
                )
                pbar.update(1)
                return
            except Exception as e:
                pbar.write(f"⚠️  task {task_id} failed to delete: {e}, retrying...")
        else:
            pbar.write(
                f"❌ task {task_id} failed to delete after {MAX_RETRIES} attempts"
            )
            pbar.update(1)

    await asyncio.gather(*[delete(task_id) for task_id in empty_tasks])
    pbar.close()
    print()


def get_generations_from_tasks(tasks):
    generations = []
    for task in tasks:
        for generation in task.get("generations", []):
            generations.append(
                {
                    "created_at": task.get("created_at"),
                    "id": generation.get("id"),
                    "task_id": generation.get("task_id"),
                    "url": generation.get("url"),
                    "prompt": generation.get("prompt"),
                }
            )
    generations = sorted(generations, key=lambda x: x["created_at"])
    return generations


def save_dataset_generations(dataset: str, generations):
    if dataset is None:
        return
    if len(generations) == 0:
        df = pd.DataFrame(columns=["created_at", "id", "task_id", "url", "prompt"])
    else:
        df = pd.DataFrame(generations)
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


async def download_all_images(generations, download_folder="sora_images"):
    total = len(generations)
    pbar = tqdm(total=total, desc="Downloading images")

    async def download(generation):
        generation_id = generation["id"]
        url = generation["url"]
        file_name = f"{generation_id}.png"
        file_path = get_output_path(os.path.join(download_folder, file_name))

        if os.path.exists(file_path):
            pbar.write(f"⏭️  {file_name} skipped, already exists")
            pbar.update(1)
            return

        for _ in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        content = await response.read()
                        with open(file_path, "wb") as f:
                            f.write(content)
                        pbar.write(f"✅ {file_name}")
                        pbar.update(1)
                        return
            except aiohttp.ClientError as e:
                if not http_retryable(e.status):
                    pbar.write(f"❌ {file_name} HTTP error: {e}, not retryable")
                    pbar.update(1)
                    return
                pbar.write(f"⚠️  {file_name} HTTP error: {e}, retrying...")
            except Exception as e:
                pbar.write(f"⚠️  {file_name} error: {e}, retrying...")
        else:
            pbar.write(f"❌ {file_name} failed after {MAX_RETRIES} attempts")
            pbar.update(1)

    await asyncio.gather(*[download(gen) for gen in generations])
    pbar.close()
    print()


async def delete_generation(id: str):
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.delete(f"{BASE_URL}/generations/{id}") as response:
            response.raise_for_status()
            json_data = await response.json()
            return json_data


async def delete_generations(generations):
    total = len(generations)
    pbar = tqdm(total=total, desc="Deleting generations")

    async def delete(generation):
        generation_id = generation.get("id")
        for _ in range(MAX_RETRIES):
            try:
                await delete_generation(generation_id)
                pbar.write(f"✅ {generation_id}")
                pbar.update(1)
                return
            except aiohttp.ClientError as e:
                if not http_retryable(e.status):
                    pbar.write(f"❌ {generation_id} HTTP error: {e}, not retryable")
                    pbar.update(1)
                    return
                pbar.write(f"⚠️  {generation_id} HTTP error: {e}, retrying...")
            except Exception as e:
                pbar.write(f"⚠️  {generation_id} error: {e}, retrying...")
        else:
            pbar.write(
                f"❌ {generation_id} failed to delete after {MAX_RETRIES} attempts"
            )
            pbar.update(1)

    await asyncio.gather(*[delete(gen) for gen in generations])
    pbar.close()
    print()


async def delete_generations_already_uploaded_to_notion(
    generations,
    db_id,
):
    total = len(generations)
    pbar = tqdm(total=total, desc="Deleting uploaded generations")

    async def delete(generation):
        generation_id = generation.get("id")
        for _ in range(MAX_RETRIES):
            try:
                file_name = f"{generation_id}.png"
                if await is_page_exists_in_db(db_id, file_name):
                    await delete_generation(generation_id)
                    pbar.write(f"✅ {generation_id} deleted")
                else:
                    pbar.write(
                        f"⏭️  {generation_id} skipped, not uploaded to notion yet"
                    )
                pbar.update(1)
                return
            except aiohttp.ClientError as e:
                if not http_retryable(e.status):
                    pbar.write(f"❌ {generation_id} HTTP error: {e}, not retryable")
                    pbar.update(1)
                    return
                pbar.write(f"⚠️  {generation_id} HTTP error: {e}, retrying...")
            except Exception as e:
                pbar.write(f"⚠️  {generation_id} error: {e}, retrying...")
        else:
            pbar.write(f"❌ {generation_id} failed after {MAX_RETRIES} attempts")
            pbar.update(1)

    await asyncio.gather(*[delete(gen) for gen in generations])
    pbar.close()
    print()


async def trash_generations_already_uploaded_to_notion(
    generations,
    db_id,
):
    total = len(generations)
    pbar = tqdm(total=total, desc="Trashing uploaded generations")

    async def trash(generation):
        generation_id = generation.get("id")
        for _ in range(MAX_RETRIES):
            try:
                file_name = f"{generation_id}.png"
                if await is_page_exists_in_db(db_id, file_name):
                    await archive_generation(generation_id)
                    pbar.write(f"✅ {generation_id}")
                else:
                    pbar.write(
                        f"⏭️  {generation_id} skipped, not uploaded to notion yet"
                    )
                pbar.update(1)
                return
            except aiohttp.ClientError as e:
                if not http_retryable(e.status):
                    pbar.write(f"❌ {generation_id} HTTP error: {e}, not retryable")
                    pbar.update(1)
                    return
                pbar.write(f"⚠️  {generation_id} HTTP error: {e}, retrying...")
            except Exception as e:
                pbar.write(f"⚠️  {generation_id} error: {e}, retrying...")
        else:
            pbar.write(f"❌ {generation_id} failed after {MAX_RETRIES} attempts")
            pbar.update(1)

    await asyncio.gather(*[trash(gen) for gen in generations])
    pbar.close()
    print()


async def upload_to_notion(
    image_folder: str = None,
    db_id=None,
    upload_to_notion=True,
    trash_in_sora=False,
    remove_in_sora=False,
    limit=100,
):
    if trash_in_sora and remove_in_sora:
        raise ValueError("trash_in_sora and remove_in_sora cannot be both True.")

    data = await fetch_recent_tasks(limit=limit, archived=False)
    tasks = data.get("task_responses", [])
    generations = get_generations_from_tasks(tasks)

    await download_all_images(
        generations=generations,
        download_folder=image_folder,
    )

    if upload_to_notion:
        await upload_all_images_to_notion(
            generations=generations,
            db_id=db_id,
            image_folder=image_folder,
        )

    if trash_in_sora:
        await trash_generations_already_uploaded_to_notion(
            generations=generations,
            db_id=db_id,
        )

    if remove_in_sora:
        await delete_generations_already_uploaded_to_notion(
            generations=generations,
            db_id=db_id,
        )


async def cleanup_trash(task_limit=100):
    tasks = await fetch_all_lists_tasks(limit=task_limit, archived=True)
    generations = get_generations_from_tasks(tasks)
    await delete_generations(generations=generations)


async def cleanup_tasks():
    await delete_empty_tasks()
