"""
Sora API interactions for fetching tasks, downloading images, and deleting generations.
NOTE: Required to use `curl` due to sora don't have offical API Access, and using `requests` causes 403 Forbidden errors.
"""

import asyncio
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import pandas as pd
import requests
from dotenv import dotenv_values

from notion import is_page_exists_in_db, upload_all_images_to_notion
from util import MAX_RETRIES, get_output_path, msg_prefix_progress

config = dotenv_values()

BASE_URL = "https://sora.chatgpt.com"
AUTHORIZATION_TOKEN = config.get("CHATGPT_AUTHORIZATION_TOKEN")
USER_AGENT = config.get("CHATGPT_USER_AGENT")


def archive_task(task_id: str):
    """
    Trash in Sora
    """
    command = [
        "curl",
        "-s",
        "-L",
        "-X",
        "POST",
        f"{BASE_URL}/backend/video_gen/{task_id}/archive",
        "-H",
        f"authorization: Bearer {AUTHORIZATION_TOKEN}",
        "-H",
        f"user-agent: {USER_AGENT}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    json_data = json.loads(result.stdout)
    return json_data


def delete_task(task_id: str):
    command = [
        "curl",
        "-s",
        "-L",
        "-X",
        "DELETE",
        f"{BASE_URL}/backend/video_gen/{task_id}",
        "-H",
        f"authorization: Bearer {AUTHORIZATION_TOKEN}",
        "-H",
        f"user-agent: {USER_AGENT}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    json_data = json.loads(result.stdout)
    return json_data


def fetch_lists_tasks(
    task_limit=20,
    after_task_id=None,
    archived=False,  # this is trash in sora
):
    url = f"{BASE_URL}/backend/v2/list_tasks?limit={task_limit}"
    if after_task_id:
        url += f"&after={after_task_id}"
    if archived:
        url += "&archived=true"

    command = [
        "curl",
        "-s",
        "-L",
        url,
        "-H",
        f"authorization: Bearer {AUTHORIZATION_TOKEN}",
        "-H",
        f"user-agent: {USER_AGENT}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    json_data = json.loads(result.stdout)
    return json_data


def fetch_all_lists_tasks(
    task_limit=100,
    archived=False,  # this is trash in sora
):
    all_tasks = []
    last_id = None
    request_count = 1
    while True:
        try:
            data = fetch_lists_tasks(
                task_limit=task_limit,
                after_task_id=last_id,
                archived=archived,
            )
            last_id = data.get("last_id", None)
            all_tasks.extend(data.get("task_responses", []))
            print(
                f"{request_count} Fetched last_id {last_id}, has_more: {data['has_more']}"
            )
            request_count += 1
            if not data.get("has_more", False):
                break
        except Exception as e:
            print(f"{request_count} Error fetching tasks, retrying..., error: {e}")

    return all_tasks


def delete_all_empty_tasks(max_workers=10):
    """
    Delete all tasks that have no generations.

    :param max_workers: max parallel workers for deleting tasks
    :return: None
    """
    empty_tasks = []
    tasks = fetch_all_lists_tasks(task_limit=100, archived=False)
    for task in tasks:
        if len(task.get("generations", [])) == 0:
            task_id = task.get("id")
            empty_tasks.append(task_id)
    print()
    processed = 0
    total = len(empty_tasks)
    print(f"Total failed tasks to delete: {total}\n")

    def delete(task_id):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                archive_data = archive_task(task_id)
                deleted_data = delete_task(task_id)
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(delete, id) for id in empty_tasks]
        for future in as_completed(futures):
            future.result()


def save_dataset_from_generations(
    dataset: str,  # filename with extension (e.g. generations.csv)
    task_limit=100,
    archived=False,  # this is trash in sora
):
    generation_results = []
    tasks = fetch_all_lists_tasks(
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


def get_generation_download_url(generation_id):
    command = [
        "curl",
        "-s",
        "-L",
        f"{BASE_URL}/backend/generations/{generation_id}/download",
        "-H",
        f"authorization: Bearer {AUTHORIZATION_TOKEN}",
        "-H",
        f"user-agent: {USER_AGENT}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    json_data = json.loads(result.stdout)
    return json_data.get("url", None)


def download_generation_image(download_folder, generation_id):
    url = get_generation_download_url(generation_id)
    download = requests.get(url)
    download.raise_for_status()
    if download.status_code == 200:
        file_name = f"{generation_id}.png"
        save_path = get_output_path(os.path.join(download_folder, file_name))
        with open(save_path, "wb") as f:
            f.write(download.content)


async def download_all_images(dataset, download_folder="sora_images"):

    df = pd.read_csv(get_output_path(dataset))
    processed = 0
    total = len(df)

    async def download(row):
        nonlocal processed

        generation_id = row["id"]
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
                async with aiohttp.ClientSession() as session:
                    async with session.get(row["url"]) as response:
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

    await asyncio.gather(*[download(row) for _, row in df.iterrows()])


def delete_generation(id: str):
    command = [
        "curl",
        "-s",
        "-L",
        "-X",
        "DELETE",
        f"{BASE_URL}/backend/generations/{id}",
        "-H",
        f"authorization: Bearer {AUTHORIZATION_TOKEN}",
        "-H",
        f"user-agent: {USER_AGENT}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    json_data = json.loads(result.stdout)
    return json_data


def delete_generations(dataset, max_workers=10):
    df = pd.read_csv(get_output_path(dataset))
    processed = 0
    total = len(df)

    delete_tasks = []
    for _, row in df.iterrows():
        generation_id = row["id"]
        delete_tasks.append(generation_id)

    def delete(generation_id):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                delete_generation(generation_id)
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(delete, gen_id) for gen_id in delete_tasks]
        for future in as_completed(futures):
            future.result()


async def delete_generations_already_uploaded_to_notion(
    dataset,
    db_id,
):
    df = pd.read_csv(get_output_path(dataset))
    processed = 0
    total = len(df)

    delete_tasks = []
    for _, row in df.iterrows():
        generation_id = row["id"]
        delete_tasks.append(generation_id)

    async def delete(generation_id):
        nonlocal processed
        for _ in range(MAX_RETRIES):
            try:
                file_name = f"{generation_id}.png"
                if await is_page_exists_in_db(db_id, file_name):
                    delete_generation(generation_id)
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

    await asyncio.gather(*[delete(gen_id) for gen_id in delete_tasks])


def upload_to_notion(
    dataset: str, image_folder: str, db_id, upload_to_notion=True, remove_in_sora=False
):
    print("📊 Saving dataset from generations...")
    save_dataset_from_generations(dataset=dataset)
    print("✅ Dataset saved.\n")

    print("🖼️  Downloading all images...")
    asyncio.run(
        download_all_images(
            dataset=dataset,
            download_folder=image_folder,
        )
    )
    print("✅ All images downloaded.\n")

    if upload_to_notion:
        print("📤 Uploading all images to Notion...")
        asyncio.run(
            upload_all_images_to_notion(
                dataset=dataset,
                db_id=db_id,
                image_folder=image_folder,
            )
        )
        print("✅ All images uploaded to Notion.\n")

    if remove_in_sora:
        print("🗑️  Deleting generations already uploaded to Notion...")
        asyncio.run(
            delete_generations_already_uploaded_to_notion(
                dataset=dataset,
                db_id=db_id,
            )
        )
        print("✅ Deletion completed.\n")


def cleanup_trash(dataset: str):
    print("📊 Saving dataset from generations in trash folder...")
    save_dataset_from_generations(dataset=dataset, archived=True)
    print("✅ Dataset saved.\n")

    print("🗑️  Deleting all generations in trash...")
    delete_generations(dataset=dataset)
    print("✅ Cleanup completed.\n")


def cleanup_tasks():
    print("🗑️  Deleting all empty tasks...")
    delete_all_empty_tasks()
    print("✅ All empty tasks deleted.\n")
