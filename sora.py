"""
Sora API interactions for fetching tasks, downloading images, and deleting generations.
NOTE: Required to use `curl` due to sora don't have offical API Access, and using `requests` causes 403 Forbidden errors.
"""

from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
import json
import os
import glob
from collections import deque
import subprocess
import pandas as pd
import requests
from notion import is_page_exists_in_db
from util import msg_prefix_progress
from dotenv import dotenv_values

config = dotenv_values()

BASE_URL = "https://sora.chatgpt.com"
AUTHORIZATION_TOKEN = config.get("SORA_AUTHORIZATION_TOKEN")
USER_AGENT = config.get("SORA_USER_AGENT")


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


def save_dataset_from_generations(
    dataset_folder: str,
    task_limit=100,
    archived=False,  # this is trash in sora
    dataset_count=0,
):
    dataset_folder = os.path.join(dataset_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    condition = {"is_first": True, "has_more": False, "last_id": None}
    while condition["is_first"] or condition["has_more"]:
        list_tasks = None
        fetch_list_tasks_attempt = 1
        while not list_tasks:
            try:
                list_tasks = fetch_lists_tasks(
                    task_limit=task_limit,
                    after_task_id=condition["last_id"],
                    archived=archived,
                )
            except Exception as e:
                print(
                    f"[{dataset_count}] Attempt {fetch_list_tasks_attempt} failed: {e}, retrying..."
                )
                fetch_list_tasks_attempt += 1
                list_tasks = None

        condition["is_first"] = False
        condition["has_more"] = list_tasks.get("has_more", False)
        condition["last_id"] = list_tasks.get("last_id", None)

        generation_results = []
        for task in list_tasks.get("task_responses", []):
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
        file_path = os.path.join(dataset_folder, f"generations_{dataset_count}.csv")
        df.to_csv(file_path, index=False)
        print(f"[{dataset_count}] Done fetching generations")
        dataset_count += 1


def get_generation_download_url(generation_id):
    command = [
        "curl",
        "-s",
        "-L",
        f"{BASE_URL}/backend/generations/{generation_id}/download",
        "-H",
        "-H",
        f"authorization: Bearer {AUTHORIZATION_TOKEN}",
        "-H",
        f"user-agent: {USER_AGENT}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    json_data = json.loads(result.stdout)
    return json_data.get("url")


def download_generation_image(download_folder, generation_id):
    url = get_generation_download_url(generation_id)
    download = requests.get(url)
    if download.status_code == 200:
        save_path = os.path.join(download_folder, f"{generation_id}.png")
        with open(save_path, "wb") as f:
            f.write(download.content)


def download_all_images(dataset_folder, download_folder, max_workers=10):
    files = glob.glob(os.path.join(dataset_folder, "*.csv"))
    df_list = [pd.read_csv(file) for file in files]
    df = pd.concat(df_list, ignore_index=True)

    download_folder = os.path.join(download_folder)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    processed = 0
    total = len(df)

    download_tasks = []
    for generation_id in df["id"]:
        download_path = os.path.join(download_folder, f"{generation_id}.png")
        if not os.path.exists(download_path):
            download_tasks.append(generation_id)

    def download(download_folder, generation_id):
        nonlocal processed
        try:
            download_generation_image(download_folder, generation_id)
            processed += 1
            print(
                f"[{msg_prefix_progress(processed, total)}] {generation_id} downloaded."
            )
        except Exception as e:
            print(
                f"[{msg_prefix_progress(processed, total)}] {generation_id} failed, error: {e}."
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download, download_folder, gen_id)
            for gen_id in download_tasks
        ]
        for future in as_completed(futures):
            future.result()


def delete_generation(id: str):
    command = [
        "curl",
        "-s",
        "-L",
        "-X",
        "DELETE",
        f"{BASE_URL}/backend/generations/{id}",
        "-H",
        "-H",
        f"authorization: Bearer {AUTHORIZATION_TOKEN}",
        "-H",
        f"user-agent: {USER_AGENT}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    json_data = json.loads(result.stdout)
    return json_data


def delete_generations(dataset_folder):
    files = glob.glob(os.path.join(dataset_folder, "*.csv"))
    df_list = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)

    queue = deque()
    for _, row in combined_df.iterrows():
        generation_id = row["id"]
        queue.append(generation_id)

    processed = 0
    total = len(combined_df)
    while queue:
        generation_id = queue.popleft()
        try:
            delete_generation(generation_id)
            processed += 1
            print(msg_prefix_progress(processed, total), generation_id, "deleted")
        except Exception as e:
            print(msg_prefix_progress(processed, total), generation_id, "error:", e)
            queue.append(generation_id)


def delete_generations_already_uploaded_to_notion(
    dataset_folder,
    db_id,
):
    files = glob.glob(os.path.join(dataset_folder, "*.csv"))
    df_list = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)

    queue = deque()
    for i, row in combined_df.iterrows():
        generation_id = row["id"]
        queue.append(generation_id)

    processed = 0
    total = len(combined_df)
    while queue:
        generation_id = queue.popleft()
        file_name = f"{generation_id}.png"
        try:
            if is_page_exists_in_db(file_name, db_id):
                delete_generation(generation_id)
                processed += 1
                print(
                    f"{msg_prefix_progress(processed, total)} {generation_id} deleted"
                )
            else:
                print(
                    f"{msg_prefix_progress(processed, total)} {generation_id} skipped"
                )
        except Exception as e:
            print(f"{msg_prefix_progress(processed, total)} {generation_id} error: {e}")
            queue.append(generation_id)
