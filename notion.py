from functools import lru_cache
import requests
import os
import pandas as pd
from util import msg_prefix_progress
from dotenv import dotenv_values
from concurrent.futures import ThreadPoolExecutor, as_completed

config = dotenv_values()

BASE_URL = "https://api.notion.com"

API_KEY = config.get("NOTION_API_KEY")
DB_ID = config.get("NOTION_DATABASE_ID")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}


@lru_cache(maxsize=32)
def get_db_data_sources(db_id: str):
    response = requests.get(f"{BASE_URL}/v1/databases/{db_id}", headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get("data_sources", [])


def is_page_exists_in_db(
    db_id: str,
    query: str,
) -> bool:
    data_sources = get_db_data_sources(db_id)
    for data_source in data_sources:
        response = requests.post(
            f"{BASE_URL}/v1/data_sources/{data_source['id']}/query",
            headers=headers,
            json={
                "filter": {
                    "and": [{"property": "Name", "rich_text": {"equals": query}}]
                }
            },
        )
        response.raise_for_status()
        data = response.json()
        for page in data.get("results", []):
            name = page["properties"]["Name"]["title"][0]["text"]["content"]
            if name == query:
                return True
    return False


def create_upload_img(file_path: str):
    response = requests.post(
        f"{BASE_URL}/v1/file_uploads",
        headers=headers,
        json={
            "mode": "single_part",
            "filename": os.path.basename(file_path),
            "content_type": "image/png",
        },
    )
    response.raise_for_status()
    return response.json()


def send_upload_img(file_upload_id: str, file_path: str):
    with open(file_path, "rb") as f:
        file_name = os.path.basename(file_path)
        files = {
            "file": (file_name, f, "image/png"),
        }
        response = requests.post(
            f"{BASE_URL}/v1/file_uploads/{file_upload_id}/send",
            headers={
                "Authorization": headers["Authorization"],
                "Notion-Version": headers["Notion-Version"],
            },
            files=files,
        )
    response.raise_for_status()
    return response.json()


def add_page_to_db(file_path, prompt, model="Sora", face="_original_"):
    file_name = os.path.basename(file_path)
    create_upload_res = create_upload_img(file_path)
    send_upload_res = send_upload_img(create_upload_res["id"], file_path)
    response = requests.post(
        f"{BASE_URL}/v1/pages",
        headers=headers,
        json={
            "parent": {"database_id": DB_ID},
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
                "Prompt": {"rich_text": [{"text": {"content": prompt}}]},
                "Model": {"select": {"name": model}},
                "Face": {"select": {"name": face}},
            },
        },
    )
    response.raise_for_status()
    return response.json()


def upload_all_images_to_notion(dataset, db_id, image_folder, max_workers=10):
    df = pd.read_csv(os.path.join(dataset))
    processed = 0
    total = len(df)

    def upload(generation_id, prompt):
        nonlocal processed
        file_name = f"{generation_id}.png"
        file_path = os.path.join(image_folder, file_name)
        if os.path.exists(file_path):
            while True:
                try:
                    if is_page_exists_in_db(db_id, file_name):
                        processed += 1
                        print(
                            f"[{msg_prefix_progress(processed, total)}] {file_name} skipped, already exists",
                        )
                        break

                    add_page_to_db(file_path, prompt, model="Sora")
                    processed += 1
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {file_name} uploaded"
                    )
                    break
                except Exception as e:
                    print(
                        f"[{msg_prefix_progress(processed, total)}] {file_name} failed: {e}"
                    )

    upload_tasks = []
    for _, row in df.iterrows():
        upload_tasks.append((row["id"], row["prompt"]))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload, id, prompt) for (id, prompt) in upload_tasks]
        for future in as_completed(futures):
            future.result()
