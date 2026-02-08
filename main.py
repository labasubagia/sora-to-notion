from sora import (
    save_dataset_from_generations,
    download_all_images,
    delete_generations_already_uploaded_to_notion,
    delete_generations,
)
from notion import upload_all_images_to_notion, DB_ID as NOTION_DB_ID


def sora_upload_to_notion(
    dataset: str, image_folder: str, db_id, upload_to_notion=True, remove_in_sora=False
):
    print("📊 Saving dataset from generations...")
    save_dataset_from_generations(dataset=dataset)
    print("✅ Dataset saved.\n")

    print("🖼️  Downloading all images...")
    download_all_images(
        dataset=dataset,
        download_folder=image_folder,
    )
    print("✅ All images downloaded.\n")

    if upload_to_notion:
        print("📤 Uploading all images to Notion...")
        upload_all_images_to_notion(
            dataset=dataset,
            db_id=db_id,
            image_folder=image_folder,
        )
        print("✅ All images uploaded to Notion.\n")

    if remove_in_sora:
        print("🗑️  Deleting generations already uploaded to Notion...")
        delete_generations_already_uploaded_to_notion(
            dataset=dataset,
            db_id=db_id,
        )
        print("✅ Deletion completed.\n")


def sora_cleanup_trash(dataset: str):
    print("📊 Saving dataset from generations in trash folder...")
    save_dataset_from_generations(dataset=dataset, archived=True)
    print("✅ Dataset saved.\n")

    print("🗑️  Deleting all generations in trash...")
    delete_generations(dataset=dataset)
    print("✅ Cleanup completed.\n")


if __name__ == "__main__":
    sora_upload_to_notion(
        dataset="generations.csv",
        image_folder="images",
        db_id=NOTION_DB_ID,
        upload_to_notion=True,
        remove_in_sora=True,
    )

    sora_cleanup_trash(dataset="trash_generations.csv")
