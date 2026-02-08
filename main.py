from sora import (
    save_dataset_from_generations,
    download_all_images,
    delete_generations_already_uploaded_to_notion,
    delete_generations,
)
from notion import upload_all_images_to_notion, DB_ID


def sora_upload_to_notion(remove_in_sora=False):
    dataset_folder = "generations"
    save_dataset_from_generations(dataset_folder=dataset_folder)
    download_all_images(
        dataset_folder=dataset_folder,
        download_folder="images",
    )
    upload_all_images_to_notion(
        dataset_folder=dataset_folder,
        db_id=DB_ID,
    )
    if remove_in_sora:
        delete_generations_already_uploaded_to_notion(
            dataset_folder=dataset_folder,
            db_id=DB_ID,
        )


def sora_cleanup_trash():
    dataset_folder = "trash_generations"
    save_dataset_from_generations(dataset_folder=dataset_folder, archived=True)
    delete_generations(dataset_folder=dataset_folder)


if __name__ == "__main__":
    # save_dataset_from_generations(dataset_folder="generations")
    sora_cleanup_trash()
