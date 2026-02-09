from sora import (
    sora_upload_to_notion,
    sora_cleanup_trash,
    sora_cleanup_tasks,
)
from chatgpt import (
    chatgpt_upload_to_notion,
    delete_conversation_of_image_generation_uploaded_to_notion,
)
from notion import DB_ID as NOTION_DB_ID
import asyncio


if __name__ == "__main__":
    # sora_upload_to_notion(
    #     dataset="generations.csv",
    #     image_folder="images",
    #     db_id=NOTION_DB_ID,
    #     upload_to_notion=True,
    #     remove_in_sora=True,
    # )

    # sora_cleanup_trash(dataset="trash_generations.csv")

    # sora_cleanup_tasks()

    asyncio.run(
        chatgpt_upload_to_notion(
            dataset="chatgpt_generations.csv",
            image_folder="images",
            db_id=NOTION_DB_ID,
            upload_to_notion=True,
        ),
        # delete_conversation_of_image_generation_uploaded_to_notion(
        #     dataset="chatgpt_generations.csv", db_id=NOTION_DB_ID
        # ),
    )
