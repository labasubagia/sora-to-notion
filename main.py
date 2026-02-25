import asyncio
from datetime import datetime
from typing import Annotated

import typer

import chatgpt
import sora
import util
from notion import DB_ID as NOTION_DB_ID

app = typer.Typer()


def validate_db_id(db_id: str) -> str:
    """Validate Notion database ID"""
    if not db_id or len(db_id) < 10:
        raise typer.BadParameter("Notion database ID must be a valid ID")
    return db_id


@app.command()
def sora_upload_to_notion(
    image_folder: Annotated[
        str, typer.Option(help="Path to the folder containing images")
    ] = "sora_images",
    db_id: Annotated[
        str,
        typer.Option(
            help="Notion Database ID", callback=validate_db_id
        ),
    ] = NOTION_DB_ID,
    upload_to_notion: Annotated[
        bool, typer.Option(help="Whether to upload to Notion")
    ] = True,
    trash_in_sora: Annotated[
        bool, typer.Option(help="Whether to trash uploaded items in Sora")
    ] = False,
    remove_in_sora: Annotated[
        bool, typer.Option(help="Whether to remove uploaded items in Sora")
    ] = False,
    dataset: Annotated[
        str, typer.Option(help="Save generations to dataset CSV file")
    ] = f"sora_{datetime.now().isoformat()}.csv",
):
    """Upload Sora generations to Notion"""
    util.validate_env_vars([
        "NOTION_API_KEY",
        "NOTION_DATABASE_ID",
        "CHATGPT_AUTHORIZATION_TOKEN",
        "CHATGPT_USER_AGENT",
    ])
    asyncio.run(
        sora.upload_to_notion(
            image_folder,
            db_id,
            upload_to_notion=upload_to_notion,
            trash_in_sora=trash_in_sora,
            remove_in_sora=remove_in_sora,
            dataset=dataset,
        )
    )


@app.command()
def sora_cleanup_trash(
    dataset: Annotated[
        str, typer.Option(help="Save generations to dataset CSV file")
    ] = f"sora_trash_{datetime.now().isoformat()}.csv",
):
    """Clean up trashed Sora generations"""
    util.validate_env_vars([
        "CHATGPT_AUTHORIZATION_TOKEN",
        "CHATGPT_USER_AGENT",
    ])
    asyncio.run(sora.cleanup_trash(dataset=dataset))


@app.command()
def sora_cleanup_tasks():
    """Clean up empty Sora tasks"""
    util.validate_env_vars([
        "CHATGPT_AUTHORIZATION_TOKEN",
        "CHATGPT_USER_AGENT",
    ])
    asyncio.run(sora.cleanup_tasks())


@app.command()
def chatgpt_upload_to_notion(
    image_folder: Annotated[
        str, typer.Option(help="Path to the folder containing images")
    ] = "chatgpt_images",
    db_id: Annotated[
        str,
        typer.Option(
            help="Notion Database ID", callback=validate_db_id
        ),
    ] = NOTION_DB_ID,
    upload_to_notion: Annotated[
        bool, typer.Option(help="Whether to upload to Notion")
    ] = True,
    remove_in_chatgpt: Annotated[
        bool, typer.Option(help="Whether to remove uploaded items in ChatGPT")
    ] = False,
    dataset: Annotated[
        str, typer.Option(help="Save generations to dataset CSV file")
    ] = f"chatgpt_{datetime.now().isoformat()}.csv",
    limit: Annotated[
        int, typer.Option(help="Limit number of image generations to process")
    ] = 100,
):
    """Upload ChatGPT image generations to Notion"""
    util.validate_env_vars([
        "NOTION_API_KEY",
        "NOTION_DATABASE_ID",
        "CHATGPT_AUTHORIZATION_TOKEN",
        "CHATGPT_USER_AGENT",
        "CHATGPT_COOKIE_STRING_BASE64",
    ])
    asyncio.run(
        chatgpt.upload_to_notion(
            image_folder=image_folder,
            db_id=db_id,
            upload_to_notion=upload_to_notion,
            remove_in_chatgpt=remove_in_chatgpt,
            dataset=dataset,
            limit=limit,
        )
    )


@app.command()
def clean_output_path():
    """Clean the output directory (except .gitkeep)"""
    print("Cleaning output path...")
    util.clean_output_path()
    print("Output path cleaned.")


if __name__ == "__main__":
    app()
