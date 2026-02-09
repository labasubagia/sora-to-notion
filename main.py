import sora
import chatgpt
import asyncio
import typer
from notion import DB_ID as NOTION_DB_ID
from typing import Annotated


app = typer.Typer()


@app.command()
def sora_upload_to_notion(
    dataset: Annotated[
        str, typer.Option(help="Path to the dataset CSV file")
    ] = "generations.csv",
    image_folder: Annotated[
        str, typer.Option(help="Path to the folder containing images")
    ] = "images",
    db_id: Annotated[str, typer.Option(help="Notion Database ID")] = NOTION_DB_ID,
    upload_to_notion: Annotated[
        bool, typer.Option(help="Whether to upload to Notion")
    ] = True,
    remove_in_sora: Annotated[
        bool, typer.Option(help="Whether to remove uploaded items in Sora")
    ] = False,
):
    sora.upload_to_notion(
        dataset,
        image_folder,
        db_id,
        upload_to_notion=upload_to_notion,
        remove_in_sora=remove_in_sora,
    )


@app.command()
def sora_cleanup_trash(
    dataset: Annotated[
        str, typer.Option(help="Path to the dataset CSV file")
    ] = "generations.csv",
):
    sora.cleanup_trash(dataset)


@app.command()
def sora_cleanup_tasks():
    sora.cleanup_tasks()


@app.command()
def chatgpt_upload_to_notion(
    dataset: Annotated[
        str, typer.Option(help="Path to the dataset CSV file")
    ] = "chatgpt_generations.csv",
    image_folder: Annotated[
        str, typer.Option(help="Path to the folder containing images")
    ] = "chatgpt_images",
    db_id: Annotated[str, typer.Option(help="Notion Database ID")] = NOTION_DB_ID,
):
    asyncio.run(
        chatgpt.upload_to_notion(
            dataset=dataset,
            image_folder=image_folder,
            db_id=db_id,
        )
    )


@app.command()
def chatgpt_cleanup_if_uploaded_to_notion(
    dataset: Annotated[
        str, typer.Option(help="Path to the dataset CSV file")
    ] = "chatgpt_generations.csv",
    db_id: Annotated[str, typer.Option(help="Notion Database ID")] = NOTION_DB_ID,
):
    asyncio.run(
        chatgpt.delete_conversation_of_image_generation_uploaded_to_notion(
            dataset=dataset,
            db_id=db_id,
        )
    )


if __name__ == "__main__":
    app()
