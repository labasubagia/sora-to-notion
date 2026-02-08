import os
import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from util import msg_prefix_progress


def edit_png_info(file_path, payload: dict[str, str], overwrite=True):
    with Image.open(file_path) as img:
        metadata = PngInfo()
        for key, value in img.info.items():
            if isinstance(value, (str, int)):
                metadata.add_text(key, str(value))
        for key, value in payload.items():
            if overwrite or key not in img.info:
                metadata.add_text(key, value)
        img.save(file_path, pnginfo=metadata)


def add_prompt_to_all_images(dataset, max_workers=10):
    """
    This function adds prompt text metadata to all PNG images listed in the given dataset CSV file.

    How to read file result:

    `$ exiftool -Comment <file_path>`

    Example:

    `$ exiftool -Comment images/gen_01k2pct920ebyte8a69jyds5ds.png`
    """
    df = pd.read_csv(os.path.join(dataset))

    total = len(df)
    processed = 0

    def add_prompt(row):
        nonlocal processed
        file_path = f"images/{row['id']}.png"

        if os.path.exists(file_path):
            try:
                edit_png_info(
                    file_path,
                    payload={"Comment": row["prompt"]},
                )
                processed += 1
                print(f"[{msg_prefix_progress(processed, total)}] {file_path} edited.")
            except Exception as e:
                print(
                    f"[{msg_prefix_progress(processed, total)}] {file_path} failed: {e}"
                )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(add_prompt, row) for (_, row) in df.iterrows()]
        for future in as_completed(futures):
            future.result()
