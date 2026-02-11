import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from tqdm import tqdm

from util import MAX_RETRIES, get_output_path


def edit_png_info(file_path, payload: dict[str, str], overwrite=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with Image.open(file_path) as img:
        metadata = PngInfo()
        for key, value in img.info.items():
            if isinstance(value, (str, int)):
                metadata.add_text(key, str(value))
        for key, value in payload.items():
            if overwrite or key not in img.info:
                metadata.add_text(key, value)
        img.save(file_path, pnginfo=metadata)


def add_prompt_to_all_images(dataset, folder, max_workers=10):
    """
    This function adds prompt text metadata to all PNG images listed in the given dataset CSV file.

    How to read file result:

    `$ exiftool -Comment <file_path>`

    Example:

    `$ exiftool -Comment images/gen_01k2pct920ebyte8a69jyds5ds.png`
    """
    df = pd.read_csv(get_output_path(dataset))

    total = len(df)
    pbar = tqdm(total=total, desc="Adding prompts to images")

    def add_prompt(row):
        file_name = f"{row['id']}.png"
        file_path = get_output_path(os.path.join(folder, file_name))
        if not os.path.exists(file_path):
            pbar.write(f"⚠️  {file_name} not found, skipped")
            pbar.update(1)
            return

        for _ in range(MAX_RETRIES):
            try:
                edit_png_info(
                    file_path,
                    payload={"Comment": row["prompt"]},
                )
                pbar.write(f"✅ {file_name} edited")
                pbar.update(1)
                break
            except Exception as e:
                pbar.write(f"⚠️  {file_name} edit error: {e}, retrying...")
        else:
            pbar.write(f"❌ {file_name} edit failed after {MAX_RETRIES} retries")
            pbar.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(add_prompt, row) for row in df.itertuples(index=False)
        ]
        for future in as_completed(futures):
            future.result()

    pbar.close()
    print()
