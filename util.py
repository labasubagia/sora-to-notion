import shutil
from pathlib import Path

import pandas as pd

MAX_RETRIES = 5

OUTPUT_PATH = "./output"


def save_to_dataset(dataset: str, data: list[dict]):
    if dataset is None:
        return
    if len(data) == 0:
        print("No generations to save to dataset.")
        return
    else:
        df = pd.DataFrame(data)
    file_path = get_output_path(dataset)
    df.to_csv(file_path, index=False)
    print(f"✅ Saved dataset to {file_path}\n")


def get_output_path(input_path_str: str, is_dir=False) -> Path:
    input_path = Path(input_path_str)

    if input_path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {input_path_str}")

    base_dir: Path = Path(OUTPUT_PATH).resolve()
    final_path: Path = (base_dir / input_path).resolve()
    if base_dir not in final_path.parents:
        raise ValueError("Path attempts to escape the output directory!")

    if is_dir:
        final_path.mkdir(parents=True, exist_ok=True)
    else:
        final_path.parent.mkdir(parents=True, exist_ok=True)

    return final_path


def clean_output_path():
    # Except .gitkeep, force remove all files and folders in OUTPUT_PATH
    base_dir: Path = Path(OUTPUT_PATH).resolve()
    if not base_dir.exists():
        return
    for item in base_dir.iterdir():
        if item.is_file() and item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def http_retryable(status_code: int) -> bool:
    retryable_statuses = (
        500,
        502,
        503,
        504,
    )
    return status_code in retryable_statuses
