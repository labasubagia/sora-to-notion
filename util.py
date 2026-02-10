import shutil
from pathlib import Path

MAX_RETRIES = 10

OUTPUT_PATH = "./output"


def msg_prefix_progress(processed: int, total: int) -> str:
    digit = str(total)
    percent: float = (processed / total) * 100
    processed_str: str = str(processed).rjust(len(digit), " ")
    return f"{processed_str}/{total}|{percent:6.2f}%"


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
