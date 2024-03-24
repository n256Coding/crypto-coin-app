from pathlib import Path
import os

from constants import TEMP_DIR_NAME

def create_temp_folder_if_not_exists():
    Path(TEMP_DIR_NAME).mkdir(parents=True, exist_ok=True)

def is_file_exits(file_path: str):
    create_temp_folder_if_not_exists()
    return os.path.isfile(file_path)

def get_temp_file_path(*paths):
    return os.path.join(TEMP_DIR_NAME, "_".join(paths))
