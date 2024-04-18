from pathlib import Path
import os
from os import walk
import time

from constants import TEMP_DIR_NAME

def create_temp_folder_if_not_exists():
    Path(TEMP_DIR_NAME).mkdir(parents=True, exist_ok=True)

def is_file_exits(file_path: str) -> bool:
    create_temp_folder_if_not_exists()

    for (_, _, filenames) in walk(TEMP_DIR_NAME):

        for filename in filenames:
            file_name_prefix = "_".join(filename.split("_")[:-1])

            if file_name_prefix and file_name_prefix in file_path:
                return True
    
    return False

def get_temp_file_path(*paths, is_scaler: bool = False) -> str:

    file_name_prefix = "_".join(paths)
    is_scaler_indicator = 'scaler' if is_scaler else ''

    for (_, _, filenames) in walk(TEMP_DIR_NAME):
        for filename in filenames:
            if file_name_prefix in filename:
                if is_scaler and filename.endswith('.scalermodel'):
                    return os.path.join(TEMP_DIR_NAME, filename)
                
                elif not is_scaler and not filename.endswith('.scalermodel'):
                    return os.path.join(TEMP_DIR_NAME, filename)
    
    return os.path.join(TEMP_DIR_NAME, f'{file_name_prefix}_{int(time.time())}.{is_scaler_indicator}model')

def get_model_time_by_file_path(file_path) -> str:
    timestamp = int(file_path.split("_")[-1].split('.')[0])
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp))

def get_model_time(coin_name: str, model_type: str):
    cache_file_prefix = f'{coin_name}_{model_type}'

    for (root, _, filenames) in walk(TEMP_DIR_NAME):
        for filename in filenames:
            if filename.startswith(cache_file_prefix):
                return get_model_time_by_file_path(filename)

    return ''

def delete_cache_files(coin_name: str, model_type: str):

    cache_file_prefix = f'{coin_name}_{model_type}'

    for (root, _, filenames) in walk(TEMP_DIR_NAME):
        for filename in filenames:
            if filename.startswith(cache_file_prefix):
                candidate_file_path_to_remove = os.path.join(root, filename)
                os.remove(candidate_file_path_to_remove)
                print(f"File: {candidate_file_path_to_remove} has been removed")
