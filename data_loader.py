import pandas as pd
import yfinance as yf
from pathlib import Path
import os

from config import CLUSTER_DATASET_CACHE_NAME, CLUSTER_DATASET_PERIOD, CURRENCY_LIST, DATASET_CACHE_NAME, DATASET_PERIOD, INTERESTED_DATA_FIELD, TEMP_DIR_NAME
from util.file_handler import create_temp_folder_if_not_exists

def load_clustering_data():

    create_temp_folder_if_not_exists()
    dataset_file_path = Path(TEMP_DIR_NAME, CLUSTER_DATASET_CACHE_NAME)

    if not dataset_file_path.is_file():
        print("Downloding the cluster dataset....")

        dataset_df = yf.download(CURRENCY_LIST, period=CLUSTER_DATASET_PERIOD)
        dataset_df = dataset_df[INTERESTED_DATA_FIELD]
        dataset_df.to_csv(dataset_file_path)
    
    else:
        print("Using the cached clustering dataset..")
        dataset_df = pd.read_csv(dataset_file_path)
        dataset_df.set_index("Date", inplace=True)

    return dataset_df

def load_data():

    create_temp_folder_if_not_exists()
    dataset_file_path = Path(TEMP_DIR_NAME, DATASET_CACHE_NAME)

    if not dataset_file_path.is_file():
        print("Downloding the dataset....")

        dataset_df = yf.download(CURRENCY_LIST, period=DATASET_PERIOD)
        dataset_df = dataset_df[INTERESTED_DATA_FIELD]
        dataset_df.to_csv(dataset_file_path)
    
    dataset_df = pd.read_csv(dataset_file_path)
    dataset_df.set_index("Date", inplace=True)

    return dataset_df

def delete_dataset_cache():
    dataset_file_path = Path(TEMP_DIR_NAME, DATASET_CACHE_NAME)
    if dataset_file_path.is_file():
        os.remove(dataset_file_path)
