import pandas as pd
import streamlit as st
import yfinance as yf
from pathlib import Path
import os

from config import (CLUSTER_DATASET_CACHE_NAME, CLUSTER_DATASET_PERIOD, CURRENCY_LIST, 
                    DATASET_CACHE_NAME, DATASET_PERIOD, INTERESTED_DATA_FIELD, SELECTED_COINS, 
                    TEMP_DIR_NAME)
from util.file_handler import create_temp_folder_if_not_exists, delete_cache_files

def load_clustering_dataset() -> pd.DataFrame:

    create_temp_folder_if_not_exists()
    dataset_file_path = Path(TEMP_DIR_NAME, CLUSTER_DATASET_CACHE_NAME)

    if not dataset_file_path.is_file():
        print("Downloding the cluster dataset....")

        dataset_df = yf.download(CURRENCY_LIST, period=CLUSTER_DATASET_PERIOD)
        dataset_df = dataset_df[INTERESTED_DATA_FIELD]
        dataset_df.to_csv(dataset_file_path)
    
    dataset_df = pd.read_csv(dataset_file_path)
    dataset_df.set_index("Date", inplace=True)

    return dataset_df

def load_main_dataset() -> pd.DataFrame:

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

def delete_main_dataset_cache():
    dataset_file_path = Path(TEMP_DIR_NAME, DATASET_CACHE_NAME)
    if dataset_file_path.is_file():
        os.remove(dataset_file_path)

def delete_cluster_dataset_cache():
    dataset_file_path = Path(TEMP_DIR_NAME, CLUSTER_DATASET_CACHE_NAME)
    if dataset_file_path.is_file():
        os.remove(dataset_file_path)


def get_main_dataset(cryptocurrency_list: list = None) -> pd.DataFrame:
    """Returns the coin dataset
    coins (list): filters the dataset with given coins. If None, nothing is filtered
    """
    main_dataset = load_main_dataset()

    if cryptocurrency_list:
        return main_dataset[cryptocurrency_list]
    
    else:
        return main_dataset


def reload_dataset_and_train_model(selected_coin, model_type) -> pd.DataFrame:
    print("Calling data reload .............")

    with st.spinner("Clearing the old dataset"):
        delete_main_dataset_cache()

    with st.spinner("Deleting the model cache"):
        delete_cache_files(selected_coin, model_type)

    with st.spinner("Downloading the latest data from Yahoo Finance ..."):
        coin_data_df = get_main_dataset(SELECTED_COINS)

    return coin_data_df
