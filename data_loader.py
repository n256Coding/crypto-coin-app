import pandas as pd
import yfinance as yf
from pathlib import Path

from constants import CURRENCY_LIST, DATASET_CACHE_NAME, DATASET_PERIOD, INTERESTED_DATA_FIELD

def load_data():

    dataset_file_path = Path(DATASET_CACHE_NAME)

    if not dataset_file_path.is_file():
        print("Downloding the dataset....")

        dataset_df = yf.download(CURRENCY_LIST, period=DATASET_PERIOD)
        dataset_df = dataset_df[INTERESTED_DATA_FIELD]
        dataset_df.to_csv(DATASET_CACHE_NAME)
    
    else:
        print("Using the cached dataset..")
        dataset_df = pd.read_csv(DATASET_CACHE_NAME)
        dataset_df.set_index("Date", inplace=True)

    return dataset_df
