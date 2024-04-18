import pandas as pd
from constants import CLUSTER_COLUMN_NAME, COIN_COLUMN_NAME, MODEL_UPDATED_TIME, SELECTED_COINS
from data_loader import load_clustering_data, load_data
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import streamlit as st

from services.file_handler import delete_cache_files, get_model_time

def perform_clusterization():
    data = load_clustering_data()
    data = transpose(data)
    data = preprocess(data)
    data = dimentionality_reduction(data, None)
    data = clusterize(data)

    return data

def get_coin_data(coins: list = None):
    """Returns the coin dataset
    coins (list): filters the dataset with given coins. If None, nothing is filtered
    """
    data = load_data()

    if coins:
        return data[coins]
    else:
        return data

def preprocess(data: DataFrame):
    scaler = MinMaxScaler((0, 1))
    standadizer = StandardScaler()
    normalizer = Normalizer(norm='max')
    data_columns = [item for item in data.columns.tolist() if COIN_COLUMN_NAME not in item]
    # data[data_columns] = scaler.fit_transform(data[data_columns])
    data[data_columns] = normalizer.fit_transform(data[data_columns])
    data[data_columns] = standadizer.fit_transform(data[data_columns])

    return data

def dimentionality_reduction(data: DataFrame, algorithm: str):
    pca = PCA(n_components=10)
    data_columns = [item for item in data.columns.tolist() if COIN_COLUMN_NAME not in item]
    components = pca.fit_transform(data[data_columns])

    pca_df = DataFrame()
    pca_df.index = data.index
    component_names = [f"PC{i+1}" for i in range(components.shape[1])]

    pca_df[component_names] = components
    # pca_df.insert(loc=0, column=COIN_COLUMN_NAME, value=data[COIN_COLUMN_NAME])

    return pca_df

def transpose(data: DataFrame):
    transposed_data = data.transpose(copy=True)
    transposed_data.columns.names = [COIN_COLUMN_NAME]
    transposed_data[COIN_COLUMN_NAME] = transposed_data.index

    return transposed_data

def clusterize(data: DataFrame):
    data_columns = [item for item in data.columns.tolist() if COIN_COLUMN_NAME not in item]

    kmeans = KMeans(n_clusters=4, random_state=0, init="random")
    kmeans.fit_transform(data[data_columns])
    clusteres = kmeans.predict(data[data_columns])

    data.insert(loc=0, column=CLUSTER_COLUMN_NAME, value=clusteres)

    return data

def reload_dataset_and_train_model(selected_coin_for_forecast, model_type):
    print("Calling data reload .............")
    with st.spinner("Clearing the old dataset"):
        # delete_dataset_cache() TODO: temporarily commented 
        pass
            
    with st.spinner("Deleting the model cache"):
        delete_cache_files(selected_coin_for_forecast, model_type)

    with st.spinner("Downloading the latest data from Yahoo Finance ..."):
        coin_data_df = get_coin_data(SELECTED_COINS)
        
    return coin_data_df

def prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, updated_time_placeholder, 
                          forecasted_dataset, model_cache_type):
    
    model_date = get_model_time(selected_coin_for_forecast, model_cache_type)
    updated_time_placeholder.text(MODEL_UPDATED_TIME.format(model_date))

    temp_df = pd.DataFrame({
        "Current": coin_data_df[selected_coin_for_forecast],
    }, index=coin_data_df.index)

    temp_df1 = pd.DataFrame({
        "Forecasted": forecasted_dataset["Prediction"],
    }, index=forecasted_dataset.index)

    plotted_df = pd.concat([temp_df, temp_df1])
    plotted_df.index = pd.to_datetime(plotted_df.index)

    return plotted_df