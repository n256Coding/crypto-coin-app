from prophet import Prophet
import pickle
import streamlit as st
import pandas as pd
import numpy as np

from constants import PROPHET_CACHE
from services.file_handler import get_temp_file_path, is_file_exits
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model(dataset, selected_coin):

    temp_dataset_df = dataset[selected_coin]
    temp_dataset_df = temp_dataset_df.reset_index()
    temp_dataset_df.columns = ["ds", "y"]

    train_size = int(0.8 * len(temp_dataset_df))
    train_df = temp_dataset_df.iloc[:train_size]
    test_df = temp_dataset_df.iloc[train_size:]

    cached_model_name = get_temp_file_path(selected_coin, PROPHET_CACHE)
    if not is_file_exits(cached_model_name):
        # Initialize and fit the model
        model = Prophet()
        model.fit(train_df)

        with open(cached_model_name, "wb") as f:
            pickle.dump(model, f)

    else:
        with open(cached_model_name, "rb") as f:
            model = pickle.load(f)

        st.toast('Model loaded from cache', icon='🎉')

    # Make predictions for the test set
    forecast = model.predict(test_df.drop(columns='y'))

    forecast_series = pd.DataFrame({
        "Prediction": forecast["yhat"].values, 
    }, index=pd.to_datetime(forecast["ds"].values))
    test_series = pd.DataFrame({
        "Actual": test_df["y"].values, 
    }, index=pd.to_datetime(test_df["ds"].values))

    return forecast_series, test_series

def get_evaluations(prediction_data, test_data):
    # Calculate root mean squared error
    rmse_score = np.sqrt(mean_squared_error(test_data, prediction_data))
    mse_score = mean_squared_error(test_data, prediction_data)
    mae_score = mean_absolute_error(test_data, prediction_data)

    return rmse_score, mse_score, mae_score
