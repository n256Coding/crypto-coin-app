from prophet import Prophet
import pickle
import streamlit as st
import pandas as pd
from pandas import DataFrame
import numpy as np

from config import PROPHET_CACHE, PROPHET_EVAL_CACHE
from constant import MODEL_PROPHET
from util.file_handler import get_temp_file_path, is_file_exits
from sklearn.metrics import mean_squared_error, mean_absolute_error

from util.forecast_helper import translate_forecast_period

def build_model(dataframe: pd.DataFrame):
    """Initialize and fit the prophet model"""
    model = Prophet(yearly_seasonality=True)
    model.fit(dataframe)

    return model

def train_full_model(dataset: DataFrame, selected_coin: str, forecast_period: str):

    temp_dataset_df = dataset[selected_coin]
    temp_dataset_df = temp_dataset_df.reset_index()
    temp_dataset_df.columns = ["ds", "y"]

    cached_model_name = get_temp_file_path(selected_coin, PROPHET_CACHE)

    if not is_file_exits(cached_model_name):
        model = build_model(temp_dataset_df)

        with open(cached_model_name, "wb") as f:
            pickle.dump(model, f)

    else:
        with open(cached_model_name, "rb") as f:
            model = pickle.load(f)

        st.toast(f'{MODEL_PROPHET} model loaded from cache', icon='ℹ️')

    period = translate_forecast_period(forecast_period)

    df_future = model.make_future_dataframe(periods=period, include_history=False)

    # Make predictions for the test set
    forecast = model.predict(df_future)

    forecast_dataframe = pd.DataFrame({
        "Prediction": forecast["yhat"].values, 
    }, index=pd.to_datetime(forecast["ds"].values))
    first_row = pd.DataFrame({
        "Prediction": [temp_dataset_df["y"].values[-1]]
    }, index=pd.to_datetime([temp_dataset_df["ds"].values[-1]]))

    return pd.concat([first_row, forecast_dataframe])

def train_model(dataset: DataFrame, selected_coin: str):

    temp_dataset_df = dataset[selected_coin]
    temp_dataset_df = temp_dataset_df.reset_index()
    temp_dataset_df.columns = ["ds", "y"]

    train_size = int(0.8 * len(temp_dataset_df))
    train_df = temp_dataset_df.iloc[:train_size]
    test_df = temp_dataset_df.iloc[train_size:]

    cached_model_name = get_temp_file_path(selected_coin, PROPHET_EVAL_CACHE)

    if not is_file_exits(cached_model_name):
        model = build_model(train_df)

        with open(cached_model_name, "wb") as f:
            pickle.dump(model, f)

    else:
        with open(cached_model_name, "rb") as f:
            model = pickle.load(f)

        st.toast(f'{MODEL_PROPHET} model loaded from cache', icon='ℹ️')

    # Make predictions for the test set
    forecast = model.predict(test_df.drop(columns='y'))

    forecast_series = pd.DataFrame({
        "Prediction": forecast["yhat"].values, 
    }, index=pd.to_datetime(forecast["ds"].values))
    test_series = pd.DataFrame({
        "Actual": test_df["y"].values, 
    }, index=pd.to_datetime(test_df["ds"].values))

    return forecast_series, test_series, forecast

def get_evaluations(prediction_data, test_data):
    # Calculate root mean squared error
    rmse_score = np.sqrt(mean_squared_error(test_data, prediction_data))
    mse_score = mean_squared_error(test_data, prediction_data)
    mae_score = mean_absolute_error(test_data, prediction_data)

    return rmse_score, mse_score, mae_score
