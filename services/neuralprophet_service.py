import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet

from constants import MODEL_NEURALPROPHET, NEURALPROPHET_CACHE, NEURALPROPHET_EVAL_CACHE, ONE_MONTH, ONE_WEEK, THREE_MONTHS
from services.file_handler import get_temp_file_path, is_file_exits

def train_full_model(dataset, selected_coin, forecast_period: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    temp_dataset_df = dataset[selected_coin]
    temp_dataset_df = temp_dataset_df.reset_index()
    temp_dataset_df.columns = ["ds", "y"]

    cached_model_name = get_temp_file_path(selected_coin, NEURALPROPHET_CACHE)
    if not is_file_exits(cached_model_name):
        m = NeuralProphet(n_lags=0, yearly_seasonality=True)
        m.fit(temp_dataset_df, freq='D', epochs=500, batch_size=50)

        with open(cached_model_name, "wb") as f:
            pickle.dump(m, f)
    
    else:
        with open(cached_model_name, "rb") as f:
            m = pickle.load(f)
            m.restore_trainer()

        st.toast(f"{MODEL_NEURALPROPHET} model loaded from cache", icon='ℹ️')
    
    if forecast_period == ONE_WEEK:
        period = 7

    elif forecast_period == ONE_MONTH:
        period = 30

    elif forecast_period == THREE_MONTHS:
        period = 90

    future = m.make_future_dataframe(temp_dataset_df, periods=period)
    
    forecast = m.predict(future)

    forecast_series = pd.DataFrame({
        "Prediction": forecast["yhat1"].values, 
    }, index=pd.to_datetime(forecast["ds"].values))

    return forecast_series

def train_model(dataset, selected_coin) -> tuple[pd.DataFrame, pd.DataFrame]:

    temp_dataset_df = dataset[selected_coin]
    temp_dataset_df = temp_dataset_df.reset_index()
    temp_dataset_df.columns = ["ds", "y"]

    

    # train_df, test_df = m.split_df(temp_dataset_df, freq="MS", valid_p=0.2)
    train_size = int(0.8 * len(temp_dataset_df))
    train_df = temp_dataset_df.iloc[:train_size]
    test_df = temp_dataset_df.iloc[train_size:]

    cached_model_name = get_temp_file_path(selected_coin, NEURALPROPHET_EVAL_CACHE)
    if not is_file_exits(cached_model_name):
        m = NeuralProphet(yearly_seasonality=True)
        m.fit(train_df, freq='D', epochs=1000)

        with open(cached_model_name, "wb") as f:
            pickle.dump(m, f)
    
    else:
        with open(cached_model_name, "rb") as f:
            m = pickle.load(f)
            m.restore_trainer()

        st.toast(f"{MODEL_NEURALPROPHET} model loaded from cache", icon='ℹ️')
    
    forecast = m.predict(test_df)

    forecast_series = pd.DataFrame({
        "Prediction": forecast["yhat1"].values, 
    }, index=pd.to_datetime(forecast["ds"].values))
    test_series = pd.DataFrame({
        "Actual": test_df["y"].values, 
    }, index=pd.to_datetime(test_df["ds"].values))

    return forecast_series, test_series

def get_evaluations(prediction_data, test_data):
    rmse_score = np.sqrt(mean_squared_error(test_data, prediction_data))
    mse_score = mean_squared_error(test_data, prediction_data)
    mae_score = mean_absolute_error(test_data, prediction_data)

    return rmse_score, mse_score, mae_score
