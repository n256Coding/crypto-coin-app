# Import the library
from pmdarima import auto_arima
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from pandas import DataFrame
import pickle
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse
from datetime import datetime, timedelta

# Ignore harmless warnings
import warnings

from config import ARIMA_CACHE, ARIMA_EVAL_CACHE
from constant import MODEL_ARIMA
from util.file_handler import get_temp_file_path, is_file_exits
from util.forecast_helper import translate_forecast_period
warnings.filterwarnings("ignore")


def find_best_params(dataset, selected_coin):
    """
    Arguments
        dataset: (DataFrame) - dataset
        selected_coin: (str) - name of the selected coin

    Returns: (tuple)
        [0]: arima parameters
        [1]: seasonal arima parameters
    """

    cached_model_name = get_temp_file_path(selected_coin, ARIMA_EVAL_CACHE)

    if is_file_exits(cached_model_name):
        return None, None

    stepwise_fit = auto_arima(dataset[[selected_coin]], 
                            start_p = 1, 
                            start_q = 1, 
                            max_p = 3, 
                            max_q = 3, 
                            m = 12,
                            start_P = 0, 
                            seasonal = True,
                            d = None, 
                            D = 1, 
                            trace = True,
                            error_action ='ignore', # we don't want to know if an order does not work
                            suppress_warnings = True, # we don't want convergence warnings
                            stepwise = True) # set to stepwise

    return stepwise_fit.order, stepwise_fit.seasonal_order

def train_full_model(dataset: DataFrame, selected_coin: str, forecast_period: str):

    temp_dataset_df = dataset[[selected_coin]]

    cached_model_name = get_temp_file_path(selected_coin, ARIMA_CACHE)
    
    if not is_file_exits(cached_model_name):
        result = pm.auto_arima(temp_dataset_df, 
                               error_action='ignore', 
                               suppress_warnings=True, 
                               D=1, 
                               seasonal=True, 
                               m=12)

        with open(cached_model_name, "wb") as f:
            pickle.dump(result, f)

    else:
        with open(cached_model_name, "rb") as f:
            result = pickle.load(f)
        
        st.toast(f'{MODEL_ARIMA} model loaded from cache', icon='ℹ️')

    period = translate_forecast_period(forecast_period)

    predictions, _ = result.predict(n_periods=period, return_conf_int=True)

    forecast_dataframe = pd.DataFrame({
        "Prediction": predictions.values
    }, index=pd.to_datetime(predictions.index))
    first_row = pd.DataFrame({
        "Prediction": [temp_dataset_df.values[-1][0]]
    }, index=pd.to_datetime([temp_dataset_df.index[-1]]))

    return pd.concat([first_row, forecast_dataframe])

def train_model(dataset: DataFrame, selected_coin: str):

    temp_dataset_df = dataset[[selected_coin]]

    # Split data into train / test sets
    train_size = int(0.8 * len(temp_dataset_df))
    train = temp_dataset_df.iloc[:train_size]
    test = temp_dataset_df.iloc[train_size:]

    cached_model_name = get_temp_file_path(selected_coin, ARIMA_EVAL_CACHE)
    
    if not is_file_exits(cached_model_name):
        result = pm.auto_arima(train, 
                               error_action='ignore', 
                               suppress_warnings=True, 
                               D=1, 
                               seasonal=True, 
                               m=12)

        with open(cached_model_name, "wb") as f:
            pickle.dump(result, f)

    else:
        with open(cached_model_name, "rb") as f:
            result = pickle.load(f)
        
        st.toast(f'{MODEL_ARIMA} model loaded from cache', icon='ℹ️')

    start = len(train)
    end = len(train) + len(test) - 1
    
    predictions, *conf_int = result.predict(n_periods=len(test), return_conf_int=True)
    test.index = pd.to_datetime(test.index)

    forecast_dataframe = pd.DataFrame({
        "Prediction": predictions
    }, index=pd.to_datetime(predictions.index))
    test_dataframe = pd.DataFrame({
        "Actual": test.values.reshape(-1),
    }, index=test.index)

    return forecast_dataframe, test_dataframe

def predict(dataset: DataFrame, model_params: tuple, selected_coin: str, num_future_days: int):
    
    cached_model_name = get_temp_file_path(selected_coin, ARIMA_CACHE)
    start_date = datetime.strptime(dataset.index[-1], "%Y-%m-%d")
    end_date = start_date + timedelta(days=num_future_days)
    future_dates = pd.date_range(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if not is_file_exits(cached_model_name):
        model = SARIMAX(dataset[selected_coin], order=model_params[0], seasonal_order=model_params[1])
        model = model.fit()

        with open(cached_model_name, "wb") as f:
            pickle.dump(model, f)

    else:
        with open(cached_model_name, "rb") as f:
            model = pickle.load(f)

    start = len(dataset)
    end = len(dataset) + num_future_days

    prediction = model.predict(start, end, typ='levels').rename("Prediction")
    prediction.index = future_dates

    actual = dataset[selected_coin]
    actual.index = pd.to_datetime(actual.index)

    return prediction, actual

def get_evaluations(prediction_data: DataFrame, test_data: DataFrame):
    # Calculate root mean squared error
    rmse_score = rmse(test_data, prediction_data)
    
    # Calculate mean squared error
    mse_score = mean_squared_error(test_data, prediction_data)

    mae_score = mean_absolute_error(test_data, prediction_data)

    return rmse_score[0], mse_score, mae_score
