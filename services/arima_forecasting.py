# Import the library
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import pickle
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse
 
# Ignore harmless warnings
import warnings

from constants import ARIMA_CACHE
from services.file_handler import get_temp_file_path, is_file_exits
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

    cached_model_name = get_temp_file_path(selected_coin, ARIMA_CACHE)

    if is_file_exits(cached_model_name):
        return None, None

    # Fit auto_arima function to AirPassengers dataset
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

def train_model(dataset, model_params, selected_coin):

    # Split data into train / test sets
    train = dataset.iloc[:len(dataset)-12]
    test = dataset.iloc[len(dataset)-12:] # set one year(12 months) for testing
    cached_model_name = get_temp_file_path(selected_coin, ARIMA_CACHE)
    
    if not is_file_exits(cached_model_name):
        model = SARIMAX(train[selected_coin], order = model_params[0], seasonal_order = model_params[1])
        result = model.fit()

        with open(cached_model_name, "wb") as f:
            pickle.dump(result, f)

    else:
        with open(cached_model_name, "rb") as f:
            result = pickle.load(f)
        
        st.toast('Model loaded from cache', icon='🎉')

    start = len(train)
    end = len(train) + len(test) - 1
    
    # Predictions for one-year against the test set
    predictions = result.predict(start, end, typ = 'levels').rename("Prediction")
    test.index = pd.to_datetime(test.index)

    return predictions, test[selected_coin]

def get_evaluations(prediction_data, test_data):
    # Calculate root mean squared error
    rmse_score = rmse(test_data, prediction_data)
    
    # Calculate mean squared error
    mse_score = mean_squared_error(test_data, prediction_data)

    mae_score = mean_absolute_error(test_data, prediction_data)

    return rmse_score, mse_score, mae_score
