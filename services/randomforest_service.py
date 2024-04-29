import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from config import RANDOMFOREST_CACHE, RANDOMFOREST_EVAL_CACHE
from constant import MODEL_RANDOMFOREST
from util.file_handler import get_temp_file_path, is_file_exits
from util.forecast_helper import translate_forecast_period

def build_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def create_features(dataset: pd.DataFrame, selected_coin: str):
    temp_dataset_df = dataset[[selected_coin]]

    temp_dataset_df['Lag1'] = dataset[selected_coin].shift(1)
    temp_dataset_df['Lag2'] = dataset[selected_coin].shift(2)
    temp_dataset_df['Lag3'] = dataset[selected_coin].shift(3)
    temp_dataset_df['MA7'] = dataset[selected_coin].rolling(7).mean()
    temp_dataset_df['MA14'] = dataset[selected_coin].rolling(14).mean()

    temp_dataset_df.dropna(inplace=True)

    X = temp_dataset_df[['Lag1', 'Lag2', 'Lag3', 'MA7', 'MA14']]
    y = temp_dataset_df[selected_coin]

    return X, y

def train_full_model(dataset: pd.DataFrame, selected_coin: str, forecast_period: str):

    X, y = create_features(dataset, selected_coin)
    
    cached_model_name = get_temp_file_path(selected_coin, RANDOMFOREST_CACHE)

    if not is_file_exits(cached_model_name):
        model = build_model(X, y)

        with open(cached_model_name, "wb") as f:
            pickle.dump(model, f)

    else:
        with open(cached_model_name, "rb") as f:
            model = pickle.load(f)

        st.toast(f'{MODEL_RANDOMFOREST} model loaded from cache', icon='ℹ️')

    future_prices = []
    input_features = X.tail(1)
    temp_X = X.copy(deep=True)

    period = translate_forecast_period(forecast_period)
    for _ in range(period):
        # Predict the next day price
        next_day_price = model.predict(input_features)[0]
        future_prices.append(next_day_price)

        new_row = input_features.shift(-1)

        new_row["Lag3"] = temp_X["Lag2"].iloc[-1] if np.isnan(new_row["Lag2"].values[0]) else new_row["Lag2"]
        new_row["Lag2"] = temp_X["Lag1"].iloc[-1] if np.isnan(new_row["Lag1"].values[0]) else new_row["Lag1"]
        new_row["Lag1"] = next_day_price

        new_ma7 = np.append(temp_X[-6:].values, next_day_price).mean()
        new_ma14 = np.append(temp_X[-13:].values, next_day_price).mean()
        new_row['MA7'] = new_ma7
        new_row['MA14'] = new_ma14

        temp_X = pd.concat([temp_X, new_row])
        input_features = new_row

    datetime_index = pd.to_datetime(X.index)

    forecast_dataframe = pd.DataFrame({
        "Prediction": future_prices, 
    }, index=pd.date_range(start=datetime_index[-1], end=datetime_index[-1] + pd.Timedelta(days=(period-1))))
    first_row = pd.DataFrame({
        "Prediction": [y[-1]]
    }, index=pd.to_datetime([X.index[-1]]))

    return pd.concat([first_row, forecast_dataframe])


def train_model(dataset: pd.DataFrame, selected_coin: str):
    X, y = create_features(dataset, selected_coin)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

    cached_model_name = get_temp_file_path(selected_coin, RANDOMFOREST_EVAL_CACHE)

    if not is_file_exits(cached_model_name):
        model = build_model(X_train, y_train)

        with open(cached_model_name, "wb") as f:
            pickle.dump(model, f)

    else:
        with open(cached_model_name, "rb") as f:
            model = pickle.load(f)

        st.toast(f'{MODEL_RANDOMFOREST} model loaded from cache', icon='ℹ️')

    # Make predictions for the test set
    forecast = model.predict(X_test)

    forecast_series = pd.DataFrame({
        "Prediction": forecast, 
    }, index=pd.to_datetime(X_test.index))
    test_series = pd.DataFrame({
        "Actual": y_test.values, 
    }, index=pd.to_datetime(X_test.index))

    return forecast_series, test_series, forecast

def get_evaluations(prediction_data, test_data):
    # Calculate root mean squared error
    rmse_score = np.sqrt(mean_squared_error(test_data, prediction_data))
    mse_score = mean_squared_error(test_data, prediction_data)
    mae_score = mean_absolute_error(test_data, prediction_data)

    return rmse_score, mse_score, mae_score
