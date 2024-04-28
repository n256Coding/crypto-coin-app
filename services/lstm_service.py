import pickle
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
from keras.saving import load_model
from sklearn.preprocessing import MinMaxScaler

from config import LSTM_CACHE, LSTM_EVAL_CACHE, ONE_MONTH, ONE_WEEK, THREE_MONTHS
from constant import MODEL_LSTM
from util.file_handler import get_temp_file_path, is_file_exits

sequence_length = 10

def train_model(dataset: pd.DataFrame, selected_coin: str):
    temp_dataset_df = dataset[[selected_coin]]
    
    scaler = MinMaxScaler()
    scaled_dataset = scaler.fit_transform(temp_dataset_df)

    sequences, labels = generate_sequence(scaled_dataset, sequence_length)
    train_size = int(0.8 * len(sequences))
    train_x, test_x = sequences[:train_size], sequences[train_size:]
    train_y, test_y = labels[:train_size], labels[train_size:]

    cached_model_name = get_temp_file_path(selected_coin, LSTM_EVAL_CACHE)

    if not is_file_exits(cached_model_name):

        model = compile_model(train_x)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        history = model.fit(
            train_x, train_y,
            epochs=500,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        model.save(cached_model_name + '.keras')

    else:
        model = load_model(cached_model_name)

        st.toast(f'{MODEL_LSTM} model loaded from cache', icon='ℹ️')

    
    test_y_copies = np.repeat(test_y.reshape(-1, 1), test_x.shape[-1], axis=1)
    true_temp = scaler.inverse_transform(test_y_copies)[:,0]
    
    prediction = model.predict(test_x)
    prediction_temp = scaler.inverse_transform(prediction)[:,0]

    forecast_series = pd.DataFrame({ 
        "Prediction": prediction_temp, 
    }, index=pd.to_datetime(dataset.index[-len(test_x):]))
    test_series = pd.DataFrame({
        "Actual": true_temp[-len(test_x):], 
    }, index=pd.to_datetime(dataset.index[-len(test_x):]))

    return forecast_series, test_series

def train_full_model(dataset: pd.DataFrame, selected_coin: str, forecast_period: str):
    temp_dataset_df = dataset[[selected_coin]]

    scaler = MinMaxScaler()
    scaled_dataset = scaler.fit_transform(temp_dataset_df)
    
    sequences, labels = generate_sequence(scaled_dataset, sequence_length)

    cached_model_name = get_temp_file_path(selected_coin, LSTM_CACHE)
    cached_scaler_name = get_temp_file_path(selected_coin, LSTM_CACHE, is_scaler=True)

    if not is_file_exits(cached_model_name):
        model = compile_model(sequences)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = model.fit(
            sequences, labels,
            epochs=500,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        model.save(cached_model_name + '.keras')
        with open(cached_scaler_name, "wb") as f:
            pickle.dump(scaler, f)
    
    else:
        model = load_model(cached_model_name)
        with open(cached_scaler_name, "rb") as f:
            scaler = pickle.load(f)

        st.toast(f'{MODEL_LSTM} model loaded from cache', icon='ℹ️')

    predicted_sequence = []
    start_sequence = scaled_dataset[-sequence_length:]
    start_sequence = start_sequence[np.newaxis, :]

    if forecast_period == ONE_WEEK:
        period = 7

    elif forecast_period == ONE_MONTH:
        period = 30

    elif forecast_period == THREE_MONTHS:
        period = 90

    for i in range(period):
        prediction = model.predict(start_sequence)
        prediction = scaler.inverse_transform(prediction)[0:]
        predicted_sequence.append(prediction[0][0])
        start_sequence = np.append(start_sequence, [prediction], axis=1)
        start_sequence = start_sequence[:, 1:]

    datetime_index = pd.to_datetime(dataset.index)
    
    forecast_dataframe = pd.DataFrame({
            "Prediction": predicted_sequence, 
    }, index=pd.date_range(start=datetime_index[-1], end=datetime_index[-1] + pd.Timedelta(days=(period-1))))
    first_row = pd.DataFrame({
        "Prediction": [temp_dataset_df[selected_coin].values[-1]]
    }, index=pd.to_datetime([temp_dataset_df.index[-1]]))

    return pd.concat([first_row, forecast_dataframe])


def compile_model(train_x):
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.6))

    model.add(LSTM(32))
    model.add(Dropout(0.7))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def get_evaluations(prediction_data, test_data):
    rmse_score = np.sqrt(mean_squared_error(test_data, prediction_data))
    mse_score = mean_squared_error(test_data, prediction_data)
    mae_score = mean_absolute_error(test_data, prediction_data)

    return rmse_score, mse_score, mae_score

def generate_sequence(scaled_dataset, sequence_length: int) -> tuple[np.array, np.array]:
    sequences = []
    labels = []

    for i in range(len(scaled_dataset) - sequence_length):
        seq = scaled_dataset[i:i+sequence_length]
        label = scaled_dataset[i+sequence_length][0]
        sequences.append(seq)
        labels.append(label)

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels