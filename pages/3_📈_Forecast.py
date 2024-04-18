from constants import (ARIMA_CACHE, LSTM_CACHE, MODEL_ARIMA, MODEL_LSTM, MODEL_NEURALPROPHET, MODEL_PROPHET, 
                       MODEL_RETRAIN_WILL_TAKE_TIME, MODEL_TRAINING_IN_PROGRESS, MODEL_UPDATED_TIME, 
                       NEURALPROPHET_CACHE, ONE_MONTH, ONE_WEEK, PROPHET_CACHE, SELECTED_COINS, THREE_MONTHS, 
                       UPDATE_MODEL)
from services.file_handler import get_model_time
from services.main_service import get_coin_data, reload_dataset_and_train_model
from services import lstm_service, neuralprophet_service, prophet_service, arima_forecasting
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Forecast", page_icon="📈")

st.markdown("# Forecast")

coin_data_df = get_coin_data(SELECTED_COINS)

col1, col2 = st.columns(2)
with col1:
    selected_coin_for_forecast = st.selectbox('Select the coin which you want to forcast', SELECTED_COINS, key="forecast_coin_select")

with col2:
    forecast_period = st.selectbox('Period', (ONE_WEEK, ONE_MONTH, THREE_MONTHS))

tab_arima, tab_prophet, tab_neural_prophet, tab_lstm = st.tabs([MODEL_ARIMA, MODEL_PROPHET, MODEL_NEURALPROPHET, MODEL_LSTM])

with tab_arima:
    arima_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_arima_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, ARIMA_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_arima_dataset = arima_forecasting.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
    model_date = get_model_time(selected_coin_for_forecast, ARIMA_CACHE)
    arima_updated_time_placeholder.text(MODEL_UPDATED_TIME.format(model_date))

    temp_arima_df = pd.DataFrame({
        "Current": coin_data_df[selected_coin_for_forecast],
    }, index=coin_data_df.index)

    temp_arima_df1 = pd.DataFrame({
        "Forecasted": forecasted_arima_dataset["Prediction"],
    }, index=forecasted_arima_dataset.index)

    plotted_arima_df = pd.concat([temp_arima_df, temp_arima_df1])
    plotted_arima_df.index = pd.to_datetime(plotted_arima_df.index)

    st.line_chart(data=plotted_arima_df)

with tab_prophet:
    prophet_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_prophet_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, PROPHET_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_prophet_dataset = prophet_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
    model_date = get_model_time(selected_coin_for_forecast, PROPHET_CACHE)
    prophet_updated_time_placeholder.text(MODEL_UPDATED_TIME.format(model_date))

    temp_prophet_df = pd.DataFrame({
        "Current": coin_data_df[selected_coin_for_forecast],
    }, index=coin_data_df.index)

    temp_prophet_df1 = pd.DataFrame({
        "Forecasted": forecasted_prophet_dataset["Prediction"],
    }, index=forecasted_prophet_dataset.index)

    plotted_prophet_df = pd.concat([temp_prophet_df, temp_prophet_df1])
    plotted_prophet_df.index = pd.to_datetime(plotted_prophet_df.index)

    st.line_chart(data=plotted_prophet_df)

with tab_neural_prophet:
    neuralprophet_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_neuralprophet_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, NEURALPROPHET_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_neuralprophet_dataset = neuralprophet_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
    model_date = get_model_time(selected_coin_for_forecast, NEURALPROPHET_CACHE)
    neuralprophet_updated_time_placeholder.text(MODEL_UPDATED_TIME.format(model_date))

    temp_neuralprophet_df = pd.DataFrame({
        "Current": coin_data_df[selected_coin_for_forecast],
    }, index=coin_data_df.index)

    temp_neuralprophet_df1 = pd.DataFrame({
        "Forecasted": forecasted_neuralprophet_dataset["Prediction"],
    }, index=forecasted_neuralprophet_dataset.index)

    plotted_neuralprophet_df = pd.concat([temp_neuralprophet_df, temp_neuralprophet_df1])
    plotted_neuralprophet_df.index = pd.to_datetime(plotted_neuralprophet_df.index)

    st.line_chart(data=plotted_neuralprophet_df)


with tab_lstm:
    lstm_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_lstm_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, LSTM_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_lstm_dataset = lstm_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
    model_date = get_model_time(selected_coin_for_forecast, LSTM_CACHE)
    lstm_updated_time_placeholder.text(MODEL_UPDATED_TIME.format(model_date))

    temp_lstm_df = pd.DataFrame({
        "Current": coin_data_df[selected_coin_for_forecast],
    }, index=coin_data_df.index)

    temp_lstm_df1 = pd.DataFrame({
        "Forecasted": forecasted_lstm_dataset["Prediction"],
    }, index=forecasted_lstm_dataset.index)

    plotted_lstm_df = pd.concat([temp_lstm_df, temp_lstm_df1])
    plotted_lstm_df.index = pd.to_datetime(plotted_lstm_df.index)

    # plotted_df = pd.DataFrame({
    #     "Current": coin_data_df[selected_coin_for_forecast],
    #     "Forecasted": forecasted_dataset["Prediction"]
    # }, index=pd.to_datetime(coin_data_df.index.union(forecasted_dataset.index)))

    st.line_chart(data=plotted_lstm_df)
