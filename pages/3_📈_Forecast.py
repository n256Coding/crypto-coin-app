from constants import ARIMA_CACHE, LSTM_CACHE, NEURALPROPHET_CACHE, ONE_MONTH, ONE_WEEK, PROPHET_CACHE, SELECTED_COINS, THREE_MONTHS
from data_loader import delete_dataset_cache
from services.file_handler import delete_cache_files, get_model_time, get_model_time_by_file_path, get_temp_file_path, is_file_exits
from services.main_service import get_coin_data, reload_dataset_and_train_model
from services.arima_forecasting import find_best_params, train_model
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

tab_arima, tab_prophet, tab_neural_prophet, tab_lstm = st.tabs(["ARIMA", "Prophet", "Neural Prophet", "LSTM"])

with tab_arima:
    model_date = get_model_time(selected_coin_for_forecast, ARIMA_CACHE)
    # if model_date:
    st.caption(f"Model last updated on: {model_date}")
    if st.button("Update the model", 
                    help="This will retrain the model with latest data. Will take few minutes.",
                    key="btn_update_arima_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, ARIMA_CACHE)

    with st.spinner('Model is training in progress ...'):
        forecasted_arima_dataset = arima_forecasting.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
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
    model_date = get_model_time(selected_coin_for_forecast, PROPHET_CACHE)
    # if model_date:
    st.caption(f"Model last updated on: {model_date}")
    if st.button("Update the model", 
                    help="This will retrain the model with latest data. Will take few minutes.",
                    key="btn_update_prophet_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, PROPHET_CACHE)

    with st.spinner('Model is training in progress ...'):
        forecasted_prophet_dataset = prophet_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
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
    model_date = get_model_time(selected_coin_for_forecast, NEURALPROPHET_CACHE)
    if model_date:
        st.caption(f"Model last updated on: {model_date}")
        if st.button("Update the model", 
                     help="This will retrain the model with latest data. Will take few minutes.",
                     key="btn_update_neuralprophet_forecast_model"):
            coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, NEURALPROPHET_CACHE)

    with st.spinner('Model is training in progress ...'):
        forecasted_neuralprophet_dataset = neuralprophet_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
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
    model_date = get_model_time(selected_coin_for_forecast, LSTM_CACHE)
    if model_date:
        st.caption(f"Model last updated on: {model_date}")
        if st.button("Update the model", 
                        help="This will retrain the model with latest data. Will take few minutes.",
                        key="btn_update_lstm_forecast_model"):
            coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, LSTM_CACHE)

    with st.spinner('Model is training in progress ...'):
        forecasted_lstm_dataset = lstm_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
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
