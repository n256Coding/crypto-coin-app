from constants import (ARIMA_CACHE, LSTM_CACHE, MODEL_ARIMA, MODEL_LSTM, MODEL_NEURALPROPHET, MODEL_PROPHET, 
                       MODEL_RETRAIN_WILL_TAKE_TIME, MODEL_TRAINING_IN_PROGRESS, NEURALPROPHET_CACHE, 
                       ONE_MONTH, ONE_WEEK, PROPHET_CACHE, SELECTED_COINS, THREE_MONTHS, UPDATE_MODEL)
from services.main_service import get_coin_data, prepare_forecast_dataset, reload_dataset_and_train_model
from services import lstm_service, neuralprophet_service, prophet_service, arima_forecasting
import streamlit as st

st.set_page_config(page_title="Forecast", page_icon="📈")

st.markdown("# Forecast")

coin_data_df = get_coin_data(SELECTED_COINS)


# Input Panel
col1, col2 = st.columns(2)
with col1:
    selected_coin_for_forecast = st.selectbox('Select the coin which you want to forcast', SELECTED_COINS, key="forecast_coin_select")

with col2:
    forecast_period = st.selectbox('Period', (ONE_WEEK, ONE_MONTH, THREE_MONTHS))


# Trade Signal Panel
trade_signal_panel = st.empty()


# Tabbed Panel
tab_arima, tab_prophet, tab_neural_prophet, tab_lstm = st.tabs([MODEL_ARIMA, MODEL_PROPHET, MODEL_NEURALPROPHET, MODEL_LSTM])

with tab_arima:
    arima_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_arima_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, ARIMA_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_arima_dataset = arima_forecasting.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)

    plotted_arima_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, 
                                                arima_updated_time_placeholder, forecasted_arima_dataset, ARIMA_CACHE)

    st.line_chart(data=plotted_arima_df)

with tab_prophet:
    prophet_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_prophet_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, PROPHET_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_prophet_dataset = prophet_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)

    plotted_prophet_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, 
                                                  prophet_updated_time_placeholder, forecasted_prophet_dataset, PROPHET_CACHE)

    st.line_chart(data=plotted_prophet_df)

with tab_neural_prophet:
    neuralprophet_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_neuralprophet_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, NEURALPROPHET_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_neuralprophet_dataset = neuralprophet_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)

    plotted_neuralprophet_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, 
                                                        neuralprophet_updated_time_placeholder, forecasted_neuralprophet_dataset, 
                                                        NEURALPROPHET_CACHE)

    st.line_chart(data=plotted_neuralprophet_df)

with tab_lstm:
    lstm_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_lstm_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, LSTM_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_lstm_dataset = lstm_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
    plotted_lstm_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, 
                                               lstm_updated_time_placeholder, forecasted_lstm_dataset, LSTM_CACHE)

    # plotted_df = pd.DataFrame({
    #     "Current": coin_data_df[selected_coin_for_forecast],
    #     "Forecasted": forecasted_dataset["Prediction"]
    # }, index=pd.to_datetime(coin_data_df.index.union(forecasted_dataset.index)))

    st.line_chart(data=plotted_lstm_df)
