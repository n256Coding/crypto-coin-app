from config import (ARIMA_CACHE, COIN_SUGGESTION_THRESHOULD, LSTM_CACHE, MAX_SUGGESTED_COINS, NEURALPROPHET_CACHE, ONE_MONTH, 
                       ONE_WEEK, PROPHET_CACHE, SELECTED_COINS, THREE_MONTHS)
from constant import MODEL_ARIMA, MODEL_LSTM, MODEL_NEURALPROPHET, MODEL_PROPHET, MODEL_RETRAIN_WILL_TAKE_TIME, MODEL_TRAINING_IN_PROGRESS, UPDATE_MODEL
from services.main_service import (get_coin_data, get_most_voted_trade_signal, get_trade_signal, 
                                   prepare_forecast_dataset, reload_dataset_and_train_model, 
                                   update_trade_signal_placeholder)
from services import lstm_service, neuralprophet_service, prophet_service, arima_forecasting
import streamlit as st
import plotly.express as px

from util.rss_feed_handler import load_financial_rss_feeds, load_financial_rss_feeds_dict

st.set_page_config(page_title="Trade Assistant - Forcast", page_icon="📈", layout="wide")

st.markdown("# Trade Assistant - Forcast")

coin_data_df = get_coin_data(SELECTED_COINS)
trade_signals = []

with st.sidebar:
    # st.write('## Latest Financial News Feed')
    st.subheader('Latest Financial News Feed', help='News feed is powered by CNBC', divider='rainbow')
    with st.expander("Click to see"):
        with st.spinner("Loading latest financial news ..."):

            with st.container():
                rss_data = load_financial_rss_feeds_dict()
                for item in rss_data:
                    st.write(item.get('content'))
                    st.write(item.get('link'))
                    st.divider()
        # st.table(rss_data)


# Input Panel
col1, col2 = st.columns(2)
with col1:
    selected_coin_for_forecast = st.selectbox('Select the coin which you want to get the assitance with', SELECTED_COINS, key="forecast_coin_select")

with col2:
    forecast_period = st.selectbox('Targetted trading period', (ONE_WEEK, ONE_MONTH, THREE_MONTHS))


# Trade Signal Panel
st.subheader(f'Trade Signal _:gray[- {selected_coin_for_forecast} in {forecast_period}]_', divider='rainbow')
col_trade_signal_1, col_trade_signal_2, col_trade_signal_3, col_trade_signal_4 = st.columns(4)
with col_trade_signal_1:
    txt_arima_trade_signal = st.empty()
    txt_arima_trade_signal.markdown(f"###### {MODEL_ARIMA}: :blue[Loading ...]")

with col_trade_signal_2:
    txt_prophet_trade_signal = st.empty()
    txt_prophet_trade_signal.markdown(f"###### {MODEL_PROPHET}: :blue[Loading ...]")

with col_trade_signal_3:
    txt_neuralprophet_trade_signal = st.empty()
    txt_neuralprophet_trade_signal.markdown(f"###### {MODEL_NEURALPROPHET}: :blue[Loading ...]")

with col_trade_signal_4:
    txt_lstm_trade_signal = st.empty()
    txt_lstm_trade_signal.markdown(f"###### {MODEL_LSTM}: :blue[Loading ...]")

col_avg_trade_signal_left, col_avg_trade_signal_right = st.columns([0.8, 0.2])
with col_avg_trade_signal_left:
    txt_avg_trade_signal = st.empty()
    txt_avg_trade_signal.markdown("Determing the most voted decision ...")

with col_avg_trade_signal_right:
    txt_avg_trade_signal_indicator = st.empty()
    txt_avg_trade_signal_indicator.markdown("# :hourglass_flowing_sand:")


# This section shows the seggested coins which are correlated with the selected coin
col_trade_suggestion_left, col_trade_suggestion_right = st.columns(2)
with col_trade_suggestion_left:
    # shows the positive correlated coins
    txt_positive_correlated_coins = st.empty()
    txt_positive_correlated_coins.markdown("Please wait, We are preparing suggestions for you ...")

with col_trade_suggestion_right:
    # shows the negative correlated coins
    txt_least_correlated_coins = st.empty()
    txt_least_correlated_coins.markdown("Please wait, We are preparing suggestions for you ...")

# Tabbed Panel
st.subheader('Detailed View', divider='rainbow')
tab_arima, tab_prophet, tab_neural_prophet, tab_lstm = st.tabs([MODEL_ARIMA, MODEL_PROPHET, MODEL_NEURALPROPHET, MODEL_LSTM])

with tab_arima:
    arima_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_arima_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, ARIMA_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_arima_dataset = arima_forecasting.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)

    plotted_arima_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, 
                                                arima_updated_time_placeholder, forecasted_arima_dataset, ARIMA_CACHE)
    arima_trade_signal = get_trade_signal(coin_data_df, selected_coin_for_forecast, forecasted_arima_dataset)
    update_trade_signal_placeholder(txt_arima_trade_signal, arima_trade_signal, MODEL_ARIMA)
    trade_signals.append(arima_trade_signal)


    fig = px.line(plotted_arima_df, labels={'index': 'Timestamp'})
    st.plotly_chart(fig, use_container_width=True)
    # st.line_chart(data=plotted_arima_df)

with tab_prophet:
    prophet_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_prophet_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, PROPHET_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_prophet_dataset = prophet_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)

    plotted_prophet_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, 
                                                  prophet_updated_time_placeholder, forecasted_prophet_dataset, PROPHET_CACHE)
    prophet_trade_signal = get_trade_signal(coin_data_df, selected_coin_for_forecast, forecasted_prophet_dataset)
    update_trade_signal_placeholder(txt_prophet_trade_signal, prophet_trade_signal, MODEL_PROPHET)
    trade_signals.append(prophet_trade_signal)

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
    neuralprophet_trade_signal = get_trade_signal(coin_data_df, selected_coin_for_forecast, forecasted_neuralprophet_dataset)
    update_trade_signal_placeholder(txt_neuralprophet_trade_signal, neuralprophet_trade_signal, MODEL_NEURALPROPHET)
    trade_signals.append(neuralprophet_trade_signal)

    st.line_chart(data=plotted_neuralprophet_df)

with tab_lstm:
    lstm_updated_time_placeholder = st.empty()

    if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_lstm_forecast_model"):
        coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, LSTM_CACHE)

    with st.spinner(MODEL_TRAINING_IN_PROGRESS):
        forecasted_lstm_dataset = lstm_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)
    
    plotted_lstm_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, 
                                               lstm_updated_time_placeholder, forecasted_lstm_dataset, LSTM_CACHE)
    lstm_trade_signal = get_trade_signal(coin_data_df, selected_coin_for_forecast, forecasted_lstm_dataset)
    update_trade_signal_placeholder(txt_lstm_trade_signal, lstm_trade_signal, MODEL_LSTM)
    trade_signals.append(lstm_trade_signal)

    # plotted_df = pd.DataFrame({
    #     "Current": coin_data_df[selected_coin_for_forecast],
    #     "Forecasted": forecasted_dataset["Prediction"]
    # }, index=pd.to_datetime(coin_data_df.index.union(forecasted_dataset.index)))

    st.line_chart(data=plotted_lstm_df)

most_voted_trade_signal = get_most_voted_trade_signal(trade_signals)

txt_avg_trade_signal.markdown(f"Based on most votes, it is recommended to **{most_voted_trade_signal}** your {selected_coin_for_forecast} based on future {forecast_period} forecast.")
txt_avg_trade_signal_indicator.markdown(f"# **:red[{most_voted_trade_signal}]**")


full_coin_data_correlation = get_coin_data().corr()
correlated_coins_df = (full_coin_data_correlation[selected_coin_for_forecast]
                        .where(lambda x: x > COIN_SUGGESTION_THRESHOULD)
                        .dropna()
                        .sort_values(ascending=False))
positive_correlated_coins = correlated_coins_df.iloc[1:MAX_SUGGESTED_COINS].index.to_list()
least_correlated_coins = correlated_coins_df.iloc[-MAX_SUGGESTED_COINS:].index.to_list()

with txt_positive_correlated_coins.container():
    st.markdown(f"Based on our analysis, we can recommend you to **{most_voted_trade_signal}** following coins as well")
    st.markdown(f"##### {', '.join(positive_correlated_coins)}")

with txt_least_correlated_coins.container():
    st.markdown('Based on our analysis, these coins are least correlated coins')
    st.markdown(f"##### {', '.join(least_correlated_coins)}")
