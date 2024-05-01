from config import (ARIMA_CACHE, BASE_CURRENCY, LSTM_CACHE, MAX_SUGGESTED_COINS, NEURALPROPHET_CACHE, ONE_MONTH, 
                       ONE_WEEK, PROPHET_CACHE, RANDOMFOREST_CACHE, SELECTED_COINS, THREE_MONTHS, TWO_WEEKS)
from constant import (MODEL_ARIMA, MODEL_LSTM, MODEL_NEURALPROPHET, MODEL_PROPHET, MODEL_RANDOMFOREST, MODEL_RETRAIN_WILL_TAKE_TIME, 
                      MODEL_TRAINING_IN_PROGRESS, SAME, UPDATE_MODEL)
from services.data_loader_service import get_main_dataset, reload_dataset_and_train_model
from util.common_util import get_currency_name
from util.forecast_helper import update_profit_loss_placeholder
from services import arima_service, lstm_service, neuralprophet_service, prophet_service, randomforest_service
import streamlit as st
import plotly.express as px

from util.finance_manager import calculate_expected_return
from util.forecast_helper import (get_most_voted_trade_signal, get_trade_signal, prepare_forecast_dataset, 
                                  update_trade_signal_placeholder)
from util.rss_feed_handler import load_financial_rss_feeds_dict

st.set_page_config(page_title="Trade Assistant - Forcast", page_icon="📈", layout="wide")

st.markdown("# Trade Assistant - Forcast")

coin_data_df = get_main_dataset(SELECTED_COINS)
trade_signals = []

with st.sidebar:
    st.subheader('Latest Financial News Feed', help='News feed is powered by CNBC', divider='rainbow')
    
    with st.expander("Click to hide/see", expanded=True):
        with st.spinner("Loading latest financial news ..."):
            with st.container():
                rss_data = load_financial_rss_feeds_dict()
                
                for item in rss_data:
                    st.markdown(f"{item.get('content')}  \n:blue[{item.get('link')}]")

# Input Panel
col1, col2 = st.columns(2)
with col1:
    selected_coin_for_forecast = st.selectbox('Select the coin which you want to get the assistance with', SELECTED_COINS, key="forecast_coin_select")

with col2:
    forecast_period = st.selectbox('Targetted trading period', (ONE_WEEK, TWO_WEEKS, ONE_MONTH, THREE_MONTHS))


with st.container(border=True):
    # Trade Signal Panel
    st.subheader(f'Trade Signal _:gray[- {selected_coin_for_forecast} in {forecast_period}]_', divider='rainbow')
    col_trade_signal_1, col_trade_signal_2, col_trade_signal_3, col_trade_signal_4, col_trade_signal_5 = st.columns(5)
    with col_trade_signal_1:
        txt_arima_trade_signal = st.empty()
        txt_arima_trade_signal.markdown(f"###### {MODEL_ARIMA}: :blue[Loading ...]")

    with col_trade_signal_2:
        txt_prophet_trade_signal = st.empty()
        txt_prophet_trade_signal.markdown(f"###### {MODEL_PROPHET}: :blue[Loading ...]")

    with col_trade_signal_3:
        txt_randomforest_trade_signal = st.empty()
        txt_randomforest_trade_signal.markdown(f"###### {MODEL_RANDOMFOREST}: :blue[Loading ...]")

    with col_trade_signal_4:
        txt_neuralprophet_trade_signal = st.empty()
        txt_neuralprophet_trade_signal.markdown(f"###### {MODEL_NEURALPROPHET}: :blue[Loading ...]")

    with col_trade_signal_5:
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


with st.container(border=True):
    st.subheader('Forecasted Return Calculator', divider='rainbow')
    col_invest_calc_input_1, col_invest_calc_input_2 = st.columns([.3, .7])
    with col_invest_calc_input_1:
        amount_to_invest = st.number_input(f'Enter desired invesment in {BASE_CURRENCY}', min_value=1)

    st.write(f'''If you invest {amount_to_invest} {BASE_CURRENCY} today on 
             {get_currency_name(selected_coin_for_forecast)}, at the end of {forecast_period.lower()}, your return would be: ''')
    col_invest_calc_1, col_invest_calc_2, col_invest_calc_3, col_invest_calc_4, col_invest_calc_5 = st.columns(5)
    with col_invest_calc_1:
        txt_arima_invest_calc = st.empty()
        txt_arima_invest_calc.markdown(f"###### {MODEL_ARIMA}: :blue[Calculating ...]")

    with col_invest_calc_2:
        txt_prophet_invest_calc = st.empty()
        txt_prophet_invest_calc.markdown(f"###### {MODEL_PROPHET}: :blue[Calculating ...]")

    with col_invest_calc_3:
        txt_randomforest_invest_calc = st.empty()
        txt_randomforest_invest_calc.markdown(f"###### {MODEL_RANDOMFOREST}: :blue[Calculating ...]")

    with col_invest_calc_4:
        txt_neuralprophet_invest_calc = st.empty()
        txt_neuralprophet_invest_calc.markdown(f"###### {MODEL_NEURALPROPHET}: :blue[Calculating ...]")

    with col_invest_calc_5:
        txt_lstm_invest_calc = st.empty()
        txt_lstm_invest_calc.markdown(f"###### {MODEL_LSTM}: :blue[Calculating ...]")


# Tabbed Panel
st.subheader('Detailed View', divider='rainbow')
with st.container(border=False):
    tab_arima, tab_prophet, tab_randomforest, tab_neural_prophet, tab_lstm = st.tabs([MODEL_ARIMA, MODEL_PROPHET, MODEL_RANDOMFOREST, MODEL_NEURALPROPHET, MODEL_LSTM])

    with tab_arima:
        arima_updated_time_placeholder = st.empty()

        if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_arima_forecast_model"):
            coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, ARIMA_CACHE)

        with st.spinner(MODEL_TRAINING_IN_PROGRESS):
            forecasted_arima_dataset = arima_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)

        plotted_arima_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, 
                                                    arima_updated_time_placeholder, forecasted_arima_dataset, ARIMA_CACHE)
        arima_trade_signal = get_trade_signal(coin_data_df, selected_coin_for_forecast, forecasted_arima_dataset)
        update_trade_signal_placeholder(txt_arima_trade_signal, arima_trade_signal, MODEL_ARIMA)
        trade_signals.append(arima_trade_signal)

        arima_expected_return = calculate_expected_return(coin_data_df, selected_coin_for_forecast, forecasted_arima_dataset, amount_to_invest)
        update_profit_loss_placeholder(txt_arima_invest_calc, MODEL_ARIMA, *arima_expected_return)

        fig = px.line(plotted_arima_df, labels={'index': 'Timestamp', 'variable': f'{selected_coin_for_forecast} Trend'})
        st.plotly_chart(fig, use_container_width=True)

    with tab_prophet:
        prophet_updated_time_placeholder = st.empty()

        if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_prophet_forecast_model"):
            coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, PROPHET_CACHE)

        with st.spinner(MODEL_TRAINING_IN_PROGRESS):
            forecasted_prophet_dataset = prophet_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)

        plotted_prophet_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, prophet_updated_time_placeholder, 
                                                      forecasted_prophet_dataset, PROPHET_CACHE)
        prophet_trade_signal = get_trade_signal(coin_data_df, selected_coin_for_forecast, forecasted_prophet_dataset)
        update_trade_signal_placeholder(txt_prophet_trade_signal, prophet_trade_signal, MODEL_PROPHET)
        trade_signals.append(prophet_trade_signal)

        prophet_expected_return = calculate_expected_return(coin_data_df, selected_coin_for_forecast, forecasted_prophet_dataset, amount_to_invest)
        update_profit_loss_placeholder(txt_prophet_invest_calc, MODEL_PROPHET, *prophet_expected_return)

        fig = px.line(plotted_prophet_df, labels={'index': 'Timestamp', 'variable': f'{selected_coin_for_forecast} Trend'})
        st.plotly_chart(fig, use_container_width=True)

    with tab_randomforest:
        randomforest_updated_time_placeholder = st.empty()

        if st.button(UPDATE_MODEL, help=MODEL_RETRAIN_WILL_TAKE_TIME, key="btn_update_randomforest_forecast_model"):
            coin_data_df = reload_dataset_and_train_model(selected_coin_for_forecast, RANDOMFOREST_CACHE)

        with st.spinner(MODEL_TRAINING_IN_PROGRESS):
            forecasted_randomforest_dataset = randomforest_service.train_full_model(coin_data_df, selected_coin_for_forecast, forecast_period)

        plotted_randomforest_df = prepare_forecast_dataset(coin_data_df, selected_coin_for_forecast, randomforest_updated_time_placeholder, 
                                                           forecasted_randomforest_dataset, RANDOMFOREST_CACHE)
        randomforest_trade_signal = get_trade_signal(coin_data_df, selected_coin_for_forecast, forecasted_randomforest_dataset)
        update_trade_signal_placeholder(txt_randomforest_trade_signal, randomforest_trade_signal, MODEL_RANDOMFOREST)
        trade_signals.append(randomforest_trade_signal)

        randomforest_expected_return = calculate_expected_return(coin_data_df, selected_coin_for_forecast, forecasted_randomforest_dataset, amount_to_invest)
        update_profit_loss_placeholder(txt_randomforest_invest_calc, MODEL_RANDOMFOREST, *randomforest_expected_return)

        fig = px.line(plotted_randomforest_df, labels={'index': 'Timestamp', 'variable': f'{selected_coin_for_forecast} Trend'})
        st.plotly_chart(fig, use_container_width=True)

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

        neuralprophet_expected_return = calculate_expected_return(coin_data_df, selected_coin_for_forecast, forecasted_neuralprophet_dataset, amount_to_invest)
        update_profit_loss_placeholder(txt_neuralprophet_invest_calc, MODEL_NEURALPROPHET, *neuralprophet_expected_return)

        fig = px.line(plotted_neuralprophet_df, labels={'index': 'Timestamp', 'variable': f'{selected_coin_for_forecast} Trend'})
        st.plotly_chart(fig, use_container_width=True)

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

        lstm_expected_return = calculate_expected_return(coin_data_df, selected_coin_for_forecast, forecasted_lstm_dataset, amount_to_invest)
        update_profit_loss_placeholder(txt_lstm_invest_calc, MODEL_LSTM, *lstm_expected_return)

        # plotted_df = pd.DataFrame({
        #     "Current": coin_data_df[selected_coin_for_forecast],
        #     "Forecasted": forecasted_dataset["Prediction"]
        # }, index=pd.to_datetime(coin_data_df.index.union(forecasted_dataset.index)))

        fig = px.line(plotted_lstm_df, labels={'index': 'Timestamp', 'variable': f'{selected_coin_for_forecast} Trend'})
        st.plotly_chart(fig, use_container_width=True)

most_voted_trade_signal = get_most_voted_trade_signal(trade_signals)

if most_voted_trade_signal == SAME:
    txt_avg_trade_signal.empty()
    txt_avg_trade_signal_indicator.empty()
else:
    txt_avg_trade_signal.markdown(f"Based on most votes, it is recommended to **{most_voted_trade_signal}** your {selected_coin_for_forecast} based on future {forecast_period.lower()} forecast.")
    txt_avg_trade_signal_indicator.markdown(f"# **:red[{most_voted_trade_signal}]**")


full_coin_data_correlation = get_main_dataset().corr()
correlated_coins_df = full_coin_data_correlation[selected_coin_for_forecast].sort_values(ascending=False)
positive_correlated_coins = correlated_coins_df.iloc[1:MAX_SUGGESTED_COINS].index.to_list()
least_correlated_coins = correlated_coins_df.iloc[-MAX_SUGGESTED_COINS:].index.to_list()

with txt_positive_correlated_coins.container():
    st.markdown(f"Based on our analysis, we can recommend you to **{most_voted_trade_signal}** following coins as well")
    st.markdown(f"##### {', '.join(positive_correlated_coins)}")

with txt_least_correlated_coins.container():
    st.markdown('Based on our analysis, these coins are least correlated coins')
    st.markdown(f"##### {', '.join(least_correlated_coins)}")
