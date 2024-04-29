from collections import Counter
import streamlit as st
from config import BASE_CURRENCY, CURRENT_DATA_SHOWN_DAYS
from constant import BUY, LOSS, MODEL_UPDATED_TIME, SAME, SELL, TRAINED_MODELS_ARE_CACHED
from util.file_handler import get_model_time


import pandas as pd
from pandas import DataFrame


def prepare_forecast_dataset(coin_data_df: DataFrame, selected_coin_for_forecast: str, updated_time_placeholder,
                             forecasted_dataset: DataFrame, model_cache_type: str) -> DataFrame:

    model_date = get_model_time(selected_coin_for_forecast, model_cache_type)
    updated_time_placeholder.text(MODEL_UPDATED_TIME.format(model_date), help=TRAINED_MODELS_ARE_CACHED)

    temp_df = pd.DataFrame({
        "Current": coin_data_df[selected_coin_for_forecast].iloc[-CURRENT_DATA_SHOWN_DAYS:],
    }, index=coin_data_df.index)

    temp_df1 = pd.DataFrame({
        "Forecasted": forecasted_dataset["Prediction"],
    }, index=forecasted_dataset.index)

    plotted_df = pd.concat([temp_df, temp_df1])
    plotted_df.index = pd.to_datetime(plotted_df.index)

    return plotted_df


def get_trade_signal(coin_data_df: DataFrame, selected_coin_for_forecast: str, forecasted_dataset: DataFrame) -> str:
    today_price = coin_data_df[selected_coin_for_forecast].iloc[-1]
    forecasted_price = forecasted_dataset.iloc[-1].values[0]

    if today_price > forecasted_price:
        return SELL

    elif today_price < forecasted_price:
        return BUY

    else:
        return SAME


def update_trade_signal_placeholder(placeholder, trade_signal: str, model_name: str):
    with placeholder.container():
        st.markdown(f"###### {model_name}: :red[{trade_signal}]")


def get_most_voted_trade_signal(trade_signal_list: list) -> str:
    trade_signal_count = Counter(trade_signal_list)
    return trade_signal_count.most_common(1)[0][0]


def update_profit_loss_placeholder(placeholder, model_name: str, *args):
    expected_return, income_loss, indicator = args

    with placeholder.container():
        st.markdown(f'###### {model_name}')
        st.markdown(f'''Expected return: **{expected_return} {BASE_CURRENCY}**  
                    {indicator}: **:{"red" if indicator == LOSS else "green"}[{income_loss} {BASE_CURRENCY}]**''')