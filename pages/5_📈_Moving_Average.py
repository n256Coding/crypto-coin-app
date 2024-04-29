from config import SELECTED_COINS
from services.data_loader_service import (get_main_dataset)

import streamlit as st
import plotly.express as px

from util.common_util import get_currency_name


st.set_page_config(page_title="Moving Average", page_icon="📈", layout="wide")

st.markdown("# Moving Average")

coin_data_df = get_main_dataset(SELECTED_COINS)

# Input Panel
col1, col2 = st.columns(2)
with col1:
    selected_coin_for_forecast = st.selectbox('Select the coin which you want to get the assistance with', SELECTED_COINS, key="forecast_coin_select")

with col2:
    forecast_period = st.slider('Enter the period', min_value=1, max_value=100, step=2)


st.subheader(f'Moving Average for {get_currency_name(selected_coin_for_forecast)}', divider='rainbow')

with st.spinner("Chart is being generated ..."):
    filtered_df = coin_data_df[[selected_coin_for_forecast]]
    filtered_df[f'{forecast_period} days moving average'] = filtered_df.rolling(window=forecast_period).mean()

    # st.line_chart(filtered_df)
    fig = px.line(filtered_df, labels={'index': 'Timestamp'})
    st.plotly_chart(fig, use_container_width=True)
