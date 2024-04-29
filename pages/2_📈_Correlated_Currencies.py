import streamlit as st
import plotly.express as px

from config import MAX_SUGGESTED_COINS, SELECTED_COINS
from services.data_loader_service import get_main_dataset
from util.common_util import get_currency_name

st.set_page_config(page_title="Correlated Currencies", page_icon="📈", layout="wide")

st.markdown("# Correlated Currencies")

coin_data_df = get_main_dataset()

selected_coin = st.selectbox('Select the cryptocurrency', SELECTED_COINS, key="coin_select")

correlated_df = coin_data_df.corr(numeric_only=True)[[selected_coin]].sort_values(by=selected_coin, ascending=False)
top_positive = correlated_df.iloc[1:MAX_SUGGESTED_COINS+1]
least_correlated = correlated_df.iloc[-MAX_SUGGESTED_COINS:].sort_values(by=selected_coin, ascending=True)

col_1_left, col_1_right = st.columns(2)

with col_1_left:
    with st.container(border=True):
        st.subheader(f'Top {MAX_SUGGESTED_COINS} Positively Correlated with {get_currency_name(selected_coin)}', divider='rainbow')

        st.markdown(f"This will shows the top most {MAX_SUGGESTED_COINS} cryptocurrencies that are **positively** correlated with **{get_currency_name(selected_coin)}**")

        fig = px.imshow(top_positive.T, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)


with col_1_right:
    with st.container(border=True):
        st.subheader(f'Top {MAX_SUGGESTED_COINS} Negatively* Correlated with {get_currency_name(selected_coin)}', divider='rainbow')
        st.markdown(f"""This will shows the top most {MAX_SUGGESTED_COINS} cryptocurrencies that are **negatively\*** correlated with **{get_currency_name(selected_coin)}**.""")
        
        with st.spinner("Chart is being generated ..."):
            fig = px.imshow(least_correlated.T, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("\* *:blue[If there will be no negatively correlated values, this will show the least correlated cryptocurrencies]*.")