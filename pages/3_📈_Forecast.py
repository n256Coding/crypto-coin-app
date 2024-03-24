from constants import SELECTED_COINS
from services.main_service import get_coin_data
from services.arima_forecasting import find_best_params, train_model
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Forecast", page_icon="📈")


