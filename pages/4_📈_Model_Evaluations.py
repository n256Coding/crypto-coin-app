import pandas as pd
import streamlit as st

from constants import ARIMA_CACHE, EVALUATIONS, PROPHET_CACHE, SELECTED_COINS
from services import arima_forecasting, prophet_service
from services.file_handler import get_temp_file_path, is_file_exits
from services.main_service import get_coin_data

st.set_page_config(page_title="Model Evaluations", page_icon="📈")

st.markdown("# Model Evaluations")
st.sidebar.header("Model Evaluations")
st.write(
    """This section trains and evaluate the models."""
)

tab_arima, tab_prophet, tab_neural_prophet, tab_lstm = st.tabs(["ARIMA", "Prophet", "Neural Prophet", "LSTM"])


coin_data_df = get_coin_data(SELECTED_COINS)

with tab_arima:
    selected_coin_for_evaluation = st.selectbox('Select the coin which you want to evaluate on', SELECTED_COINS, key="arima_eval_coin_select")

    with st.spinner('Finding the best arima parameters...'):
        best_params = arima_forecasting.find_best_params(coin_data_df, selected_coin_for_evaluation)

    with st.spinner('Fitting the arima model...'):
        prediction_data, test_data = arima_forecasting.train_model(coin_data_df, best_params, selected_coin_for_evaluation)

    result_df = pd.concat([prediction_data, test_data], axis=1)

    st.line_chart(data=result_df)


    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = arima_forecasting.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS["arima"] = [rmse_score, mse_score, mae_score]

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, ARIMA_CACHE)
    if is_file_exits(cached_model_name):
        if st.button('Reset Model Cache', key="arima_cache_clear"):
            st.toast("Model cache clear triggered!")

with tab_prophet:
    selected_coin_for_evaluation = st.selectbox('Select the coin which you want to evaluate on', SELECTED_COINS, key="prophet_eval_coin_select")
    
    with st.spinner("Prophet model is training in progress ..."):
        prediction_data, test_data = prophet_service.train_model(coin_data_df, selected_coin_for_evaluation)

    result_df = pd.concat([prediction_data, test_data], axis=1)
    st.line_chart(data=result_df)

    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = prophet_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS["prophet"] = [rmse_score, mse_score, mae_score]

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, PROPHET_CACHE)
    if is_file_exits(cached_model_name):
        if st.button('Reset Model Cache', key="prophet_cache_clear"):
            st.toast("Model cache clear triggered!")