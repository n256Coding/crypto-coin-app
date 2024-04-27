import pandas as pd
import streamlit as st

from config import ARIMA_EVAL_CACHE, EVALUATIONS, LSTM_EVAL_CACHE, MODEL_ARIMA, MODEL_LSTM, MODEL_NEURALPROPHET, MODEL_PROPHET, NEURALPROPHET_EVAL_CACHE, PROPHET_EVAL_CACHE, SELECTED_COINS
from services import arima_forecasting, prophet_service, neuralprophet_service, lstm_service
from util.file_handler import get_temp_file_path, is_file_exits
from services.main_service import get_coin_data

st.set_page_config(page_title="Model Evaluations", page_icon="📈")

st.markdown("# Model Evaluations")
st.sidebar.header("Model Evaluations")
st.write(
    """This section trains and evaluate the models."""
)

tab_arima, tab_prophet, tab_neural_prophet, tab_lstm = st.tabs([MODEL_ARIMA, MODEL_PROPHET, MODEL_NEURALPROPHET, MODEL_LSTM])


coin_data_df = get_coin_data(SELECTED_COINS)

with tab_arima:
    selected_coin_for_evaluation = st.selectbox('Select the coin which you want to evaluate on', SELECTED_COINS, key="arima_eval_coin_select")

    with st.spinner(f'Finding the best {MODEL_ARIMA.lower()} parameters...'):
        best_params = arima_forecasting.find_best_params(coin_data_df, selected_coin_for_evaluation)

    with st.spinner(f'Fitting the {MODEL_ARIMA.lower()} model...'):
        prediction_data, test_data = arima_forecasting.train_model(coin_data_df, best_params, selected_coin_for_evaluation)

    result_df = pd.concat([prediction_data, test_data], axis=1)

    st.line_chart(data=result_df)


    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = arima_forecasting.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_ARIMA] = [rmse_score, mse_score, mae_score]

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, ARIMA_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button('Reset Model Cache', key="arima_cache_clear"):
            st.toast(f"{MODEL_ARIMA} model cache clear triggered!")

with tab_prophet:
    selected_coin_for_evaluation = st.selectbox('Select the coin which you want to evaluate on', SELECTED_COINS, key="prophet_eval_coin_select")
    
    with st.spinner(f"{MODEL_PROPHET.lower()} model is training in progress ..."):
        prediction_data, test_data, full_prediction = prophet_service.train_model(coin_data_df, selected_coin_for_evaluation)

    result_df = pd.concat([prediction_data, test_data], axis=1)
    st.line_chart(data=result_df)

    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = prophet_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_PROPHET] = [rmse_score, mse_score, mae_score]

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, PROPHET_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button('Reset Model Cache', key="prophet_cache_clear"):
            st.toast(f"{MODEL_PROPHET} cache clear triggered!")

with tab_neural_prophet:
    selected_coin_for_evaluation = st.selectbox('Select the coin which you want to evaluate on', SELECTED_COINS, key="neuralprophet_eval_coin_select")

    with st.spinner(f"{MODEL_NEURALPROPHET.lower()} model is training in progress ..."):
        prediction_data, test_data = neuralprophet_service.train_model(coin_data_df, selected_coin_for_evaluation)

    result_df = pd.concat([prediction_data, test_data], axis=1)
    st.line_chart(data=result_df)

    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = neuralprophet_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_NEURALPROPHET] = [rmse_score, mse_score, mae_score]

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, NEURALPROPHET_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button("Reset Model Cache", key="neuralprophet_cache_clear"):
            st.toast(f"{MODEL_NEURALPROPHET} model cache clear triggered!")

with tab_lstm:
    selected_coin_for_evaluation = st.selectbox('Select the coin which you want to evaluate on', SELECTED_COINS, key="lstm_eval_coin_select")

    with st.spinner(f"{MODEL_LSTM.lower()} model is training in progress ..."):
        prediction_data, test_data = lstm_service.train_model(coin_data_df, selected_coin_for_evaluation)

    result_df = pd.concat([prediction_data, test_data], axis=1)
    st.line_chart(data=result_df)

    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = lstm_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_LSTM] = [rmse_score, mse_score, mae_score]

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, LSTM_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button("Reset Model Cache", key="lstm_cache_clear"):
            st.toast(f"{MODEL_LSTM} model cache clear triggered!")
