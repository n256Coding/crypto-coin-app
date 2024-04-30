import pandas as pd
import streamlit as st
import plotly.express as px

from config import ARIMA_EVAL_CACHE, EVALUATIONS, LSTM_EVAL_CACHE, NEURALPROPHET_EVAL_CACHE, PROPHET_EVAL_CACHE, RANDOMFOREST_EVAL_CACHE, SELECTED_COINS
from constant import MODEL_ARIMA, MODEL_LSTM, MODEL_NEURALPROPHET, MODEL_PROPHET, MODEL_RANDOMFOREST, MODEL_RETRAIN_WILL_TAKE_TIME, UPDATE_MODEL
from services.data_loader_service import get_main_dataset
from services import arima_service, prophet_service, neuralprophet_service, lstm_service, randomforest_service
from util.file_handler import get_temp_file_path, is_file_exits
from services.data_loader_service import reload_dataset_and_train_model

st.set_page_config(page_title="Model Evaluations", page_icon="📈", layout="wide")

st.markdown("# Model Evaluations")
st.sidebar.header("Model Evaluations")
st.write(
    """This section trains and evaluate the models."""
)

selected_coin_for_evaluation = st.selectbox('Select the coin which you want to evaluate on', SELECTED_COINS, key="eval_coin_select")

plt_overall_prediction = st.empty()
plt_overall_prediction.markdown("**:blue[Overall evaluation is preparing in progress :hourglass_flowing_sand: ...]**")
overall_prediction_df = pd.DataFrame()


st.subheader("Modelwise Results", divider="rainbow")
st.write("Go though each tab to view details of each model seperately.")

tab_arima, tab_prophet, tab_random_forest, tab_neural_prophet, tab_lstm = st.tabs([MODEL_ARIMA, MODEL_PROPHET, MODEL_RANDOMFOREST, MODEL_NEURALPROPHET, MODEL_LSTM])


coin_data_df = get_main_dataset(SELECTED_COINS)

with tab_arima:

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, ARIMA_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button(UPDATE_MODEL, key="arima_cache_clear", help=MODEL_RETRAIN_WILL_TAKE_TIME):
            reload_dataset_and_train_model(selected_coin_for_evaluation, ARIMA_EVAL_CACHE)
            st.toast(f"{MODEL_ARIMA} model cache clear triggered!")

    with st.spinner(f'Fitting the {MODEL_ARIMA.lower()} model...'):
        prediction_data, test_data = arima_service.train_model(coin_data_df, selected_coin_for_evaluation)

    overall_prediction_df = test_data.copy(deep=True)
    overall_prediction_df[f'{MODEL_ARIMA} prediction'] = prediction_data["Prediction"]

    result_df = pd.concat([prediction_data, test_data], axis=1)

    st.line_chart(data=result_df)


    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = arima_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_ARIMA] = [rmse_score, mse_score, mae_score]

with tab_prophet:

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, PROPHET_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button(UPDATE_MODEL, key="prophet_cache_clear", help=MODEL_RETRAIN_WILL_TAKE_TIME):
            reload_dataset_and_train_model(selected_coin_for_evaluation, PROPHET_EVAL_CACHE)
            st.toast(f"{MODEL_PROPHET} cache clear triggered!")
    
    with st.spinner(f"{MODEL_PROPHET.lower()} model is training in progress ..."):
        prediction_data, test_data, full_prediction = prophet_service.train_model(coin_data_df, selected_coin_for_evaluation)

    overall_prediction_df[f'{MODEL_PROPHET} prediction'] = prediction_data["Prediction"]

    result_df = pd.concat([prediction_data, test_data], axis=1)
    st.line_chart(data=result_df)


    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = prophet_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_PROPHET] = [rmse_score, mse_score, mae_score]

with tab_random_forest:

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, RANDOMFOREST_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button(UPDATE_MODEL, key="randomforest_cache_clear", help=MODEL_RETRAIN_WILL_TAKE_TIME):
            reload_dataset_and_train_model(selected_coin_for_evaluation, RANDOMFOREST_EVAL_CACHE)
            st.toast(f"{MODEL_RANDOMFOREST} cache clear triggered!")
    
    with st.spinner(f"{MODEL_RANDOMFOREST.lower()} model is training in progress ..."):
        prediction_data, test_data, full_prediction = randomforest_service.train_model(coin_data_df, selected_coin_for_evaluation)

    overall_prediction_df[f'{MODEL_RANDOMFOREST} prediction'] = prediction_data["Prediction"]

    result_df = pd.concat([prediction_data, test_data], axis=1)
    st.line_chart(data=result_df)


    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = randomforest_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_RANDOMFOREST] = [rmse_score, mse_score, mae_score]

with tab_neural_prophet:

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, NEURALPROPHET_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button(UPDATE_MODEL, key="neuralprophet_cache_clear", help=MODEL_RETRAIN_WILL_TAKE_TIME):
            reload_dataset_and_train_model(selected_coin_for_evaluation, NEURALPROPHET_EVAL_CACHE)
            st.toast(f"{MODEL_NEURALPROPHET} model cache clear triggered!")

    with st.spinner(f"{MODEL_NEURALPROPHET.lower()} model is training in progress ..."):
        prediction_data, test_data = neuralprophet_service.train_model(coin_data_df, selected_coin_for_evaluation)

    overall_prediction_df[f'{MODEL_NEURALPROPHET} prediction'] = prediction_data["Prediction"]

    result_df = pd.concat([prediction_data, test_data], axis=1)
    st.line_chart(data=result_df)

    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = neuralprophet_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_NEURALPROPHET] = [rmse_score, mse_score, mae_score]

with tab_lstm:

    cached_model_name = get_temp_file_path(selected_coin_for_evaluation, LSTM_EVAL_CACHE)
    if is_file_exits(cached_model_name):
        if st.button(UPDATE_MODEL, key="lstm_cache_clear", help=MODEL_RETRAIN_WILL_TAKE_TIME):
            reload_dataset_and_train_model(selected_coin_for_evaluation, LSTM_EVAL_CACHE)
            st.toast(f"{MODEL_LSTM} model cache clear triggered!")

    with st.spinner(f"{MODEL_LSTM.lower()} model is training in progress ..."):
        prediction_data, test_data = lstm_service.train_model(coin_data_df, selected_coin_for_evaluation)

    overall_prediction_df[f'{MODEL_LSTM} prediction'] = prediction_data["Prediction"]

    result_df = pd.concat([prediction_data, test_data], axis=1)
    st.line_chart(data=result_df)

    with st.spinner("Evaluating ..."):
        rmse_score, mse_score, mae_score = lstm_service.get_evaluations(prediction_data, test_data)
        evaluation_df = pd.DataFrame({
            "Score": [rmse_score, mse_score, mae_score],
        }, index=["RMSE", "MSE", "MAE"])
    st.table(evaluation_df)

    EVALUATIONS[MODEL_LSTM] = [rmse_score, mse_score, mae_score]


with plt_overall_prediction.container():
    col_overall_pred_left, col_overall_pred_right = st.columns([0.3, 0.7])

    with col_overall_pred_left:
        st.markdown("# ")
        st.subheader('Overall Results', divider='rainbow')
        overall_evaluation_df = pd.DataFrame(EVALUATIONS, index=["RMSE", "MSE", "MAE"])
        st.table(overall_evaluation_df.T)
    
    with col_overall_pred_right:
        fig = px.line(overall_prediction_df, labels={'index': 'Timestamp', 'variable': 'Trend', 'value': selected_coin_for_evaluation})
        st.plotly_chart(fig, use_container_width=True)
