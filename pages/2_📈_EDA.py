import streamlit as st
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from matplotlib.dates import WeekdayLocator, MonthLocator
from constants import CLUSTER_COLUMN_NAME, INTERESTED_DATA_FIELD, SELECTED_COINS
from pandas.plotting import lag_plot

from main_service import get_coin_data, perform_clusterization

st.set_page_config(page_title="EDA", page_icon="📈")

st.markdown("# EDA")
st.sidebar.header("EDA")
st.write(
    """This section performs various explanatory data analysis (EDA) on the dataset."""
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

coin_data_df = get_coin_data(SELECTED_COINS)

coins = st.multiselect(
    "Choose coins", SELECTED_COINS, SELECTED_COINS
)

st.write("## Explore")
# chart = st.line_chart(last_rows)
st.line_chart(data=coin_data_df[coins])

st.write("## Correlation with Lags")
col1, col2 = st.columns(2)

with col1:
    selected_coin_for_lag_plot = st.selectbox('Select the coin which you want to find the lag plot', SELECTED_COINS)

with col2:
    lags = st.slider('How many lags?', 1, 120, 2)

fig, ax = plt.subplots(figsize=(8, 8))
# Draw a lag plot
lag_plot(coin_data_df[selected_coin_for_lag_plot], ax=ax, lag=lags)
ax.set_title(f'Lags of coin: {selected_coin_for_lag_plot}')
st.pyplot(fig)

# https://plotly.com/python/box-plots/
# df = px.data.tips()
# fig = px.box(coin_data_df, y="total_bill")
# fig.show()

# for i in range(1, 101):
#     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#     status_text.text("%i%% Complete" % i)
#     chart.add_rows(new_rows)
#     progress_bar.progress(i)
#     last_rows = new_rows
#     time.sleep(0.05)

progress_bar.empty()


st.write("## Seasonal Decomposition")

selected_coin_for_seasonality = st.selectbox('Select the coin which you want to find the seasonal decomposition', SELECTED_COINS)
# Perform seasonal decomposition
result = seasonal_decompose(coin_data_df[[selected_coin_for_seasonality]], model='additive', period=30)  # Adjust the period according to your dataset's seasonality

# Plot the original, trend, seasonal, and residual components using plt.subplots()
fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

# Original time series
axs[0].plot(coin_data_df[["BTC-USD"]], label='Original')
axs[0].legend()

# Trend component
axs[1].plot(result.trend, label='Trend')
axs[1].legend()

# Seasonal component
axs[2].plot(result.seasonal, label='Seasonal')
axs[2].legend()

# Residual component
axs[3].plot(result.resid, label='Residual')
axs[3].legend()

for ax in axs:
    ax.xaxis.set_major_locator(MonthLocator())
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

st.pyplot(fig)

x_data = SELECTED_COINS
y_data = [coin_data_df[coin] for coin in x_data]
colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)']


st.write("## Investigate")

tab1, tab2, tab3 = st.tabs(["Box Plots", "Histograms", "Correlation"])

with tab1:

    st.write("#### Box Plots")
    col1, col2, col3, col4 = st.columns(4)

    def draw_box_plot(index: int, x_data, y_data, color):
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=y_data[index],
            name=x_data[index],
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=color[index],
            marker_size=2,
            line_width=1)
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with col1:  
        draw_box_plot(0, x_data, y_data, colors)
    with col2:  
        draw_box_plot(1, x_data, y_data, colors)
    with col3:  
        draw_box_plot(2, x_data, y_data, colors)
    with col4:  
        draw_box_plot(3, x_data, y_data, colors)

with tab2:
    def draw_histogram(index: int, x_data):
        fig = go.Figure()
        fig = px.histogram(coin_data_df, x=x_data[index])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.write("#### Histograms")
    col1, col2 = st.columns(2)

    with col1:  
        draw_histogram(0, x_data)
        draw_histogram(1, x_data)
    with col2:  
        draw_histogram(2, x_data)
        draw_histogram(3, x_data)

with tab3:
    clustered_data = perform_clusterization()
    group1_coins = list(clustered_data[clustered_data[CLUSTER_COLUMN_NAME] == 0].index)
    group2_coins = list(clustered_data[clustered_data[CLUSTER_COLUMN_NAME] == 1].index)
    group3_coins = list(clustered_data[clustered_data[CLUSTER_COLUMN_NAME] == 2].index)
    group4_coins = list(clustered_data[clustered_data[CLUSTER_COLUMN_NAME] == 3].index)
    
    group1_selected_coins = st.multiselect(
        "Choose coins from group 1", group1_coins, group1_coins
    )
    group2_selected_coins = st.multiselect(
        "Choose coins from group 2", group2_coins, group2_coins
    )
    group3_selected_coins = st.multiselect(
        "Choose coins from group 3", group3_coins, group3_coins
    )
    group4_selected_coins = st.multiselect(
        "Choose coins from group 4", group4_coins, group4_coins
    )

    corr_coin_data = get_coin_data(group1_selected_coins + group2_selected_coins + group3_selected_coins + group4_selected_coins)
    correlation_matrix = corr_coin_data.corr()

    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig, theme="streamlit")

