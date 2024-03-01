import streamlit as st
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from constants import SELECTED_COINS

from main_service import get_coin_data

st.set_page_config(page_title="EDA", page_icon="📈")

st.markdown("# EDA")
st.sidebar.header("EDA")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

coin_data_df = get_coin_data(SELECTED_COINS)

coins = st.multiselect(
    "Choose coins", SELECTED_COINS, SELECTED_COINS
)

# chart = st.line_chart(last_rows)
st.line_chart(data=coin_data_df[coins])

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

x_data = SELECTED_COINS
y_data = [coin_data_df[coin] for coin in x_data]
colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)']


st.write("## Box Plots")
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

def draw_histogram(index: int, x_data):
    fig = go.Figure()
    fig = px.histogram(coin_data_df, x=x_data[index])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with col1:  
    draw_box_plot(0, x_data, y_data, colors)
with col2:  
    draw_box_plot(1, x_data, y_data, colors)
with col3:  
    draw_box_plot(2, x_data, y_data, colors)
with col4:  
    draw_box_plot(3, x_data, y_data, colors)







st.write("## Histograms")
col1, col2 = st.columns(2)

with col1:  
    draw_histogram(0, x_data)
    draw_histogram(1, x_data)
with col2:  
    draw_histogram(2, x_data)
    draw_histogram(3, x_data)

st.button("Re-run")

# fig.update_layout(
#     title='Points Scored by the Top 9 Scoring NBA Players in 2012',
#     yaxis=dict(
#         autorange=True,
#         showgrid=True,
#         zeroline=True,
#         dtick=5,
#         gridcolor='rgb(255, 255, 255)',
#         gridwidth=1,
#         zerolinecolor='rgb(255, 255, 255)',
#         zerolinewidth=2,
#     ),
#     margin=dict(
#         l=40,
#         r=30,
#         b=80,
#         t=100,
#     ),
#     paper_bgcolor='rgb(243, 243, 243)',
#     plot_bgcolor='rgb(243, 243, 243)',
#     showlegend=False
# )

