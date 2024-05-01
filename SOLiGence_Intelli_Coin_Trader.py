import streamlit as st


st.set_page_config(page_title="Intelli Coin Trader", page_icon="📊",)


st.write("# SOLiGence Intelli Coin Trader")
st.sidebar.success("Select any option from above")

st.markdown(
    """
    This application is an innovative tool designed for investors. 
    It uses five distinct forecasting models to predict the potential profit or loss from an investment over a given period. 
    By providing these insights, the application empowers users to make informed investment decisions based on the projected returns. 
    It's not just about tracking investments, but about optimizing them for the future. 

    ### To begin, click on any option below
"""
)

st.page_link('pages/1_📊_Data_Grouping.py', label='Data Grouping', icon='📊')
st.page_link('pages/2_📈_Correlated_Currencies.py', label='Correlated Currencies', icon='📈')
st.page_link('pages/3_📈_EDA.py', label='Explanatory Data Analysis', icon='📈')
st.page_link('pages/4_📈_Model_Evaluations.py', label='Model Evaluations', icon='📈')
st.page_link('pages/5_📈_Moving_Average.py', label='Data Moving Average', icon='📈')
st.page_link('pages/6_📈_Trade_Assistant_(Forecast).py', label='Trade Assistant (Forecast)', icon='📈')

st.markdown('# ')
st.markdown('# ')

with st.container(border=True):
    st.markdown('_:gray[Senarath Pathiranalage Nishan Sajeewee Wickramasinghe]_')
    st.markdown('_:gray[(Q102091977)]_')
    st.markdown('_:gray[Applied AI in Business]_')
    st.markdown('_:gray[AE2]_')


