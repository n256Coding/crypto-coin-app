import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

from main_service import perform_clusterization

st.set_page_config(page_title="Cluster Distribution", page_icon="📊")

st.markdown("# Cluster Distribution")
# st.sidebar.header("Cluster Distribution")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)


@st.cache_data
def get_dataset():
    clustered_data = perform_clusterization()
    grouped_cluster = clustered_data.groupby("Cluster")["Cluster"].count()
    # grouped_cluster.insert(loc=0, column="Cluster", value=grouped_cluster.index)
    return clustered_data, grouped_cluster


try:
    clustered_df, grouped_clusters = get_dataset()
    # countries = st.multiselect(
    #     "Choose countries", list(df.index), ["China", "United States of America"]
    # )
    # if not countries:
    #     st.error("Please select at least one country.")
    # else:
    # data = df.loc[countries]
    # data /= 1000000.0
    st.write("### Cluster Distribution", clustered_df)
    st.write("### Cluster Allocation", grouped_clusters)


        # data = data.T.reset_index()
        # data = pd.melt(data, id_vars=["index"]).rename(
        #     columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        # )
        # chart = (
        #     alt.Chart(data)
        #     .mark_area(opacity=0.3)
        #     .encode(
        #         x="year:T",
        #         y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
        #         color="Region:N",
        #     )
        # )
        # st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
