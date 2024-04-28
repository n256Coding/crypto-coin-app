import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

from services.main_service import perform_clusterization

st.set_page_config(page_title="Data Grouping", page_icon="📊")

st.markdown("# Data Grouping")
# st.sidebar.header("Cluster Distribution")
st.write(
    """This section shows the final dataset after applying Normalization, PCA and clustering."""
)


# @st.cache_data
def get_dataset():
    clustered_data = perform_clusterization()
    grouped_cluster = clustered_data.groupby("Cluster")["Cluster"].count()
    # grouped_cluster.insert(loc=0, column="Cluster", value=grouped_cluster.index)
    return clustered_data, grouped_cluster


clustered_df, grouped_clusters = get_dataset()

st.write("### Cluster Distribution", clustered_df)
st.write("### Cluster Allocation", grouped_clusters)
