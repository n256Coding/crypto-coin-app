import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

from services.main_service import perform_clusterization

st.set_page_config(page_title="Data Grouping", page_icon="📊")

st.markdown("# Data Grouping")
st.write(
    """This section shows the final dataset after applying Normalization, PCA and K-Means clustering."""
)

# @st.cache_data
def get_dataset():
    clustered_data = perform_clusterization()

    grouped_cluster_info = clustered_data.groupby("Cluster")["Cluster"].count()
    grouped_cluster = pd.DataFrame({
        "Cluster Number": grouped_cluster_info.index,
        "Number of Data Points Under Cluster": grouped_cluster_info.values
    })

    return clustered_data, grouped_cluster


clustered_df, grouped_clusters = get_dataset()

st.write("### Cluster Distribution", clustered_df)
st.write("### Cluster Allocation", grouped_clusters)
