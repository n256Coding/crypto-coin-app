import streamlit as st
import pandas as pd

from config import SELECTED_COINS
from services.data_grouping_service import perform_clusterization

st.set_page_config(page_title="Data Grouping", page_icon="📊", layout="wide")

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

st.write("### Cluster Distribution")
st.markdown("""This is how the final dataset looks like after applying dimentionality 
            reduction and clustering. Column "Cluster" shows the cluster which the corresponding data point is allocated. 
            "PC1 - PC10" are the principal components which are identified during the dimentionality reduction step.""")
st.table(clustered_df)


col_left, col_right = st.columns(2)

with col_left:
    st.write("### Cluster Allocation")
    grouped_clusters["Selected Currency"] = SELECTED_COINS
    st.table(grouped_clusters)

# with col_right:
#     st.markdown("### Selcted Cryptocurrencies")
#     st.write("These cryptocurrencies has been selected out of the clusters.")
#     st.write(SELECTED_COINS)
