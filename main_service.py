from constants import COIN_COLUMN_NAME
from data_loader import load_data
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def start():
    data = load_data()
    data = transpose(data)
    data = preprocess(data)
    data = dimentionality_reduction(data, None)
    data = clusterize(data)

    return data


def preprocess(data: DataFrame):
    scaler = MinMaxScaler((0, 1))
    standadizer = StandardScaler()
    normalizer = Normalizer(norm='max')
    data_columns = [item for item in data.columns.tolist() if COIN_COLUMN_NAME not in item]
    # data[data_columns] = scaler.fit_transform(data[data_columns])
    # data[data_columns] = standadizer.fit_transform(data[data_columns])
    data[data_columns] = normalizer.fit_transform(data[data_columns])

    return data

def dimentionality_reduction(data: DataFrame, algorithm: str):
    pca = PCA(n_components=10)
    data_columns = [item for item in data.columns.tolist() if COIN_COLUMN_NAME not in item]
    components = pca.fit_transform(data[data_columns])

    pca_df = DataFrame()
    pca_df.index = data.index
    component_names = [f"PC{i+1}" for i in range(components.shape[1])]

    pca_df[component_names] = components
    # pca_df.insert(loc=0, column=COIN_COLUMN_NAME, value=data[COIN_COLUMN_NAME])

    return pca_df

def transpose(data: DataFrame):
    transposed_data = data.transpose(copy=True)
    transposed_data.columns.names = [COIN_COLUMN_NAME]
    transposed_data[COIN_COLUMN_NAME] = transposed_data.index

    return transposed_data

def clusterize(data: DataFrame):
    data_columns = [item for item in data.columns.tolist() if COIN_COLUMN_NAME not in item]

    kmeans = KMeans(n_clusters=4, random_state=0, init="random")
    kmeans.fit_transform(data[data_columns])
    clusteres = kmeans.predict(data[data_columns])

    data.insert(loc=0, column="Cluster", value=clusteres)

    return data
