# CURRENCY_LIST = "CRO-USD SHIB-USD ETH-USD HBAR-USD XLM-USD LTC-USD MATIC-USD AVAX-USD XMR-USD ICP-USD ETC-USD RUNE-USD USDC-USD SOL-USD XRP-USD DOT-USD BTC-USD USDT-USD ATOM-USD ADA-USD VET-USD BCH-USD DAI-USD FIL-USD LINK-USD BNB-USD DOGE-USD TRX-USD STX4847-USD UNI7083-USD"
CURRENCY_LIST = "ADA-GBP ANKR-GBP ANT-GBP ATOM-GBP BAT-GBP BCH-GBP BNB-GBP BSV-GBP BTC-GBP BTG-GBP CRO-GBP DASH-GBP DCR-GBP DOGE-GBP ENJ-GBP EOS-GBP ETC-GBP ETH-GBP FET-GBP FIL-GBP FLUX-GBP FTM-GBP GAS-GBP GLM-GBP GNO-GBP ICX-GBP IOTX-GBP LINK-GBP LRC-GBP LTC-GBP"
SELECTED_COINS = ["ADA-GBP", "DOGE-GBP", "FET-GBP", "LTC-GBP"]
INTERESTED_DATA_FIELD = "Close"

DATASET_PERIOD = "1y"
DATASET_CACHE_NAME = "currency-dataset.csv"

CLUSTER_DATASET_PERIOD = "1y"
CLUSTER_DATASET_CACHE_NAME = "cluster-currency-dataset.csv"

TEMP_DIR_NAME = "temp"

COIN_SUGGESTION_THRESHOULD = 0.7
MAX_SUGGESTED_COINS = 4

CURRENT_DATA_SHOWN_DAYS = 183

# Dataframe column names
COIN_COLUMN_NAME = "Coin"
CLUSTER_COLUMN_NAME = "Cluster"


# Model Cache Names
ARIMA_EVAL_CACHE = "arima_eval_cache"
PROPHET_EVAL_CACHE = "prophet_eval_cache"
NEURALPROPHET_EVAL_CACHE = "neuralprophet_eval_cache"
LSTM_EVAL_CACHE = "lstm_eval_cache"

ARIMA_CACHE = "arima_cache"
PROPHET_CACHE = "prophet_cache"
NEURALPROPHET_CACHE = "neuralprophet_cache"
LSTM_CACHE = "lstm_cache"

EVALUATIONS = {}


# Forecast Periods
ONE_WEEK = '7 Days'
ONE_MONTH = '30 Days'
THREE_MONTHS = '3 Months'


