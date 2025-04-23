from datetime import datetime
import os

PROJECT_ROOT_PATH = "RLDynamicHedger-Final"

# Settings configuration
SETTINGS_FOLDER = "src/main/settings"
DEFAULT_SETTINGS_NAME = "default_settings"

# LOGGING Configurations
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - : %(message)s in %(pathname)s:%(lineno)d'
CURRENT_DATE = datetime.today().strftime("%d%b%Y")
LOG_FILE = f"RLDynamicHedger_{CURRENT_DATE}.log"
LOG_FOLDER = "logs"
LOG_PATH = os.path.join(LOG_FOLDER, LOG_FILE)

# Simulation datasets path/file configurations
DATA_ROOT_FOLDER = "data"
GBM_ASSET_SIMULATION_FILE_NAME = "asset_price_gbm_simulation.csv"
GBM_OPTION_PRICE_SIMULATION_FILE_NAME = "option_price_gbm_simulation.csv"
GBM_OPTION_DELTA_SIMULATION_FILE_NAME = "option_delta_gbm_simulation.csv"
SABR_ASSET_SIMULATION_FILE_NAME = "asset_price_sabr_simulation.csv"
SABR_OPTION_PRICE_SIMULATION_FILE_NAME = "option_price_sabr_simulation.csv"
SABR_OPTION_DELTA_SIMULATION_FILE_NAME = "option_delta_sabr_simulation.csv"
HESTON_ASSET_SIMULATION_FILE_NAME = "asset_price_heston_simulation.csv"
HESTON_OPTION_PRICE_SIMULATION_FILE_NAME = "option_price_heston_simulation.csv"
HESTON_OPTION_DELTA_SIMULATION_FILE_NAME = "option_delta_heston_simulation.csv"
DELIMITER = ","

# Miscellaneous configurations
NEW_LINE = "\n"
LINE_DIVIDER = "==========" * 5

# Random seed setting
RANDOM_SEED = 100

# Option maturity
OPTION_MATURITY_MONTHS = [1, 3, 6, 12]

# Option trading frequency
OPTION_TRADING_FREQUENCY = [0.25, 1, 2, 3, 4, 5]

# Window size for price normalization
PRICE_NORMALIZATION_WINDOW = 200

# RL environment normalization configurations
PRICE_MEMORY_FIRST_SIZE = 1
ZERO_INDEX_STATE_VALUE = 0
FIRST_INDEX_STATE_VALUE = 1
SECOND__INDEX_STATE_VALUE = 2
NORMALIZATION_MEAN = 100
NORMALIZATION_STDDEV = 1
HEDGING_INTERVAL_MINUTES = 60

# MLP Network configurations
ACTIVATION_FUNCTION_NAME = "Sigmoid"

# Hedging frequency details
N_MONTHS_PER_YEAR = 12





