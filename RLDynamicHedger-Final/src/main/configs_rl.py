import os

from src.main import configs_global as configs
from src.main.market_simulator.parameters import Parameters
from src.main.utility.utils import Helpers

# Get simulation parameters
PARAMETER_SETTINGS_DATA= Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
SIMULATION_PARAMETRS = Parameters(**PARAMETER_SETTINGS_DATA)

# General configs
LINE_BREAK = "==========" * 5
SEED = 100

# General hyper-parameter settings
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
GAMMA = 0.9999
TRAIN_FREQ = 100
CHECKPOINT_FREQ = 10000
POLICY_KWARGS = dict(
    net_arch=dict(pi=[64, 64], qf=[64, 64])
)
N_EPISODES = 1e2
N_STEPS = 252
N_TEST_CYCLE_EPISODES = 1e3


# Hyper-parameter tuning configs
N_TRIALS = 10
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = N_STEPS
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_TRAIN_EPISODES = 100
N_EVAL_EPISODES = 10
# N_TUNING_TRAIN_STEPS = 3E3
N_TUNING_TRAIN_STEPS = int(SIMULATION_PARAMETRS.n_time_steps/SIMULATION_PARAMETRS.trading_frequency) * N_TRAIN_EPISODES
TUNING_TIMEOUT = 2 * 60 * 60

# Hyperparameter tuning results path
IS_USE_HYPER_PARAMETER_TUNING = True
HYPER_PARAMETER_RESULT_FOLDER = "model/hyper_parameter"
HYPER_PARAMETER_TENSORBOARD_FOLDER = "model/tensorboard"
os.makedirs(HYPER_PARAMETER_RESULT_FOLDER, exist_ok=True)
os.makedirs(HYPER_PARAMETER_TENSORBOARD_FOLDER, exist_ok=True)
HYPER_PARAMETER_RESULT_PATH = "tuning_results.csv"
HYPER_PARAMETER_HISTORY_PATH = "_tuning_optimization_history.html"
HYPER_PARAMETER_IMPORTANCE_PATH = "tuning_param_importance.html"
HYPER_PARAMETER_REWARD_CURVE_PATH = HYPER_PARAMETER_RESULT_FOLDER + "/{0}_reward_curve.png"
HYPER_PARAMETER_REWARD_CURVE_DATA_PATH = HYPER_PARAMETER_RESULT_FOLDER + "/{0}_reward_curve.csv"
HYPER_PARAMETER_BEST_VALUES = "tuning_best_values.pkl"
HYPER_PARAMETER_BEST_MODEL_PATH = HYPER_PARAMETER_RESULT_FOLDER + "_{0}_best_model"
TUNED_MODEL_PATH = "model/trained-tuned-models/{0}/{1}/"
TUNED_PARAMETER_FILE_NAME = "tuning_best_values.pkl"
TUNED_TEST_USE_CASE = "low_expiry_{0}_{1}"
DEFAULT_MODEL_USE_CASE = "low_expiry"

HYPER_PARAMETER_NOISE_TYPE = "noise_type"
HYPER_PARAMETER_NOISE_STD = "noise_std"
HYPER_PARAMETER_LR_SCHEDULE = "lr_schedule"
HYPER_PARAMETER_LOG_STD_INIT = "log_std_init"
HYPER_PARAMETER_ORTHO_INIT = "ortho_init"
HYPER_PARAMETER_NET_ARCH = "net_arch"
HYPER_PARAMETER_ACTIVATION_FN = "activation_fn"
HYPER_PARAMETER_POLICY_KWARGS = "policy_kwargs"

# Reward curve (plot) configurations
PLOT_FILTER_WINDOW = 50

# Hedging metric plot configurations
SMALL_SIZE = 10
FONT = {'color':  'black',
        'weight': 'normal',
        'size': SMALL_SIZE
       }

# RL problem domain configuration
RL_PROBLEM_TITLE = "RL Delta Hedger"

# Default path index
DEFAULT_PATH_INDEX = -1





