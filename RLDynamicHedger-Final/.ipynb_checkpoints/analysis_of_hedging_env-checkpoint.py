from itertools import product
import os
import numpy as np
from numpy.random import choice

#from google.colab import drive

import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import pprint

from src.main.black_scholes_env_cont import BlackScholesEnvCont
# from src.main.environment.env import DynamicHedgingEnv
from src.main.environment.env import DynamicHedgingEnv
from src.main.utils import observations2Dict
import src.main.configs_rl as configs2
from src.main.utility.enum_types import RLAgorithmType
from src.main.rl_algorithms.hyper_parameter_tuning.td3_hyper_parameter_tuning_v1 import TD3HyperParameterTuning as td3_v1
from src.main.rl_algorithms.hyper_parameter_tuning.td3_hyper_parameter_tuning import TD3HyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.ddpg_hyper_parameter_tuning import DDPGHyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.sac_hyper_parameter_tuning import SACHyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.ppo_hyper_parameter_tuning import PPOHyperParameterTuning

SEED: int = 0
SAVE: bool = True # @param ["False", "True"] {type:"raw"}

plt.style.use(['science','no-latex'])
np.random.seed(SEED)

MODEL_PATH = "model/td3_hedger"

def demoRandomHedgerAgent(is_legacy: bool = False):
    """
    Demo random hedger agent
    :param is_legacy: Flag to indicate the use of legacy environment or not
    :return: 
    """
    env = getEnv(is_legacy)

    env.action_space.seed(42)

    observation, info = env.reset(seed=42)
    obs_dict = observations2Dict(observation, is_legacy=is_legacy)
    print("Observations & Info after environment reset:")
    print(f"Observations:")
    pprint.pprint(obs_dict)
    print(f"Info:")
    pprint.pprint(info)
    print(f"{configs2.LINE_BREAK}\n")

    print("RL Training cycle results:")
    for iteration in range(20):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        obs_dict = observations2Dict(observation, is_legacy=is_legacy)
        print(f"Epoch: {iteration}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Observations:")
        pprint.pprint(obs_dict)
        print(f"Info:")
        pprint.pprint(info)
        print(f"{configs2.LINE_BREAK}\n")

        if terminated or truncated:
            observation, info = env.reset()

    print(configs2.LINE_BREAK)

    env.close()


def getEnv(is_legacy):
    s0 = 100
    strike = 100
    expiry = 0.25
    r = 0.0
    mu = 0
    vol = 0.20
    n_steps = configs2.N_STEPS
    env0 = BlackScholesEnvCont(s0, strike, expiry, r, mu, vol, n_steps)
    env1 = DynamicHedgingEnv()
    if is_legacy:
        env = env1
    else:
        env = env0
    return env


def demoTd3HedgerAgent(is_legacy: bool = False):
    """
    Demonstrate the use of SB3 model to create/train_evaluate_test/evaluate/test a RL hedger agent
    Steps include:
        - step 1: create the RL environment for the hedger agent
        - step 2: create the agent model (RL algorithm)
        - step 3: train_evaluate_test the RL agent
        - step 4: extract the vector environment from the RL agent
        - step 5: evaluate the trained RL agent
        - step 6: test the trained RL agent
    :return:
    """
    # step 1: create the RL environment for the hedger agent
    env = getEnv(is_legacy)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # step 2: create the agent model (RL algorithm)
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=configs2.LEARNING_RATE,
        batch_size=configs2.BATCH_SIZE,
        gamma=configs2.GAMMA,
        train_freq=configs2.TRAIN_FREQ,
        policy_kwargs=configs2.POLICY_KWARGS,
        verbose=0
    )

    # step 3: train_evaluate_test the RL agent
    model.learn(
        total_timesteps=configs2.N_STEPS * configs2.N_EPISODES,
        log_interval=10,
        progress_bar=True
    )
    model.save(MODEL_PATH)
    # del model  # remove to demonstrate saving and loading
    # model = TD3.load(MODEL_PATH)
    # model.set_env(env)

    # step 4: train_evaluate_test the RL agent
    vec_env = model.get_env()

    # step 5: evaluate the trained RL agent
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print("Evaluation of trained RL hedger agent:")
    print(f'Mean reward: {mean_reward}\tStandard deviation reward: {std_reward}')

    # step 6: test the trained RL agent

def demoHyperparameterTuningV1(is_legacy_env: bool = False):
    """
    Demo hyperparameter tuning of the RL algos
    :return: None
    """
    env = getEnv(is_legacy_env)
    algo_type = RLAgorithmType.td3
    hyper_param_tuner = td3_v1(env, algo_type)
    hyper_param_tuner.run()

def demoHyperparameterTuningV2(
        rl_algo_type: RLAgorithmType = RLAgorithmType.td3,
        is_legacy_env: bool = False
):
    """
    Demo hyperparameter tuning of the RL algos
    :return: None
    """
    env = getEnv(is_legacy_env)
    match rl_algo_type:
        case RLAgorithmType.td3:
            hyper_param_tuner = TD3HyperParameterTuning(env, rl_algo_type)
            hyper_param_tuner.run()
        case RLAgorithmType.ddpg:
            hyper_param_tuner = DDPGHyperParameterTuning(env, rl_algo_type)
            hyper_param_tuner.run()
        case RLAgorithmType.sac:
            hyper_param_tuner = SACHyperParameterTuning(env, rl_algo_type)
            hyper_param_tuner.run()
        case RLAgorithmType.ppo:
            hyper_param_tuner = PPOHyperParameterTuning(env, rl_algo_type)
            hyper_param_tuner.run()


def demoTd3HedgerAgent2():
    """

    :return:
    """
    check_env(BlackScholesEnvCont(100, 100, 1.0, 0.0, 0.0, 0.2, 252))
    expires = [0.25, 0.5, 1.0, 2.0]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_steps = [63, 126, 252, 504]

    cartesian_product = list(product(expires, sigmas, n_steps))
    indexes = choice(np.asarray(range(len(cartesian_product))), 79, replace=False)
    random_sample = np.asarray(cartesian_product)[indexes]

def getRootPath(current_path: str, levels: int = 2):
    """

    :param current_path:
    :param levels:
    :return:
    """
    tokens = current_path.split("\\")[:-levels]
    first_token = f"{tokens[0]}\\"
    root_path = os.path.join(first_token, *tokens[1:])
    return root_path



if __name__ == "__main__":
    # demoRandomHedgerAgent(is_legacy=False)
    # demoRandomHedgerAgent(is_legacy=True)
    # demoTd3HedgerAgent(is_legacy=False)
    # demoTd3HedgerAgent(is_legacy=True)
    # demoHyperparameterTuning(is_legacy_env=False)
    demoHyperparameterTuningV2(
        is_legacy_env=True,
        rl_algo_type=RLAgorithmType.td3
    )

