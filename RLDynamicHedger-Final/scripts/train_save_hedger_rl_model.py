import sys
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import sys, os

ROOT_FOLDER = f"{os.path.dirname(os.path.abspath(__file__))}\.."
print(f"Root folder: {ROOT_FOLDER}")
sys.path.append(ROOT_FOLDER)

from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.rl_algorithms.train_evaluate_test.ddpg_algorithm import DDPGTrainAlgorithm
from src.main.rl_algorithms.train_evaluate_test.td3_algorithm import TD3TrainAlgorithm
from src.main.rl_algorithms.train_evaluate_test.ppo_algorithm import PPOTrainAlgorithm
from src.main.rl_algorithms.train_evaluate_test.sac_algorithm import SACTrainAlgorithm
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.utility.logging import Logger
from src.main.utility.utils import Helpers

class TrainRLModels:
    """
    Trains the 4 RL hedger models
    """
    def __init__(
            self,
            algorithm_type: RLAgorithmType,
            hedge_type: HedgingType,
    ):
        """
        Constructor
        :param algorithm: RL algorithm type
        """
        self._logger = Logger().getLogger()
        self._algorithm_type = algorithm_type
        self._hedge_type = hedge_type
        self._env_name = f"RL Delta Hedger for {self._algorithm_type.name} algorithm type"
        self._logger.info(self._env_name)
        parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
        self._parameters = Parameters(**parameter_settings_data)
        self._env = DynamicHedgingEnv()

    def runTrainingCycle(self):
        """
        Executes the training cycle
        :return:
        """
        agent = None
        match self._algorithm_type:
            case RLAgorithmType.ddpg:
                agent = DDPGTrainAlgorithm(
                    self._env,
                    total_timesteps=configs2.N_STEPS * configs2.N_TUNING_TRAIN_STEPS,
                    hedging_type=self._hedge_type
                )
            case RLAgorithmType.td3:
                agent = TD3TrainAlgorithm(
                    self._env,
                    total_timesteps=configs2.N_STEPS * configs2.N_TUNING_TRAIN_STEPS,
                    hedging_type=self._hedge_type
                )
            case RLAgorithmType.ppo:
                agent = PPOTrainAlgorithm(
                    self._env,
                    total_timesteps=configs2.N_STEPS * configs2.N_TUNING_TRAIN_STEPS,
                    hedging_type=self._hedge_type
                )
            case RLAgorithmType.sac:
                agent = SACTrainAlgorithm(
                    self._env,
                    total_timesteps=configs2.N_STEPS * configs2.N_TUNING_TRAIN_STEPS,
                    hedging_type=self._hedge_type
                )
            case _:
                raise Exception("Invalid RL algorithm type!!")

        agent.train()
        self.evaluateTrainedModel(agent.trained_model)

    def evaluateTrainedModel(self, model):
        """
        Evaluates a trained model
        :param model: Model
        :return: None
        """
        mean_reward, std_reward = evaluate_policy(
            model,
            self._env,
            n_eval_episodes=10,
            deterministic=True)
        self._logger.info(f"mean_reward={mean_reward:.6f} +/- {std_reward}")


def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo_type",
        type=str,
        default="ddpg",
        help="RL algorithm type"
    )
    parser.add_argument(
        "--hedging_type",
        type=str,
        default="gbm",
        help="Hedging strategy type"
    )
    args = parser.parse_args()
    algorithm_type = Helpers.getEnumType(RLAgorithmType, args.algo_type)
    hedging_type = Helpers.getEnumType(HedgingType, args.hedging_type)
    model = TrainRLModels(algorithm_type, hedging_type)
    model.runTrainingCycle()

if __name__ == "__main__":
    main()

