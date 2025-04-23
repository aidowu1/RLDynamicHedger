import sys, os
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

ROOT_FOLDER = f"{os.path.dirname(os.path.abspath(__file__))}\.."
print(f"Root folder: {ROOT_FOLDER}")
sys.path.append(ROOT_FOLDER)

from src.main.environment.env import DynamicHedgingEnv
import src.main.configs_global as configs
import src.main.configs_rl as configs2
from src.main.rl_algorithms.hyper_parameter_tuning.td3_hyper_parameter_tuning import TD3HyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.ddpg_hyper_parameter_tuning import DDPGHyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.sac_hyper_parameter_tuning import SACHyperParameterTuning
from src.main.rl_algorithms.hyper_parameter_tuning.ppo_hyper_parameter_tuning import PPOHyperParameterTuning
from src.main.market_simulator.parameters import Parameters
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.utility.logging import Logger
from src.main.utility.utils import Helpers

class TuneHyperparametersForRLModels:
    """
    Provides hyper-parameter tuning of the 4 RL hedger models
    """
    def __init__(
            self,
            algorithm_type: RLAgorithmType,
            hedge_type: HedgingType,
            parameters: Parameters = None,
            model_use_case: str = configs2.DEFAULT_MODEL_USE_CASE
    ):
        """
        Constructor
        :param algorithm: RL algorithm type
        :param hedge_type: Hedging type
        :param parameters: Parameters used for tuning
        """
        self._logger = Logger().getLogger()
        self._algorithm_type = algorithm_type
        self._hedge_type = hedge_type
        self._model_use_case = model_use_case
        self._env_name = f"RL Delta Hedger for {self._algorithm_type.name} algorithm type"
        self._logger.info(self._env_name)
        if parameters is None:
            parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
            self._parameters = Parameters(**parameter_settings_data)
        else:
            self._parameters = parameters
        self._env = DynamicHedgingEnv(hedging_type=self._hedge_type, parameters=self._parameters)

    def hyperparameterTuningCyle(self) -> str:
        """
        Executes the hyper-parameter tuning cycle
        :return:
        """
        agent = None
        match self._algorithm_type:
            case RLAgorithmType.ddpg:
                hyper_param_tuner = DDPGHyperParameterTuning(
                    self._env,
                    self._algorithm_type,
                    model_use_case=self._model_use_case
                )
                best_model_path = hyper_param_tuner.run()
            case RLAgorithmType.td3:
                hyper_param_tuner = TD3HyperParameterTuning(
                    self._env,
                    self._algorithm_type,
                    model_use_case=self._model_use_case
                )
                best_model_path = hyper_param_tuner.run()
            case RLAgorithmType.ppo:
                hyper_param_tuner = PPOHyperParameterTuning(
                    self._env,
                    self._algorithm_type,
                    model_use_case=self._model_use_case
                )
                best_model_path = hyper_param_tuner.run()
            case RLAgorithmType.sac:
                hyper_param_tuner = SACHyperParameterTuning(
                    self._env,
                    self._algorithm_type,
                    model_use_case=self._model_use_case
                )
                best_model_path = hyper_param_tuner.run()
            case _:
                raise Exception("Invalid RL algorithm type!!")
        return best_model_path


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
    parser.add_argument(
        "--model_use_case",
        type=str,
        default=configs2.DEFAULT_MODEL_USE_CASE,
        help="Model use case description"
    )

    args = parser.parse_args()
    algorithm_type = Helpers.getEnumType(RLAgorithmType, args.algo_type)
    hedging_type = Helpers.getEnumType(HedgingType, args.hedging_type)
    model_use_case = args.model_use_case
    model = TuneHyperparametersForRLModels(algorithm_type, hedging_type, model_use_case=model_use_case)
    best_model_path = model.hyperparameterTuningCyle()
    print(f"The best hyper-parameter model is saved here: {best_model_path}")

if __name__ == "__main__":
    main()

