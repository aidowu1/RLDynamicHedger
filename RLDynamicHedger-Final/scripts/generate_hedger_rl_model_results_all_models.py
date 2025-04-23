import argparse
import os.path
import sys, os
from tqdm import tqdm
from typing import List, Dict

ROOT_FOLDER = f"{os.path.dirname(os.path.abspath(__file__))}\.."
print(f"Root folder: {ROOT_FOLDER}")
sys.path.append(ROOT_FOLDER)

from src.main.environment.env import DynamicHedgingEnv
from src.main.utility.logging import Logger
from src.main.market_simulator.parameters import Parameters
from src.main import configs_global as configs
from src.main import configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.utility.enum_types import PlotType, AggregationType, HedgingType, RLAgorithmType
from src.tests.rl_algorithms.train_evaluate_test.rl_algorithm_test_cycle import RLAlgorithmTestCycle
from src.main.rl_algorithms.train_evaluate_test.base_algorithms import BaseRLAlgorithm
from src.main.rl_algorithms.train_evaluate_test.td3_algorithm import TD3TrainAlgorithm
from src.main.rl_algorithms.train_evaluate_test.ddpg_algorithm import DDPGTrainAlgorithm
from src.main.rl_algorithms.train_evaluate_test.sac_algorithm import SACTrainAlgorithm
from src.main.rl_algorithms.train_evaluate_test.ppo_algorithm import PPOTrainAlgorithm

class GenerateRLModelResults:
    """
    Generates hedging performance results for the 4 RL hedger models
    """
    def __init__(
            self,
            hedge_type: HedgingType,
            is_recompute: bool,
            aggregation_type: AggregationType = AggregationType.mean,
            parameters: Parameters = None,
            model_use_case: str = None,
            algo_types_subset: List[RLAgorithmType] = None,
            test_path_index: int = 0
    ):
        """
        Constructor
        :param hedge_type: Hedging type
        :param is_recompute: Whether the hedger model should be recomputed or not
        :param aggregation_type: Aggregation type
        :param parameters: Parameters for the hedger model
        :param model_use_case: Model use-case for the hedger model
        :param algo_types_subset: Algorithm types subset for the hedger model
        :param test_path_index: Index of the hedger simulation path to be tested
        """
        self._logger = Logger().getLogger()
        self._hedge_type = hedge_type
        self._is_recompute = is_recompute
        self._env_name = f"RL Delta Hedger for 4 algorithm types for the hedging type {self._hedge_type}"
        self._logger.info(self._env_name)
        self._rl_results_path = None
        self._aggregation_type = aggregation_type
        self._model_use_case = model_use_case
        self._algo_types_subset = algo_types_subset
        self._test_path_index = test_path_index

        if parameters:
            self._parameters = parameters
        else:
            parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
            self._parameters = Parameters(**parameter_settings_data)
        self._parameters.hedging_type = self._hedge_type
        self._env = DynamicHedgingEnv(parameters=self._parameters)

    def run(self, is_plot_2_screen: bool = False):
        """
        Executes the RL test cycle and generation of the hedging results
        :param is_plot_2_screen: Flag to indicate plotting to screen
        :return:
        """
        agent_ddpg = DDPGTrainAlgorithm(
                    self._env,
                    hedging_type=self._hedge_type,
                    model_use_case=self._model_use_case
                )
        agent_td3 = TD3TrainAlgorithm(
                    self._env,
                    hedging_type=self._hedge_type,
                    model_use_case=self._model_use_case
                )
        agent_ppo = PPOTrainAlgorithm(
                    self._env,
                    hedging_type=self._hedge_type,
                    model_use_case=self._model_use_case
                )
        agent_sac = SACTrainAlgorithm(
                    self._env,
                    hedging_type=self._hedge_type,
                    model_use_case=self._model_use_case
                )

        agents_map: Dict[RLAgorithmType, BaseRLAlgorithm] = {
            RLAgorithmType.ddpg: agent_ddpg,
            RLAgorithmType.td3: agent_td3,
            RLAgorithmType.ppo: agent_ppo,
            RLAgorithmType.sac: agent_sac
        }
        if self._algo_types_subset:
            agents: List[BaseRLAlgorithm] =[agents_map.get(rl_algorithm_type)
                                            for rl_algorithm_type in self._algo_types_subset]
        else:
            agents: List[BaseRLAlgorithm] = [agents_map.get(rl_algorithm_type)
                                             for rl_algorithm_type in agents_map.keys()]
        self.generateAllRLResults(agents, is_plot_2_screen)

    def generateAllRLResults(
            self,
            agents,
            is_plot_2_screen: bool = False
    ):
        """
        Generates hedging performance results for the 4 RL hedger models
        :param agents: RL agents
        :param is_plot_2_screen: Flag to indicate plotting to screen
        :return:
        """
        for agent in tqdm(agents, desc="Iterating through RL models.."):
            rl_test_cycle = RLAlgorithmTestCycle(
                env=self._env,
                agent=agent,
                model_use_case=self._model_use_case,
                parameters=self._parameters
            )
            self._rl_results_path = rl_test_cycle.results_path
            if self._is_recompute or not os.path.exists(self._rl_results_path):
                results_df = rl_test_cycle.rlAgentTestRunAllCycles()
                print(f"Has recomputed inference and metrics results for agent "
                      f"with a total of {results_df.shape[0]} rows:")


            pnl_df, reward_df, trading_cost_df, delta_df = rl_test_cycle.aggregateResults(
                aggregation_type=self._aggregation_type
            )

            rl_test_cycle.getSinglePathResults(test_path_index=self._test_path_index)

            print(f"Sample of RL test cycle aggregate Pnl results with a total of {pnl_df.shape[0]} rows:")
            print(pnl_df.head(10))
            print(f"Sample of RL test cycle aggregate reward results with a total of {reward_df.shape[0]} rows:")
            print(reward_df.head(10))
            print(
                f"Sample of RL test cycle aggregate Trading Cost results with a total of {trading_cost_df.shape[0]} rows:")
            print(trading_cost_df.head(10))
            print(
                f"Sample of RL test cycle aggregate Delta results with a total of {delta_df.shape[0]} rows:")
            print(delta_df.head(10))

            print("\n\n")
            print("Plotting the RL hedging results..\n\n")
            rl_test_cycle.plotTwoVariableKernelDesityEstimationsAllPlots(is_plot_2_screen=is_plot_2_screen)


def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hedging_type",
        type=str,
        default="gbm",
        help="Hedging strategy type"
    )
    parser.add_argument(
        "--is_recompute",
        type=str,
        default="True",
        help="Flag to indicate the repeat RL model inference and metrics re-calculation"
    )
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="mean",
        help="Result aggregation type i.e. sum or mean"
    )
    parser.add_argument(
        "--model_use_case",
        type=str,
        default=configs2.DEFAULT_MODEL_USE_CASE,
        help="Model use case description"
    )

    args = parser.parse_args()
    hedging_type = Helpers.getEnumType(HedgingType, args.hedging_type)
    is_recompute = Helpers.text2Boolean(args.is_recompute)
    aggregation_type = Helpers.getEnumType(AggregationType, args.aggregation_type)
    model = GenerateRLModelResults(hedging_type, is_recompute, aggregation_type)
    model.run()

if __name__ == "__main__":
    main()
