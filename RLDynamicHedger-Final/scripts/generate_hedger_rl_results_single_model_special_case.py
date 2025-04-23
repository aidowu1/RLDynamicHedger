import argparse
import os.path
import sys, os
from tqdm import tqdm
from typing import List, Dict, Optional
from dataclasses import dataclass

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

@dataclass
class RLAgentDetails:
    """
    RL agent details class
    """
    algo_type: RLAgorithmType
    hedging_type: HedgingType
    model_path: str
    algo_model: BaseRLAlgorithm
    env: DynamicHedgingEnv

class GenerateRLModelSpecialCaseResults:
    """
    Generates hedging performance results for the 4 RL hedger models
    """
    def __init__(
        self,
        algo_type: RLAgorithmType,
        is_recompute: bool,
        aggregation_type: AggregationType = AggregationType.mean,
        model_path: Optional[str] = None,
        extra_description: Optional[str] = None,
        parameters: Optional[Parameters] = None,
    ):
        """
        Constructor
        :param algorithm: RL algorithm type
        """
        self._logger = Logger().getLogger()
        self._algo_type = algo_type
        self._is_recompute = is_recompute
        self._env_name = (f"RL Delta Hedger for {self._algo_type.name} algorithm type all "
                          f"the 3 simulation hedging scenarios")
        self._logger.info(self._env_name)
        self._rl_results_path = None
        self._aggregation_type = aggregation_type
        self._extra_test_description = extra_description

        if parameters is not None:
            self._parameters = parameters
        else:
            parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
            self._parameters = Parameters(**parameter_settings_data)
            self._parameters.is_test_env = True
        self._env = None
        self._hedging_types = [HedgingType.gbm, HedgingType.sabr, HedgingType.heston]
        self._algo_model_paths = self._createRLAlgoModelPathsMap(model_path)
        self._agents = None
        self._logger.info(f"parameters:")
        self._logger.info(self._parameters)

    def run(self):
        """
        Executes the RL test cycle and generation of the hedging results
        :return:
        """
        self._agents = [self._createRlAgent(hedge_type) for hedge_type in self._hedging_types]
        print(f"agents: {self._agents}")
        self.generateAllRLResults(self._agents)

    def _createRLAlgoModelPathsMap(self, model_path: str) -> Dict[HedgingType, str]:
        """
        Creates a map of the model path per hedging type
        :param model_path: Model path
        :return: Map of hedging type to model path
        """
        map = {
        hedging_type: model_path.format(hedging_type.name, self._algo_type.name)
            for hedging_type in self._hedging_types
        }
        return map

    def _createRlAgent(self, hedge_type) -> RLAgentDetails:
        """
        Creates an RL agent
        :param hedge_type:
        :return:
        """
        self._parameters.hedging_type = hedge_type
        env = DynamicHedgingEnv(parameters=self._parameters)
        model_path = self._algo_model_paths[hedge_type]
        match self._algo_type:
            case RLAgorithmType.ddpg:
                agent = DDPGTrainAlgorithm(
                    env,
                    hedging_type=hedge_type
                )
            case RLAgorithmType.td3:
                agent = TD3TrainAlgorithm(
                    env,
                    hedging_type=hedge_type
                )
            case RLAgorithmType.ppo:
                agent = PPOTrainAlgorithm(
                    env,
                    hedging_type=hedge_type
                )
            case RLAgorithmType.sac:
                agent = SACTrainAlgorithm(
                    env,
                    hedging_type=hedge_type
                )
            case _:
                raise Exception("Invalid RL algorithm type!!")
        rl_agent_details = RLAgentDetails(
            algo_type=self._algo_type,
            hedging_type=hedge_type,
            model_path=model_path,
            algo_model=agent,
            env=env
        )
        return rl_agent_details

    def generateAllRLResults(self, agents: List[RLAgentDetails]):
        """
        Generates hedging performance results for the 4 RL hedger models
        :param agents: RL agents
        :return:
        """
        for agent in tqdm(agents, desc="Iterating through RL models.."):
            rl_test_cycle = RLAlgorithmTestCycle(
                env=agent.env,
                agent=agent.algo_model,
                model_path=agent.model_path,
                extra_description=self._extra_test_description
            )
            self._rl_results_path = rl_test_cycle.results_path
            if self._is_recompute or not os.path.exists(self._rl_results_path):
                results_df = rl_test_cycle.rlAgentTestRunAllCycles()
                print(f"Has recomputed inference and metrics results for agent "
                      f"with a total of {results_df.shape[0]} rows:")


            pnl_df, reward_df, trading_cost_df, delta_df = rl_test_cycle.aggregateResults(
                aggregation_type=self._aggregation_type
            )

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
            rl_test_cycle.plotTwoVariableKernelDesityEstimationsAllPlots()

def main():
    """
    Main entry-point to run the evaluation
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo_type",
        type=str,
        default="td3",
        help="RL algorithm type"
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
        "--model_path",
        type=str,
        default="./model/trained-tuned-models/td3/high_expiry_high_freq_{}_{}",
        help="Saved RL model folder path"
    )
    parser.add_argument(
        "--extra_test_description",
        type=str,
        default="high_expiry_option_high_freq_tests",
        help="Extra description to describe this special evaluation/experiment"
    )

    args = parser.parse_args()
    algo_type = Helpers.getEnumType(RLAgorithmType, args.algo_type)
    is_recompute = Helpers.text2Boolean(args.is_recompute)
    aggregation_type = Helpers.getEnumType(AggregationType, args.aggregation_type)
    model_path = args.model_path
    extra_test_description = args.extra_test_description
    model = GenerateRLModelSpecialCaseResults(
        algo_type=algo_type,
        is_recompute=is_recompute,
        aggregation_type=aggregation_type,
        model_path=model_path,
        extra_description=extra_test_description
    )
    model.run()

if __name__ == "__main__":
    main()
