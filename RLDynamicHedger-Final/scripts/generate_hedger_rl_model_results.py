import argparse
import sys, os

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
        self._env = DynamicHedgingEnv(hedging_type=self._hedge_type)

    def run(self):
        """
        Executes the RL test cycle and generation of the hedging results
        :return:
        """
        match self._algorithm_type:
            case RLAgorithmType.ddpg:
                agent = DDPGTrainAlgorithm(
                    self._env,
                    hedging_type=self._hedge_type
                )
            case RLAgorithmType.td3:
                agent = TD3TrainAlgorithm(
                    self._env,
                    hedging_type=self._hedge_type
                )
            case RLAgorithmType.ppo:
                agent = PPOTrainAlgorithm(
                    self._env,
                    hedging_type=self._hedge_type
                )
            case RLAgorithmType.sac:
                agent = SACTrainAlgorithm(
                    self._env,
                    hedging_type=self._hedge_type
                )
            case _:
                raise Exception("Invalid RL algorithm type!!")

        agent.evaluate()
        rl_test_cycle = RLAlgorithmTestCycle(
            env=self._env,
            agent=agent,

        )
        results_df = rl_test_cycle.rlAgentTestRunAllCycles()
        print(f"Sample of RL test cycle results with a total of {results_df.shape[0]} rows:")
        print(results_df.head(10))

        pnl_df, reward_df, trading_cost_df, delta_df = rl_test_cycle.aggregateResults(
            aggregation_type=AggregationType.mean
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
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.rewards
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.pnl
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.trading_cost
        )
        rl_test_cycle.plotTwoVariableEvaluationResults(
            plot_type=PlotType.delta
        )

        rl_test_cycle.plotTwoVariableKernelDesityEstimations(
            plot_type=PlotType.rewards
        )
        rl_test_cycle.plotTwoVariableKernelDesityEstimations(
            plot_type=PlotType.pnl
        )
        rl_test_cycle.plotTwoVariableKernelDesityEstimations(
            plot_type=PlotType.trading_cost
        )
        rl_test_cycle.plotTwoVariableKernelDesityEstimations(
            plot_type=PlotType.delta
        )

def main():
    """
    Entry point
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
    model = GenerateRLModelResults(algorithm_type, hedging_type)
    model.run()

if __name__ == "__main__":
    main()
