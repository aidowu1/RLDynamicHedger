import pandas as pd
import os, sys
import argparse
from typing import List, Optional, Union
from tqdm import tqdm
from pathlib import Path

ROOT_FOLDER = f"{os.path.dirname(os.path.abspath(__file__))}\.."
print(f"Root folder: {ROOT_FOLDER}")
sys.path.append(ROOT_FOLDER)

from src.main.utility.logging import Logger
from src.main.performance_metrics.hedging_performance import HedgingPerformanceMetrics
from src.main.utility.enum_types import RLAgorithmType, HedgingType
import src.main.configs_rl as configs2
import src.main.configs_global as configs
from src.main.utility.utils import Helpers
from src.main.performance_metrics.hedging_metrics import HedgingMetrics
from src.main.market_simulator.parameters import Parameters

class GenerateComparativePerformanceResults:
    """
    Class is used to generate the RL agents comparative performance results
    for each of the 3 hedging types
    """
    def __init__(
            self,
            hedge_type: Optional[HedgingType],
            current_simulation_path_index: int = 0,
            is_plot_to_screen: bool = False,
            model_use_case: str = None,
            parameters: Parameters = None
    ):
        """
        Constructor
        :param hedge_type: Hedging type
        :param current_simulation_path_index: Current episode
        :param is_plot_to_screen: Whether the plot should be shown
        :param model_use_case: Model use-case
        :param parameters: Parameters
        """
        self._logger = Logger().getLogger()
        self._hedge_type = hedge_type
        self._current_simulation_path_index = current_simulation_path_index
        self._is_plot_to_screen = is_plot_to_screen
        self._problem_title: str = configs2.RL_PROBLEM_TITLE
        self._model_use_case = "any_use_case" if model_use_case is None else model_use_case

        if parameters:
            self._parameters = parameters
        else:
            parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
            self._parameters = Parameters(**parameter_settings_data)

        self._volatility_models = [HedgingType.gbm, HedgingType.sabr, HedgingType.heston]

    def run(
            self,
            hedging_type: HedgingType,
    ) -> HedgingMetrics:
        """
        Runs the plotting of the RL agents comparative performance results
        :param hedging_type: RL algorithm type
        :return: Hedging metrics per volatility model use cases
        """
        self._td3_evaluation_results_df = self._getRLEvaluationResultsPerAlgorithmType(
            RLAgorithmType.td3,
            hedging_type
        )
        self._ddpg_evaluation_results_df = self._getRLEvaluationResultsPerAlgorithmType(
            RLAgorithmType.ddpg,
            hedging_type
        )
        self._ppo_evaluation_results_df = self._getRLEvaluationResultsPerAlgorithmType(
            RLAgorithmType.ppo,
            hedging_type
        )
        self._sac_evaluation_results_df = self._getRLEvaluationResultsPerAlgorithmType(
            RLAgorithmType.sac,
            hedging_type
        )

        hedging_performance = HedgingPerformanceMetrics(
            td3_evaluation_results_df=self._td3_evaluation_results_df,
            ddpg_evaluation_results_df=self._ddpg_evaluation_results_df,
            ppo_evaluation_results_df=self._ppo_evaluation_results_df,
            sac_evaluation_results_df=self._sac_evaluation_results_df,
            hedge_type=hedging_type,
            is_plot_to_screen=self._is_plot_to_screen,
            model_use_case=self._model_use_case,
            parameters=self._parameters
        )
        hedging_metrics = hedging_performance.computeAllMetrics()
        hedging_performance.plotMetrics(hedging_metrics)
        return hedging_metrics

    def runAll(self) -> List[HedgingMetrics]:
        """
        Runs the plotting of the RL agents comparative performance results for all volatility model use cases
        :return: Hedging metrics for all the volatility model use cases
        """
        all_metrics = []
        for hedging_type in tqdm(self._volatility_models, desc="Volatility models.."):
            metrics = self.run(hedging_type)
            all_metrics.append(metrics)
        return all_metrics

    def execute(self) -> Union[HedgingMetrics, List[HedgingMetrics]]:
        """
        Execute the results generation
        :return: Hedging metrics for a single or all the volatility model use cases
        """
        if self._hedge_type:
            result = self.run(self._hedge_type)
        else:
            result = self.runAll()
        return result

    def _getEvaluationResultsPath(
            self,
            algorithm_type: RLAgorithmType,
            hedging_type: HedgingType
    ):
        """
        Gets the RL evaluation results path
        :param algorithm_type: Algorithm type
        :param hedging_type: Hedging type
        :return:
        """
        results_path = Helpers.getEvaluationResultsPath(
            algorithm_type=algorithm_type,
            hedging_type=hedging_type,
            extra_description=self._model_use_case
        )
        return results_path

    def _getRLEvaluationResultsPerAlgorithmType(
            self,
            algorithm_type: RLAgorithmType,
            hedging_type: HedgingType,
    ) -> pd.DataFrame:
        """
        Gets RL evaluation results for all algorithms
        :param algorithm_type: Algorithm type
        :param hedging_type: Hedging type
        :return:
        """
        results_path = self._getEvaluationResultsPath(algorithm_type, hedging_type)
        try:
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path,index_col=False)
                episode_filter = results_df.simulation_path == self._current_simulation_path_index
                episode_results_df = results_df[episode_filter]
                return episode_results_df
            else:
                raise Exception(f"{results_path} does not exist")
        except Exception as ex:
            self._logger.info(f"Exception: {ex}")

def main():
    """
    Entry point to run the generation of RL comparative performance results for each hedging type
    :return: None
    """
    logger = Logger.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hedging_type",
        type=str,
        default=None,
        help="Hedging strategy type"
    )
    parser.add_argument(
        "--sim_path_index",
        type=str,
        default="0",
        help="Current simulation path index"
    )
    parser.add_argument(
        "--model_use_case",
        type=str,
        default=configs2.DEFAULT_MODEL_USE_CASE,
        help="Model use case description"
    )

    args = parser.parse_args()
    print("Start of comparative RL agent result generations\n")
    print("Using command line arguments:")
    print(f"hedging_type: {args.hedging_type}")
    print(f"simulation path index: {args.sim_path_index}")
    print(f"simulation use case: {args.model_use_case}")
    print("\nThis will take a few seconds to complete..\n")
    if args.hedging_type:
        hedging_type = Helpers.getEnumType(HedgingType, args.hedging_type)
    else:
        hedging_type = None
    simulation_path_index = int(args.sim_path_index)
    is_plot_to_screen = True
    model_use_case = args.model_use_case
    model = GenerateComparativePerformanceResults(
        hedging_type,
        simulation_path_index,
        is_plot_to_screen,
        model_use_case
    )
    result = model.execute()
    logger.info("End of the result generation..")


if __name__ == "__main__":
    main()


