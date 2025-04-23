import pandas as pd
import os
from tqdm import tqdm
import argparse
from typing import Optional

from src.main.performance_metrics.hedging_performance_hull_metrics_single_agent import \
    HedgingPerformanceHullMetricsSingleAgent
from src.main.utility.logging import Logger
from src.main.performance_metrics.hedging_performance_hull_metrics import HedgingPerformanceHullMetrics
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.market_simulator.parameters import Parameters
from src.main import configs_global as configs
from src.main.utility.utils import Helpers

class GenerateHullMetricsResults:
    """
    Class is used to generate the RL agents comparative "Hull" metrics"
    for all the hedging strategies and volatility scenarios (constant and stochastic)
    """
    def __init__(
            self,
            algorithm_type: Optional[RLAgorithmType],
            extra_description: Optional[str] = None,
            is_display_results_on_screen: bool = False,
            parameters: Parameters = None
    ):
        """
        Constructor
        """
        self._algorithm_type = algorithm_type
        self._extra_description = "any_use_case" if extra_description is None else extra_description
        self._is_display_results_on_screen = is_display_results_on_screen
        self._logger = Logger.getLogger()
        self._tuned_model_root_path = None
        result_path = self._createHullMetricsPath()
        self._result_file_name = f"{result_path}/hull_metrics_results.csv"

        if parameters:
            self._parameters = parameters
        else:
            parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
            self._parameters = Parameters(**parameter_settings_data)

        if not algorithm_type:
            self._logger.info(f"Computing the 'Hull' metrics for all the hedging agents (DDPG, TD3, SAC, PPO)"
                              f" and volatility scenarios (constant and stochastic)")
        else:
            self._logger.info(f"Computing the 'Hull' metrics a single hedging agent: {self._algorithm_type.name.upper()}"
                              f" and volatility scenarios (constant and stochastic)")

    def run(self) -> Optional[pd.DataFrame]:
        """
        Run the result generation
        :return: Results dataframe
        """
        hedging_types = [HedgingType.gbm, HedgingType.sabr, HedgingType.heston]
        results_df_list = []
        for hedging_type in tqdm(hedging_types, desc="Hedging types.."):
            if not self._algorithm_type:
                hedging_performance = HedgingPerformanceHullMetrics(
                    hedge_type=hedging_type,
                    parameters=self._parameters
                )
            elif self._extra_description is None:
                hedging_performance = HedgingPerformanceHullMetricsSingleAgent(
                    hedge_type=hedging_type,
                    algo_type=self._algorithm_type,
                    parameters=self._parameters
                )
            else:
                hedging_performance = HedgingPerformanceHullMetricsSingleAgent(
                    hedge_type=hedging_type,
                    algo_type=self._algorithm_type,
                    extra_description=self._extra_description,
                    parameters=self._parameters
                )
            results_df = hedging_performance.tabulateAllHullMetrics()
            results_df_list.append(results_df)
        all_results_df = pd.concat(results_df_list)
        all_results_df.to_csv(self._result_file_name, index=False)
        if self._is_display_results_on_screen:
            return all_results_df

    def _createHullMetricsPath(self) -> str:
        """
        Creates the RL comparative results plot path
        :return: Test results path
        """
        if not self._algorithm_type:
            self._tuned_model_root_path = f"model/trained-tuned-models/all/{self._extra_description}"
        else:
            self._tuned_model_root_path = BaseHyperParameterTuning.createModelRootPath(
                rl_algo_type=self._algorithm_type,
                model_use_case=self._extra_description)
        hull_metrics_path = f"{self._tuned_model_root_path}/hull_metrics_results"
        os.makedirs(hull_metrics_path, exist_ok=True)
        return hull_metrics_path

def main():
    """
    Entry point
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo_type",
        type=str,
        default=None,
        help="RL algorithm type"
    )
    parser.add_argument(
        "--extra_test_description",
        type=str,
        default="None",
        help="Extra description to describe this special evaluation/experiment"
    )
    parser.add_argument(
        "--is_display_results_on_screen",
        type=str,
        default="false",
        help="Flag to indicate the display of all the results on screen"
    )

    args = parser.parse_args()
    if args.algo_type:
        algorithm_type = Helpers.getEnumType(RLAgorithmType, args.algo_type)
    else:
        algorithm_type = None

    print(f"args.algo_type: {args.algo_type}")
    extra_test_description = args.extra_test_description
    is_display_results_on_screen = Helpers.text2Boolean(args.is_display_results_on_screen)
    generator = GenerateHullMetricsResults(
        algorithm_type=algorithm_type,
        extra_description=extra_test_description,
        is_display_results_on_screen=is_display_results_on_screen
    )
    generator.run()


if __name__ == "__main__":
    main()