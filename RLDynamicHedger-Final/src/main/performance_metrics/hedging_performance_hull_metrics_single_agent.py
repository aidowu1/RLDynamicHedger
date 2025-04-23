import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

from src.main.market_simulator.parameters import Parameters
from src.main.market_simulator.simulator import MarketSimulator
from src.main import configs_global as configs
from src.main import configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.market_simulator.simulator_results import (AdjustedPnlProcessResults,
                                                         ClassicalHedgingResults)
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning

plt.rc('xtick', labelsize=configs2.SMALL_SIZE)
plt.rc('ytick', labelsize=configs2.SMALL_SIZE)
#plt.style.use(['science','no-latex'])

class HedgingPerformanceHullMetricsSingleAgent:
    """
    Class used to compute hedging performance "Hull" metrics
    """
    def __init__(
            self,
            hedge_type: HedgingType = HedgingType.gbm,
            algo_type: RLAgorithmType = RLAgorithmType.td3,
            extra_description: Optional[str] = None,
            parameters: Parameters = None
    ):
        """
        Constructor
        :param td3_evaluation_results_df:
        """
        self._algorithm_type = algo_type
        self._hedging_strategies = [
            "DH",
            self._algorithm_type.name.upper(),
        ]
        self._n_records = len(self._hedging_strategies)

        if parameters:
            self._parameters = parameters
        else:
            parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
            self._parameters = Parameters(**parameter_settings_data)

        self._hedge_type = hedge_type
        self._extra_description = "any_use_case" if extra_description is None else extra_description
        self._tuned_model_root_path = None
        result_path = self._createHullMetricsPath()

        self._algo_evaluation_results = Helpers.getRLEvaluationResultsForHullMetricsPerAlgorithmType(
            self._algorithm_type,
            self._hedge_type,
            self._extra_description
        )

        self._bs_delta_array = self._algo_evaluation_results.bs_delta*self._parameters.notional
        self._asset_price_array = self._algo_evaluation_results.asset_price
        self._option_price_array = self._algo_evaluation_results.option_price*self._parameters.notional
        self._rl_delta_array = self._algo_evaluation_results.rl_delta*self._parameters.notional

        self._volatility_strategy = {
            HedgingType.gbm: "Constant",
            HedgingType.sabr: hedge_type.sabr.name.upper(),
            HedgingType.heston: hedge_type.heston.name.title(),
        }
        self._hull_results_folder = self._createHullMetricsPath()
        self._bs_hedging_metric = None
        self._rl_hedging_metric = None


    def computeNotionalAdjustedPnlProcess(
            self,
            underlying_price: np.ndarray,
            option_price: np.ndarray,
            holding: np.ndarray
    ) -> AdjustedPnlProcessResults:
        """
        Calculates the notional-adjusted Accounting PnL process for a portfolio of a short call option,
        the underlying, with proportional trading costs.
        :param underlying_price: Underlying price process
        :param option_price: Option price (adjusted for number of underlying)
        :param holding: Holding process of number of the underlying held at each period
        :return: Outputs are:
                 accounting_pnl : process of Accounting PnL
                 holding_lagged: lagged process of number of underlying held at each period
        """
        # print(f"\n\nCost per traded stock: {self._parameters.cost_per_traded_stock}\n\n")
        results = MarketSimulator.computeNotionalAdjustedPnlProcess_(
            underlying_price,
            option_price * self._parameters.notional,
            holding * self._parameters.notional,
            self._parameters.notional,
            self._parameters.strike_price,
            self._parameters.cost_per_traded_stock
        )
        return results

    def evaluateAgainstClassicalHedging(
            self,
            accounting_pnl: np.ndarray,
            option_price: np.ndarray
    ) -> ClassicalHedgingResults:
        """
        Evaluates selected strategy against classical delta hedging method.
        :param accounting_pnl: Cumulative accounting PnL array of shape (paths, periods)
        :param option_price: Option prices of shape (paths, periods)
        :return: Returns the following outputs:
                evaluation_function: Evaluation function
                percentage_mean_ratio: Percentage mean ratio
                percentage_std_ratio: Percentage std ration
        """
        results = MarketSimulator.evaluateAgainstClassicalHedging_(
            accounting_pnl,
            option_price,
            self._parameters.stdev_coefficient,
            self._parameters.notional
        )

        return results

    def _computeSingleCaseHullMetrics(
            self,
            option_price_array: np.ndarray,
            delta_array: np.ndarray,
            stock_price_array: np.ndarray
    ) -> ClassicalHedgingResults:
        """
        Computes hedging performance for a single use case
        :param option_price_array: Option price
        :param delta_array: Delta
        :param stock_price_array: Asset price
        :return: Tuple of mean and standard deviation of cost function
        """
        adjusted_pnl_results = self.computeNotionalAdjustedPnlProcess(
            stock_price_array,
            option_price_array,
            delta_array)
        hedging_performance_results = self.evaluateAgainstClassicalHedging(
            adjusted_pnl_results.accounting_pnl,
            option_price_array
        )
        return hedging_performance_results


    def tabulateAllHullMetrics(
            self
    ) -> pd.DataFrame:
        """
        Tabulates hedging performance metrics for all use cases using Hull et al metrics
        It is based on Hull et al (2020). "Deep hedging of derivatives using reinforcement learning".
        Journal of Financial Data Science, 3(1), 10–27
        :return: BS and RL Hedging metrics
        """
        self._bs_hedging_metric: ClassicalHedgingResults = self._computeSingleCaseHullMetrics(
            self._option_price_array,
            self._bs_delta_array,
            self._asset_price_array
        )

        self._rl_hedging_metric: ClassicalHedgingResults = self._computeSingleCaseHullMetrics(
            self._option_price_array,
            self._rl_delta_array,
            self._asset_price_array
        )

        metrics_df = self.tabulateResults()
        return metrics_df

    def tabulateResults(self) -> pd.DataFrame:
        """
        Tabulates the computed "Hull" metrics
        :return: Tabulated results
        """

        hedging_type = [self._volatility_strategy[self._hedge_type]] * self._n_records
        mean_cost = [
            self._bs_hedging_metric.percentage_mean_ratio,
            self._rl_hedging_metric.percentage_mean_ratio,
        ]
        std_cost = [
            self._bs_hedging_metric.percentage_std_ratio,
            self._rl_hedging_metric.percentage_std_ratio,
        ]
        Y0_improvement = self.computeY0FunctionImprovement()
        all_results = {
            "Volatility Model": hedging_type,
            "Hedging Strategy": self._hedging_strategies,
            "Mean Cost": np.round(mean_cost,4),
            "Std Cost": np.round(std_cost,4),
            "Objective Function (mean)": np.round(Y0_improvement,4)
        }
        all_results_df = pd.DataFrame(all_results)
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

    def computeY0FunctionImprovement(
            self
    ) -> List[float]:
        """
        Computes the improvement of the Y0 objective function for all the hedging strategies
        :return: Improvement results
        """
        def computeImprovementMetricPerStrategy(strategy_function_value: float):
            """
            Computes the Y0 objective function per strategy
            :param strategy_function_value: Strategy Y0 function value
            :return: Improvement value
            """
            result = (
                    (self._bs_hedging_metric.evaluation_function[0] - strategy_function_value[0])
                    / self._bs_hedging_metric.evaluation_function[0]
            )
            return result

        scaler = self._parameters.notional * self._parameters.start_stock_price
        Y0_improvement_values = [
            np.mean(self._bs_hedging_metric.evaluation_function) / scaler,
            np.mean(self._rl_hedging_metric.evaluation_function) / scaler,
        ]

        return Y0_improvement_values

