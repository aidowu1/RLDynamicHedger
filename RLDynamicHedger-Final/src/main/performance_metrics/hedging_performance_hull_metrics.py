import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, List

from src.main.market_simulator.parameters import Parameters
from src.main import configs_global as configs
from src.main import configs_rl as configs2
from src.main.utility.utils import Helpers
from src.main.market_simulator.simulator import MarketSimulator
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.market_simulator.simulator_results import (AdjustedPnlProcessResults,
                                                         ClassicalHedgingResults)

plt.rc('xtick', labelsize=configs2.SMALL_SIZE)
plt.rc('ytick', labelsize=configs2.SMALL_SIZE)
#plt.style.use(['science','no-latex'])

class HedgingPerformanceHullMetrics:
    """
    Class used to compute hedging performance "Hull" metrics
    """
    def __init__(
            self,
            hedge_type: HedgingType = HedgingType.gbm,
            parameters: Parameters = None
    ):
        """
        Constructor
        :param td3_evaluation_results_df:
        """
        self._hedging_strategies = [
            "bs",
            RLAgorithmType.ddpg.name,
            RLAgorithmType.td3.name,
            RLAgorithmType.sac.name,
            RLAgorithmType.ppo.name
        ]
        self._n_records = len(self._hedging_strategies)

        if parameters:
            self._parameters = parameters
        else:
            parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
            self._parameters = Parameters(**parameter_settings_data)

        # parameter_settings_filename: str = configs.DEFAULT_SETTINGS_NAME
        # parameter_settings_data = Helpers.getParameterSettings(parameter_settings_filename)
        # self._parameters = Parameters(**parameter_settings_data)

        self._hedge_type = hedge_type

        self._ddpg_evaluation_results = Helpers.getRLEvaluationResultsForHullMetricsPerAlgorithmType(
            RLAgorithmType.ddpg,
            self._hedge_type
        )
        self._td3_evaluation_results = Helpers.getRLEvaluationResultsForHullMetricsPerAlgorithmType(
            RLAgorithmType.td3,
            self._hedge_type
        )
        self._sac_evaluation_results = Helpers.getRLEvaluationResultsForHullMetricsPerAlgorithmType(
            RLAgorithmType.sac,
            self._hedge_type
        )
        self._ppo_evaluation_results = Helpers.getRLEvaluationResultsForHullMetricsPerAlgorithmType(
            RLAgorithmType.ppo,
            self._hedge_type
        )

        self._bs_delta_array = self._ddpg_evaluation_results.bs_delta
        self._asset_price_array = self._ddpg_evaluation_results.asset_price
        self._option_price_array = self._ddpg_evaluation_results.option_price
        self._ddpg_delta_array = self._ddpg_evaluation_results.rl_delta
        self._td3_delta_array = self._td3_evaluation_results.rl_delta
        self._sac_delta_array = self._sac_evaluation_results.rl_delta
        self._ppo_delta_array = self._ppo_evaluation_results.rl_delta

        self._volatility_strategy = {
            HedgingType.gbm: "Constant",
            HedgingType.sabr: hedge_type.sabr.name,
            HedgingType.heston: hedge_type.heston.name,
        }

        self._hull_results_folder = self._createHullMetricsPath()

        self._bs_hedging_metric = None
        self._ddpg_hedging_metric = None
        self._td3_hedging_metric = None
        self._sac_hedging_metric = None
        self._ppo_hedging_metric = None

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
        # create lagged variables for APL
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

        self._td3_hedging_metric: ClassicalHedgingResults = self._computeSingleCaseHullMetrics(
            self._option_price_array,
            self._td3_delta_array,
            self._asset_price_array
        )

        self._ddpg_hedging_metric: ClassicalHedgingResults = self._computeSingleCaseHullMetrics(
            self._option_price_array,
            self._ddpg_delta_array,
            self._asset_price_array
        )

        self._ppo_hedging_metric: ClassicalHedgingResults = self._computeSingleCaseHullMetrics(
            self._option_price_array,
            self._ppo_delta_array,
            self._asset_price_array
        )

        self._sac_hedging_metric: ClassicalHedgingResults = self._computeSingleCaseHullMetrics(
            self._option_price_array,
            self._sac_delta_array,
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
            self._ddpg_hedging_metric.percentage_mean_ratio,
            self._td3_hedging_metric.percentage_mean_ratio,
            self._sac_hedging_metric.percentage_mean_ratio,
            self._ppo_hedging_metric.percentage_mean_ratio
        ]
        std_cost = [
            self._bs_hedging_metric.percentage_std_ratio,
            self._ddpg_hedging_metric.percentage_std_ratio,
            self._td3_hedging_metric.percentage_std_ratio,
            self._sac_hedging_metric.percentage_std_ratio,
            self._ppo_hedging_metric.percentage_std_ratio
        ]
        Y0_improvement = self.computeY0FunctionImprovement()
        all_results = {
            "Volatility Model": hedging_type,
            "Hedging Strategy": self._hedging_strategies,
            "Mean Cost": mean_cost,
            "Std Cost": std_cost,
            "Objective Function Y": Y0_improvement
        }
        all_results_df = pd.DataFrame(all_results)
        return all_results_df

    def _createHullMetricsPath(self) -> str:
        """
        Creates the RL comparative results plot path
        :return: Test results path
        """
        joined_title = "_".join(configs2.RL_PROBLEM_TITLE.split())
        log_path = f"./logs/{joined_title}_perf_comparative_results"
        plots_path = f"{log_path}/{self._hedge_type.name}"
        os.makedirs(plots_path, exist_ok=True)
        return plots_path

    def computeY0FunctionImprovement(
            self

    ) -> List[float]:
        """
        Computes the improvement of the Y0 objective function for all the hedging strategies
        :return: Improvement results
        """
        scaler = self._parameters.notional * self._parameters.start_stock_price
        Y0_improvement_values = [
            np.mean(self._bs_hedging_metric.evaluation_function) / scaler,
            np.mean(self._ddpg_hedging_metric.evaluation_function) / scaler,
            np.mean(self._td3_hedging_metric.evaluation_function) / scaler,
            np.mean(self._sac_hedging_metric.evaluation_function) / scaler,
            np.mean(self._ppo_hedging_metric.evaluation_function) / scaler,
        ]
        return Y0_improvement_values

