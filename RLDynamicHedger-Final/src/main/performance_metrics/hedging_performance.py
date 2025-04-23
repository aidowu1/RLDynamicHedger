import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, List

from src.main.market_simulator.parameters import Parameters
from src.main import configs_global as configs
from src.main import configs_rl as configs2
from src.main.performance_metrics.hedging_metrics import HedgingMetrics, HullHedgingMetrics
from src.main.utility.utils import Helpers
from src.main.utility.enum_types import RLAgorithmType, HedgingType
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning
from src.main.market_simulator.simulator_results import (BlackScholesCallResults, SABRSimulationResults,
                                                         GBMSimulationResults, SABRSimulationRunResults,
                                                         HedgingStrategyResults, AdjustedPnlProcessResults,
                                                         ClassicalHedgingResults, HestonSimulationResults)

plt.rc('xtick', labelsize=configs2.SMALL_SIZE)
plt.rc('ytick', labelsize=configs2.SMALL_SIZE)
#plt.style.use(['science','no-latex'])

class HedgingPerformanceMetrics:
    """
    Class used to compute hedging performance metrics
    """
    def __init__(
            self,
            td3_evaluation_results_df: pd.DataFrame,
            ddpg_evaluation_results_df: pd.DataFrame,
            ppo_evaluation_results_df: pd.DataFrame,
            sac_evaluation_results_df: pd.DataFrame,
            hedge_type: HedgingType = HedgingType.gbm,
            is_plot_to_screen: bool = False,
            model_use_case: str = None,
            parameters: Parameters = None
    ):
        """
        Constructor
        :param td3_evaluation_results_df: TD3 results
        :param ddpg_evaluation_results_df: DDPG results
        :param ppo_evaluation_results_df: PPO results
        :param sac_evaluation_results_df: SAC results
        :param hedge_type: Hedging type
        :param is_plot_to_screen: Whether to show plots
        :param parameters: Parameters
        """
        if parameters:
            self._parameters = parameters
        else:
            parameter_settings_data = Helpers.getParameterSettings(configs.DEFAULT_SETTINGS_NAME)
            self._parameters = Parameters(**parameter_settings_data)

        self._td3_evaluation_results_df = td3_evaluation_results_df
        self._ddpg_evaluation_results_df = ddpg_evaluation_results_df
        self._ppo_evaluation_results_df = ppo_evaluation_results_df
        self._sac_evaluation_results_df = sac_evaluation_results_df
        self._hedge_type = hedge_type
        self._is_plot_to_screen = is_plot_to_screen
        self._model_use_case = "any_use_case" if model_use_case is None else model_use_case
        self._plots_folder = self._createComparativeResultsPlotPath()

    def _getTransactionCosts(
            self,
            delta_change: float
    ):
        """
        Computes the delta hedging transaction cost
        :param delta_change: Change in delta
        :return:
        """
        transaction_cost = self._parameters.tick_size * ( (np.abs(delta_change)
                            + 0.01 * delta_change ** 2) )
        # print(f"\n\nTransaction cost {transaction_cost:.2f} based on tick_size: {self._parameters.tick_size}")
        return transaction_cost

    def _computeSingleCaseMetrics(
            self,
            option_price: List[float],
            delta: List[float],
            stock_price: List[float]
    ) -> Tuple[float, float]:
        """
        Computes hedging performance for a single use case
        :param option_price: Option price
        :param delta: Delta
        :param stock_price
        :return: Tuple of mean and standard deviation of cost function
        """
        bank_cash_amount = [option_price[0] - delta[0] * stock_price[0]]
        portfolio_amount = [option_price[0]]
        for i in range(1, len(option_price)):
            delta_change = delta[i] - delta[i - 1]
            transaction_cost = self._getTransactionCosts(delta_change)
            portfolio_amount.append(delta[i - 1] * stock_price[i] + bank_cash_amount[i - 1])
            bank_cash_amount.append(portfolio_amount[i] - (delta[i] + transaction_cost) * stock_price[i])

        hedge_error = np.array(portfolio_amount) - np.maximum(np.array(stock_price) - self._parameters.strike_price, 0)
        mean_hedge_error = np.mean(hedge_error) / option_price[0]
        std_hedge_error = np.std(hedge_error) / option_price[0]

        return mean_hedge_error, std_hedge_error

    def computeAllMetrics(
            self
    ) -> HedgingMetrics:
        """
        Computes hedging performance metrics for all use cases
        :return: BS and RL Hedging metrics
        """
        bs_mean_hedge_error, bs_std_hedge_error = self._computeSingleCaseMetrics(
            self._td3_evaluation_results_df.current_option_price.tolist(),
            self._td3_evaluation_results_df.bs_delta.tolist(),
            self._td3_evaluation_results_df.current_stock_price.tolist()
        )

        td3_mean_hedge_error, td3_std_hedge_error = self._computeSingleCaseMetrics(
            self._td3_evaluation_results_df.hedge_portfolio_value.tolist(),
            self._td3_evaluation_results_df.rl_delta.tolist(),
            self._td3_evaluation_results_df.current_stock_price.tolist()
        )

        ddpg_mean_hedge_error, ddpg_std_hedge_error = self._computeSingleCaseMetrics(
            self._ddpg_evaluation_results_df.hedge_portfolio_value.tolist(),
            self._ddpg_evaluation_results_df.rl_delta.tolist(),
            self._ddpg_evaluation_results_df.current_stock_price.tolist()
        )

        ppo_mean_hedge_error, ppo_std_hedge_error = self._computeSingleCaseMetrics(
            self._ppo_evaluation_results_df.hedge_portfolio_value.tolist(),
            self._ppo_evaluation_results_df.rl_delta.tolist(),
            self._ppo_evaluation_results_df.current_stock_price.tolist()
        )

        sac_mean_hedge_error, sac_std_hedge_error = self._computeSingleCaseMetrics(
            self._sac_evaluation_results_df.hedge_portfolio_value.tolist(),
            self._sac_evaluation_results_df.rl_delta.tolist(),
            self._sac_evaluation_results_df.current_stock_price.tolist()
        )

        metrics = HedgingMetrics(
            bs_mean_error=bs_mean_hedge_error,
            td3_mean_error=td3_mean_hedge_error,
            ddpg_mean_error=ddpg_mean_hedge_error,
            ppo_mean_error=ppo_mean_hedge_error,
            sac_mean_error=sac_mean_hedge_error,

            bs_std_error=bs_std_hedge_error,
            td3_std_error=td3_std_hedge_error,
            ddpg_std_error=ddpg_std_hedge_error,
            ppo_std_error=ppo_std_hedge_error,
            sac_std_error=sac_std_hedge_error,
        )
        return metrics

    def plotMetrics(
            self,
            metrics: HedgingMetrics,

    ):
        """
        Plots the hedging performance metrics
        :param metrics: Metrics
        :return: None
        """
        hedging_benchmark_details = Helpers.getHedgingBenchmarkName(self._hedge_type)
        plt.figure(figsize=(7, 7))

        x = [
            metrics.bs_mean_error, metrics.td3_mean_error, metrics.ddpg_mean_error,
            metrics.ppo_mean_error, metrics.sac_mean_error]
        y = [metrics.bs_std_error, metrics.td3_std_error, metrics.ddpg_std_error,
             metrics.ppo_std_error, metrics.sac_std_error]

        plt.scatter(x, y, c=['red', 'blue', 'orange', 'green', 'purple'])
        text_labels = [
            hedging_benchmark_details.get("name", "BS"),
            RLAgorithmType.td3.name.upper(),
            RLAgorithmType.ddpg.name.upper(),
            RLAgorithmType.ppo.name.upper(),
            RLAgorithmType.sac.name.upper(),
        ]

        for i, txt in enumerate(text_labels):
            plt.annotate(txt, (x[i], y[i]), xytext=(-8, 5), textcoords="offset points", size=configs2.SMALL_SIZE)

        use_case_text = " ".join(self._model_use_case.split("_"))
        title_map = {
            "gbm": f"Hedging Performance for {use_case_text} (GBM model)",
            "sabr": f"Hedging Performance for {use_case_text} (SABR model)",
            "heston": f"Hedging Performance for {use_case_text} (Heston model)",
        }

        plt.xlabel('Mean of Cost Function', fontdict=configs2.FONT)
        plt.ylabel('Standard Deviation of Cost Function', fontdict=configs2.FONT)
        plt.title(f'{title_map[self._hedge_type.name]}', fontdict=configs2.FONT)
        plt.grid()

        plot_path = f"{self._plots_folder}/comparative_results.png"
        plt.savefig(plot_path)
        if self._is_plot_to_screen:
            plt.show()
        plt.close()
        print(f"Successfully plotted the comparative results and saved the plot here:  {plot_path}")

    def _createComparativeResultsPlotPath(self) -> str:
        """
        Creates the RL comparative results plot path
        :return: Test results path
        """
        tuned_model_root_path = configs2.TUNED_MODEL_PATH.format("model_comparisons", self._model_use_case)
        plots_path = f"{tuned_model_root_path}/{self._hedge_type.name}"
        os.makedirs(plots_path, exist_ok=True)
        return plots_path
